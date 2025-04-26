// Copyright (c) 2025 PyTorch contributors
// SPDX-License-Identifier: BSD-3-Clause

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphConvolutionOps.h>
#include <ATen/ATen.h>
#include <ATen/native/TensorProperties.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <ATen/native/mps/MPSGraphHelpers.h>
#include <torch/library.h>   // for TORCH_LIBRARY_IMPL and dispatcher helpers
#include <c10/util/Optional.h>

using namespace at::native::mps;
using namespace at::mps;

// ------------- helper to run a tiny graph -----------------
static void run_graph(MPSGraph*                 graph,
                      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                      MPSGraphTensor*           fetch,
                      at::Tensor&               out) {
  // Execute the graph in one shot using the queue from the current MPS stream.
  id<MTLCommandQueue> q = at::mps::getCurrentMPSStream()->commandQueue();

  // Run and obtain the result map.  We use the simpler convenience
  // `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` API so we
  // don't have to craft an MPSGraphExecutionDescriptor or a command buffer
  // manually.
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
      [graph runWithMTLCommandQueue:q
                              feeds:feeds
                       targetTensors:@[ fetch ]
                    targetOperations:nil];

  // Copy the result into the output tensor.
  MPSGraphTensorData* resultData = results[fetch];
  MPSNDArray*         ndarray    = resultData.mpsndarray;

  // Read the NDArray's contents directly into the tensor's buffer.
  // (strideBytes == nullptr ⇒ contiguous copy)
  [ndarray readBytes:out.data_ptr() strideBytes:nullptr];
}
// ----------------------------------------------------------

// -------------- helper for tensor -> MPSGraphTensorData -------------
static MPSGraphTensorData* tensorToTensorData(const at::Tensor& t) {
  // Build shape array
  NSMutableArray* shape = [NSMutableArray arrayWithCapacity:t.dim()];
  for (int i = 0; i < t.dim(); ++i) {
    [shape addObject:@(t.size(i))];
  }

  // Describe the NDArray that will view the tensor's storage
  MPSDataType dtype = MPSDataTypeFromScalarType(t.scalar_type());
  // Older macOS runtimes (≤ 15.2) don’t expose
  // +descriptorWithDataType:shape:count:, so fall back to the
  // more widely‑available +descriptorWithShape:dataType: variant.
  MPSNDArrayDescriptor* desc =
      [MPSNDArrayDescriptor descriptorWithShape:shape
                                       dataType:dtype];
  MPSNDArray* ndarray =
      [[MPSNDArray alloc] initWithDevice:at::mps::MPSDevice::getInstance()->device() descriptor:desc];

  // Copy tensor data into the NDArray
  [ndarray writeBytes:t.data_ptr()];
  return [[[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray] autorelease];
}
// -------------------------------------------------------------------

namespace at::native {

// Prototype for dispatcher linkage (extern)
TORCH_API Tensor conv_transpose3d_mps(
        const Tensor& self,
        const Tensor& weight,
        const c10::optional<Tensor>& bias,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef output_padding,
        int64_t groups,
        IntArrayRef dilation);

// Calculated output size for transposed convolution
// Based on the formula from ConvUtils.h
static std::vector<int64_t> calc_output_size(
        const IntArrayRef& input_size,
        const IntArrayRef& weight_size,
        const IntArrayRef& padding,
        const IntArrayRef& output_padding,
        const IntArrayRef& stride,
        const IntArrayRef& dilation,
        int64_t groups) {
    
    auto dim = input_size.size() - 2;
    std::vector<int64_t> output_size(input_size.begin(), input_size.begin() + 2);
    output_size[1] = weight_size[1] * groups; // output channels
    
    for (const auto d : c10::irange(dim)) {
        auto kernel = dilation[d] * (weight_size[d + 2] - 1) + 1;
        auto tmp = input_size[d + 2] - 1 + kernel - 2 * padding[d];
        output_size.push_back(stride[d] * tmp + output_padding[d]);
    }
    
    return output_size;
}

// Add helper to fill 3D conv descriptor (copied from Convolution.mm)
// ------------- helper to program 3D conv descriptor -----------------
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;
  descriptor_.dataLayout = MPSGraphTensorNamedDataLayoutNDHWC;  // NDHWC
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIDHW; // OIDHW
  descriptor_.groups = groups;
}
// ----------------------------------------------------------

Tensor conv_transpose3d_mps(
        const Tensor& self,
        const Tensor& weight,
        const c10::optional<Tensor>& bias,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef output_padding,
        int64_t groups,
        IntArrayRef dilation) {

  TORCH_CHECK(!bias.has_value(), "MPS ConvTranspose3d: bias parameter not yet supported");
  TORCH_CHECK(groups == 1, "MPS ConvTranspose3d currently only supports groups==1");
  TORCH_CHECK(self.device().is_mps() && weight.device().is_mps(),
              "inputs must be on MPS");

  // Compute output shape (N, Cout, D, H, W)
  auto out_sizes = calc_output_size(
      self.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
  // Compute output tensor
  Tensor output = at::empty(out_sizes, self.options().device(kMPS));

  @autoreleasepool {
      // 1. Prepare tensors for MPSGraph (convert from NCDHW to NDHWC format)
      auto x = self.contiguous();
      auto x_ndhwc = x.permute({0, 2, 3, 4, 1});

      // Permute weights from (Cin, Cout/groups, kD, kH, kW) -> (Cout, Cin, kD, kH, kW)
      auto w = weight.contiguous();
      auto w_oidhw = w.permute({1, 0, 2, 3, 4});

      // 2. Initialize the graph
      MPSGraph* graph = [[MPSGraph alloc] init];

      // 3. Create placeholder tensors
      MPSGraphTensor* gx = TorchTensorToMPSPlaceholder(graph, x_ndhwc, /*rank=*/5, "x");
      MPSGraphTensor* gw = TorchTensorToMPSPlaceholder(graph, w_oidhw, /*rank=*/5, "w");

      // 4. Configure convolution descriptor
      MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
      fill_conv3d_desc(conv3dDescriptor_,
                       /*strideInX*/ stride[2],
                       /*strideInY*/ stride[1],
                       /*strideInZ*/ stride[0],
                       /*dilationRateInX*/ dilation[2],
                       /*dilationRateInY*/ dilation[1],
                       /*dilationRateInZ*/ dilation[0],
                       /*paddingHorizontal*/ padding[2],
                       /*paddingVertical*/ padding[1],
                       /*paddingDepth*/ padding[0],
                       /*groups*/ groups);

      // 5. Create convolutionTranspose3D operation using native API
      NSArray* outputShapeArr = @[
          @(out_sizes[0]),   // N
          @(out_sizes[2]),   // D
          @(out_sizes[3]),   // H
          @(out_sizes[4]),   // W
          @(out_sizes[1])    // C_out
      ];

      // Use the data gradient API instead of the transposed convolution API
      // (mathematically equivalent, but available on more macOS versions)
      MPSGraphTensor* gy_ndhwc =
        [graph convolution3DDataGradientWithIncomingGradientTensor:gx
                                                    weightsTensor:gw
                                                      outputShape:outputShapeArr
                                     forwardConvolutionDescriptor:conv3dDescriptor_
                                                            name:@"convT3D"];

      // 6. Prepare feeds and run the graph
      auto feeds = @{
        gx : tensorToTensorData(x_ndhwc),
        gw : tensorToTensorData(w_oidhw)
      };
      
      // Create a tensor to hold the NDHWC result
      auto y_ndhwc = at::empty({out_sizes[0], out_sizes[2], out_sizes[3], out_sizes[4], out_sizes[1]}, 
                              self.options().device(kMPS));
      
      // Run the graph
      run_graph(graph, feeds, gy_ndhwc, y_ndhwc);
      
      // 7. Convert from NDHWC back to NCDHW format
      output = y_ndhwc.permute({0, 4, 1, 2, 3});
      
      [graph release];
  }
  
  return output;
#else
  TORCH_CHECK(false,
      "3D transposed convolution on MPS requires macOS 14+ / Xcode 15+.");
#endif
}

// Register with dispatcher 
TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("conv_transpose3d.input", conv_transpose3d_mps);
}

}  // namespace at::native