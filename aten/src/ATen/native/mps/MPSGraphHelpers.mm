// Copyright (c) 2025 PyTorch contributors
// SPDX-License-Identifier: BSD-3-Clause

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/MPSGraphHelpers.h>

namespace at::native::mps {

// Helper functions that need full namespace qualification
TORCH_API void checkSupportsBFloat16();
TORCH_API void checkSupportsComplex();

// --- Capability checks ----------------------------------------------------
#
TORCH_API void checkSupportsBFloat16() {
#if defined(__OBJC__)
  // All Apple Silicon GPUs running macOS 14+ support BF16 for MPSGraph;
  // earlier versions do not.  Bail out gracefully if unavailable.
  if (@available(macOS 14.0, *)) {
    return; // Supported – nothing to do.
  }
#endif
  TORCH_CHECK(false,
              "BFloat16 tensors/operations are not supported on this device "
              "or macOS version for the MPS backend.");
}

TORCH_API void checkSupportsComplex() {
  // Complex dtypes are not implemented in the MPS backend yet.
  TORCH_CHECK(false,
              "Complex tensor support is not yet implemented for the "
              "MPS backend.");
}
// -------------------------------------------------------------------------

// --- Mappings ---

static MPSShape* MPSShapeFromSizesImpl(const c10::IntArrayRef& sizes, int rank) {
  TORCH_CHECK(rank <= MPS_MAX_RANK, "Input tensor rank (", rank, ") exceeds maximum (", MPS_MAX_RANK, ")");
  NSMutableArray<NSNumber*>* dims = [NSMutableArray arrayWithCapacity:rank];
  for (const auto i : c10::irange(rank)) {
    [dims addObject:[NSNumber numberWithInteger:sizes[i]]];
  }
  // Note: MPSShape is just a typedef for NSArray<NSNumber*>
  return dims;
}

MPSShape* MPSShapeFromSizes(const c10::IntArrayRef& sizes, int rank) {
  return MPSShapeFromSizesImpl(sizes, rank);
}

static MPSDataType MPSDataTypeFromScalarTypeImpl(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:  return MPSDataTypeFloat32;
    case at::ScalarType::Half:   return MPSDataTypeFloat16;
    case at::ScalarType::BFloat16:  checkSupportsBFloat16(); return MPSDataTypeBFloat16;
    case at::ScalarType::Int:    return MPSDataTypeInt32;
    case at::ScalarType::Long:   return MPSDataTypeInt64;
    case at::ScalarType::Short:  return MPSDataTypeInt16;
    case at::ScalarType::Char:   return MPSDataTypeInt8;
    case at::ScalarType::Byte:   return MPSDataTypeUInt8;
    case at::ScalarType::Bool:   return MPSDataTypeBool;
    case at::ScalarType::ComplexHalf: checkSupportsComplex(); return MPSDataTypeComplexFloat16;
    case at::ScalarType::ComplexFloat: checkSupportsComplex(); return MPSDataTypeComplexFloat32;
    default:
      TORCH_CHECK_TYPE(false, "Unsupported scalar type for MPS backend: ", scalar_type);
  }
}

MPSDataType MPSDataTypeFromScalarType(at::ScalarType scalar_type) {
  return MPSDataTypeFromScalarTypeImpl(scalar_type);
}

#ifdef __OBJC__
MPSGraphTensor* TorchTensorToMPSPlaceholder(MPSGraph* g, const at::Tensor& t, int rank, const char* name) {
  auto* shape = MPSShapeFromSizes(t.sizes(), rank);
  MPSDataType dtype = MPSDataTypeFromScalarType(t.scalar_type());
  NSString* nsName = name ? @(name) : nil;
  return [g placeholderWithShape:shape dataType:dtype name:nsName];
}

id<MTLBuffer> TensorToBuffer(const at::Tensor& t) {
  TORCH_CHECK(t.is_mps(), "Tensor must be on MPS device");
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  void* ptr = t.numel() ? t.data_ptr() : reinterpret_cast<void*>(1);
  return [device newBufferWithBytesNoCopy:ptr
                                   length:t.nbytes()
                                  options:MTLResourceStorageModeShared
                              deallocator:nil];
}
#endif // __OBJC__

} // namespace at::native::mps 