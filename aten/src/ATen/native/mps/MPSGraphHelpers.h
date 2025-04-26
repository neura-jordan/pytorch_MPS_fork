#pragma once

// Framework imports (Objective-C)
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensor.h>
#endif

// C++ includes
#include <ATen/ATen.h>

#ifndef __OBJC__
struct MPSShape;
using MPSDataType = unsigned int;
#endif

namespace at::native::mps {

constexpr int MPS_MAX_RANK = 16;

// Helper functions (defined in MPSGraphHelpers.mm)
MPSShape* MPSShapeFromSizes(const c10::IntArrayRef& sizes, int rank);
MPSDataType MPSDataTypeFromScalarType(at::ScalarType scalar_type);

// Tensor <-> MPS conversions (defined in MPSGraphHelpers.mm)
#ifdef __OBJC__
MPSGraphTensor* TorchTensorToMPSPlaceholder(MPSGraph* g,
                                            const at::Tensor& t,
                                            int rank,
                                            const char* name = nullptr);

id<MTLBuffer> TensorToBuffer(const at::Tensor& t);
#endif

// Utility helpers
void printTensorMetaData(const at::Tensor& t, bool printData = false, const char* name = nullptr);
void printTensorShape(const at::Tensor& t, const char* name = nullptr);

void checkSupportsComplex();
void checkSupportsBFloat16();

} // namespace at::native::mps
