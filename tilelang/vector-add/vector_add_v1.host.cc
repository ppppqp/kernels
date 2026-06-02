// tilelang target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[8];
  void* stack_ffi_any = stack;
  if (!((num_args == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel: num_args should be 3", (long long)(num_args), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel: args pointer is NULL");
    return -1;
  }
  int32_t a_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((a_handle_type_index == 0) || (a_handle_type_index == 4)) || (a_handle_type_index == 7)) || (64 <= a_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input a expected pointer or tensor handle");
    return -1;
  }
  int32_t b_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((b_handle_type_index == 0) || (b_handle_type_index == 4)) || (b_handle_type_index == 7)) || (64 <= b_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input b expected pointer or tensor handle");
    return -1;
  }
  int32_t c_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((c_handle_type_index == 0) || (c_handle_type_index == 4)) || (c_handle_type_index == 7)) || (64 <= c_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input c expected pointer or tensor handle");
    return -1;
  }
  void* a_handle = ((a_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* b_handle = ((b_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* c_handle = ((c_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  bool kernel_a_is_null = (a_handle == NULL);
  if (!(!kernel_a_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.a is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_b_is_null = (b_handle == NULL);
  if (!(!kernel_b_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.b is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_c_is_null = (c_handle == NULL);
  if (!(!kernel_c_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.c is expected to have non-NULL pointer");
    return -1;
  }
  void* kernel_a_shape = (((DLTensor*)a_handle)[0].shape);
  void* kernel_b_shape = (((DLTensor*)b_handle)[0].shape);
  void* kernel_c_shape = (((DLTensor*)c_handle)[0].shape);
  if (!(((((DLTensor*)a_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input a ndim mismatch, expected 1", (long long)((((DLTensor*)a_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_a_strides = (((DLTensor*)a_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)a_handle)[0].device.device_id);
  void* a = (((DLTensor*)a_handle)[0].data);
  if (!(((((DLTensor*)b_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input b ndim mismatch, expected 1", (long long)((((DLTensor*)b_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_b_strides = (((DLTensor*)b_handle)[0].strides);
  void* b = (((DLTensor*)b_handle)[0].data);
  if (!(((((DLTensor*)c_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input c ndim mismatch, expected 1", (long long)((((DLTensor*)c_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_c_strides = (((DLTensor*)c_handle)[0].strides);
  void* c = (((DLTensor*)c_handle)[0].data);
  if (!(((((((DLTensor*)a_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)a_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)a_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input a dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_a_shape)[0]) == 1048576))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input a shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_a_shape)[0])), (long long)(1048576));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((kernel_a_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)kernel_a_strides)[0]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((kernel_a_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)kernel_a_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input a strides[0] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)a_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input a byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)a_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)a_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input a device_type mismatch, expected cuda", (long long)((((DLTensor*)a_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(a == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input a data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)b_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)b_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)b_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input b dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_b_shape)[0]) == 1048576))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input b shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_b_shape)[0])), (long long)(1048576));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((kernel_b_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)kernel_b_strides)[0]);
  }
  if (!((condval_2 == 1))) {
    int32_t condval_3;
    if ((kernel_b_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)kernel_b_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input b strides[0] violates packed ABI constraint", (long long)(condval_3), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)b_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input b byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)b_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_id) == (((DLTensor*)a_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input b device_id violates packed ABI constraint", (long long)((((DLTensor*)b_handle)[0].device.device_id)), (long long)((((DLTensor*)a_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)b_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input b device_type mismatch, expected cuda", (long long)((((DLTensor*)b_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(b == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input b data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)c_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)c_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)c_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input c dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_c_shape)[0]) == 1048576))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input c shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_c_shape)[0])), (long long)(1048576));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((kernel_c_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)kernel_c_strides)[0]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((kernel_c_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)kernel_c_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input c strides[0] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)c_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input c byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)c_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)c_handle)[0].device.device_id) == (((DLTensor*)a_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input c device_id violates packed ABI constraint", (long long)((((DLTensor*)c_handle)[0].device.device_id)), (long long)((((DLTensor*)a_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)c_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input c device_type mismatch, expected cuda", (long long)((((DLTensor*)c_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(c == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input c data pointer is NULL");
    return -1;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_1;
  result_1.type_index = kTVMFFINone;
  result_1.zero_padding = 0;
  result_1.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
    return -1;
  }
  if (a == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = a;
  if (b == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = b;
  if (c == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = c;
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)4096);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)0;
  if (kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "kernel_kernel", &kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 7, &result_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_kernel(self, args, num_args, result);
}
