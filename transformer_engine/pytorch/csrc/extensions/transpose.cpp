/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

void fused_cast_transpose(at::Tensor input, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                          at::Tensor input_cast, at::Tensor input_transpose,
                          transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cast_cu =
      makeTransformerEngineTensor(input_cast.data_ptr(), {M, N}, otype, amax.data_ptr(),
                                  scale.data_ptr(), scale_inv.data_ptr());
  auto output_transpose_cu =
      makeTransformerEngineTensor(input_transpose.data_ptr(), {N, M}, otype, amax.data_ptr(),
                                  scale.data_ptr(), scale_inv.data_ptr());

  nvte_cast_transpose(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                      at::cuda::getCurrentCUDAStream());
}

void fused_cast_transpose_noop(at::Tensor input, at::Tensor noop, at::Tensor scale, at::Tensor amax,
                               at::Tensor scale_inv, at::Tensor input_cast,
                               at::Tensor input_transpose, transformer_engine::DType otype,
                               int scale_offset, int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto input_cu = makeTransformerEngineTensor(input);
  auto noop_cu = makeTransformerEngineTensor(noop);
  auto output_cast_cu = makeTransformerEngineTensor(input_cast.data_ptr(), {M, N}, otype, amax_dptr,
                                                    scale_dptr, scale_inv_dptr);
  auto output_transpose_cu = makeTransformerEngineTensor(input_transpose.data_ptr(), {N, M}, otype,
                                                         amax_dptr, scale_dptr, scale_inv_dptr);

  // Launch kernel
  nvte_cast_transpose_with_noop(input_cu.data(), noop_cu.data(), output_cast_cu.data(),
                                output_transpose_cu.data(), at::cuda::getCurrentCUDAStream());
}

std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                   at::Tensor amax, at::Tensor scale_inv,
                                                   transformer_engine::DType otype,
                                                   int scale_offset, int amax_offset,
                                                   int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Allocate output tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto grad_output_cast =
      allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto grad_output_transpose =
      allocateTorchTensor(grad_output.size(1), grad_output.size(0), DType::kByte);

  // Return immediately if tensors are empty
  if (M == 0 || N == 0) {
    return {grad_bias.zero_(), grad_output_cast, grad_output_transpose};
  }

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu = makeTransformerEngineTensor(grad_output_cast.data_ptr(), {M, N}, otype,
                                                    amax_dptr, scale_dptr, scale_inv_dptr);
  auto transposed_output_cu = makeTransformerEngineTensor(
      grad_output_transpose.data_ptr(), {N, M}, otype, amax_dptr, scale_dptr, scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(), transposed_output_cu.data(),
                            dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(), transposed_output_cu.data(),
                            dbias_cu.data(), workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_cast, grad_output_transpose};
}

std::vector<at::Tensor> fused_fp8_transpose_bgrad(at::Tensor grad_output, at::Tensor scale,
                                                  at::Tensor amax, at::Tensor scale_inv,
                                                  transformer_engine::DType otype,
                                                  transformer_engine::DType grad_bias_type,
                                                  int scale_offset, int amax_offset,
                                                  int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_bias_type);
  auto grad_output_transpose =
      allocateTorchTensor(grad_output.size(1), grad_output.size(0), DType::kByte);
  auto input_cu = makeTransformerEngineTensor(grad_output.data_ptr(), {M, N}, otype, amax_dptr,
                                              scale_dptr, scale_inv_dptr);
  auto transposed_output_cu = makeTransformerEngineTensor(
      grad_output_transpose.data_ptr(), {N, M}, otype, amax_dptr, scale_dptr, scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_fp8_transpose_dbias(input_cu.data(), transposed_output_cu.data(), dbias_cu.data(),
                           workspace.data(), at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_fp8_transpose_dbias(input_cu.data(), transposed_output_cu.data(), dbias_cu.data(),
                           workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_transpose};
}

std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input, at::Tensor scale,
                                                         at::Tensor amax, at::Tensor scale_inv,
                                                         transformer_engine::DType otype,
                                                         int scale_offset, int amax_offset,
                                                         int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dgelu = allocateTorchTensor(grad_output.size(0), grad_output.size(1), DType::kByte);
  auto dgelu_transpose =
      allocateTorchTensor(grad_output.size(1), grad_output.size(0), DType::kByte);
  auto gelu_input_cu = makeTransformerEngineTensor(gelu_input);
  auto input_cu = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu = makeTransformerEngineTensor(dgelu.data_ptr(), {M, N}, otype, amax_dptr,
                                                    scale_dptr, scale_inv_dptr);
  auto transposed_output_cu = makeTransformerEngineTensor(dgelu_transpose.data_ptr(), {N, M}, otype,
                                                          amax_dptr, scale_dptr, scale_inv_dptr);
  auto dbias_cu = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(), cast_output_cu.data(),
                                  transposed_output_cu.data(), dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(), cast_output_cu.data(),
                                  transposed_output_cu.data(), dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  return {grad_bias, dgelu, dgelu_transpose};
}

void fused_dswiglu_cast_transpose(at::Tensor grad_output, at::Tensor input, at::Tensor grad_input,
                                  at::Tensor grad_input_transpose, at::Tensor scale,
                                  at::Tensor amax, at::Tensor scale_inv,
                                  transformer_engine::DType otype, int scale_offset,
                                  int amax_offset, int scale_inv_offset) {
  using namespace transformer_engine;

  // Tensor dimensions
  auto outer_dim = [](const at::Tensor& tensor) -> size_t {
    return tensor.numel() / tensor.size(-1);
  };
  const auto M = outer_dim(grad_output);
  const auto N = static_cast<size_t>(grad_output.size(-1));

  // Check tensor dims
  NVTE_CHECK(grad_output.dim() == 2, "Expected grad output tensor to have 2 dims, but found ",
             grad_output.dim());
  NVTE_CHECK(input.dim() == 2, "Expected input tensor to have 2 dims, but found ", input.dim());
  NVTE_CHECK(outer_dim(input) == M, "Expected input tensor to have outer dimension of ", M,
             ", but found ", outer_dim(input));
  NVTE_CHECK(input.size(-1) == 2 * N, "Expected input tensor to have inner dimension of ", 2 * N,
             ", but found ", input.size(-1));
  NVTE_CHECK(grad_input.dim() == 2, "Expected grad input tensor to have 2 dims, but found ",
             grad_input.dim());
  NVTE_CHECK(outer_dim(grad_input) == M, "Expected grad input tensor to have outer dimension of ",
             M, ", but found ", outer_dim(grad_input));
  NVTE_CHECK(grad_input.size(-1) == 2 * N, "Expected grad input tensor to have inner dimension of ",
             2 * N, ", but found ", grad_input.size(-1));
  NVTE_CHECK(grad_input_transpose.dim() == 2,
             "Expected grad input transpose tensor to have 2 dims, but found ",
             grad_input_transpose.dim());
  NVTE_CHECK(grad_input_transpose.size(0) == 2 * N,
             "Expected grad input tensor to have outer dimension of ", 2 * N, ", but found ",
             grad_input_transpose.size(0));
  NVTE_CHECK(grad_input_transpose.size(1) == M,
             "Expected grad input tensor to have outer dimension of ", M, ", but found ",
             grad_input_transpose.size(1));

  // Check tensor format
  NVTE_CHECK(grad_output.is_contiguous(), "Expected grad output tensor to be contiguous");
  NVTE_CHECK(input.is_contiguous(), "Expected input tensor to be contiguous");
  NVTE_CHECK(grad_input.is_contiguous(), "Expected grad input tensor to be contiguous");
  NVTE_CHECK(grad_input_transpose.is_contiguous(),
             "Expected grad input transpose tensor to be contiguous");
  NVTE_CHECK(grad_output.scalar_type() == input.scalar_type(),
             "Expected grad output tensor and input tensor to have same dtype");
  NVTE_CHECK(grad_input.scalar_type() == at::ScalarType::Byte,
             "Expected grad input tensor to be uint8 buffer");
  NVTE_CHECK(grad_input_transpose.scalar_type() == at::ScalarType::Byte,
             "Expected grad input transpose tensor to be uint8 buffer");

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  auto dy_cu = makeTransformerEngineTensor(grad_output);
  auto x_cu = makeTransformerEngineTensor(input);
  auto dx_cu = makeTransformerEngineTensor(grad_input.data_ptr(), {M, 2 * N}, otype, amax_dptr,
                                           scale_dptr, scale_inv_dptr);
  auto dx_t_cu = makeTransformerEngineTensor(grad_input_transpose.data_ptr(), {2 * N, M}, otype,
                                             amax_dptr, scale_dptr, scale_inv_dptr);

  // Launch kernel
  nvte_dswiglu_cast_transpose(dy_cu.data(), x_cu.data(), dx_cu.data(), dx_t_cu.data(),
                              at::cuda::getCurrentCUDAStream());
}

void fused_multi_cast_transpose_base(std::vector<at::Tensor> input_list,
                                     std::vector<void*> scale_dptr_list,
                                     std::vector<at::Tensor> cast_output_list,
                                     std::vector<at::Tensor> transposed_output_list,
                                     std::vector<void*> amax_dptr_list,
                                     std::vector<void*> scale_inv_dptr_list,
                                     transformer_engine::DType otype) {
  using namespace transformer_engine;

  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, cast_output_dptr_list, transposed_output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, cast_output_shape_list,
      transposed_output_shape_list;
  std::vector<transformer_engine::DType> input_type_list, cast_output_type_list,
      transposed_output_type_list;
  auto extract_tensor_props_skip_dtype = [](at::Tensor& tensor, std::vector<void*>& dptr_list,
                                            std::vector<std::vector<size_t>>& shape_list) {
    dptr_list.push_back(tensor.data_ptr());
    shape_list.push_back({});
    for (int d = 0; d < tensor.dim(); ++d) {
      shape_list.back().push_back(tensor.size(d));
    }
  };
  auto extract_tensor_props = [](at::Tensor& tensor, std::vector<void*>& dptr_list,
                                 std::vector<std::vector<size_t>>& shape_list,
                                 std::vector<transformer_engine::DType>& type_list) {
    dptr_list.push_back(tensor.data_ptr());
    shape_list.push_back({});
    for (int d = 0; d < tensor.dim(); ++d) {
      shape_list.back().push_back(tensor.size(d));
    }
    type_list.push_back(GetTransformerEngineDType(tensor.scalar_type()));
  };
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    extract_tensor_props(input_list[tensor_id], input_dptr_list, input_shape_list, input_type_list);
    extract_tensor_props_skip_dtype(cast_output_list[tensor_id], cast_output_dptr_list,
                                    cast_output_shape_list);
    cast_output_type_list.push_back(otype);
    extract_tensor_props_skip_dtype(transposed_output_list[tensor_id], transposed_output_dptr_list,
                                    transposed_output_shape_list);
    transposed_output_type_list.push_back(otype);
  }

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list, nvte_cast_output_list, nvte_transposed_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype, void* amax_dptr,
                                        void* scale_dptr, void* scale_inv_dptr) -> NVTETensor {
    tensor_wrappers.emplace_back(
        makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr, scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    if (input_dptr_list[i] == nullptr) continue;
    nvte_input_list.emplace_back(make_tensor(input_dptr_list[i], input_shape_list[i],
                                             input_type_list[i], nullptr, nullptr, nullptr));
    nvte_cast_output_list.emplace_back(
        make_tensor(cast_output_dptr_list[i], cast_output_shape_list[i], cast_output_type_list[i],
                    amax_dptr_list[i], scale_dptr_list[i], scale_inv_dptr_list[i]));
    nvte_transposed_output_list.emplace_back(
        make_tensor(transposed_output_dptr_list[i], transposed_output_shape_list[i],
                    transposed_output_type_list[i], amax_dptr_list[i], scale_dptr_list[i],
                    scale_inv_dptr_list[i]));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_cast_output_list.size() == nvte_input_list.size(),
             "Number of input and C output tensors must match");
  NVTE_CHECK(nvte_transposed_output_list.size() == nvte_input_list.size(),
             "Number of input and T output tensors must match");

  // Launch TE kernel
  nvte_multi_cast_transpose(nvte_input_list.size(), nvte_input_list.data(),
                            nvte_cast_output_list.data(), nvte_transposed_output_list.data(),
                            at::cuda::getCurrentCUDAStream());
}

void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_list,
                                std::vector<at::Tensor> scale_inv_list,
                                transformer_engine::DType otype) {
  std::vector<void*> scale_dptr_list, amax_dptr_list, scale_inv_dptr_list;
  for (size_t i = 0; i < scale_list.size(); ++i) {
    scale_dptr_list.push_back(scale_list[i].data_ptr());
    amax_dptr_list.push_back(amax_list[i].data_ptr());
    scale_inv_dptr_list.push_back(scale_inv_list[i].data_ptr());
  }

  fused_multi_cast_transpose_base(input_list, scale_dptr_list, cast_output_list,
                                  transposed_output_list, amax_dptr_list, scale_inv_dptr_list,
                                  otype);
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> fused_multi_cast_transpose_alloc(
    std::vector<at::Tensor> input_list, at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
    std::vector<int> scale_indices, std::vector<int> amax_indices,
    std::vector<int> scale_inv_indices, transformer_engine::DType otype) {
  using namespace transformer_engine;

  std::vector<at::Tensor> cast_output_list;
  std::vector<at::Tensor> transposed_output_list;
  std::vector<void*> scale_dptr_list, amax_dptr_list, scale_inv_dptr_list;
  for (size_t i = 0; i < input_list.size(); ++i) {
    auto input_i = input_list[i];
    // construct cast output tensors
    auto cast_output_i = allocateTorchTensor(input_i.size(0), input_i.size(1), DType::kByte);
    cast_output_list.push_back(cast_output_i);
    // construct transposed output tensors
    auto transposed_output_i = allocateTorchTensor(input_i.size(1), input_i.size(0), DType::kByte);
    transposed_output_list.push_back(transposed_output_i);
    // construct amax/scale/scale_inv dptr lists
    amax_dptr_list.push_back(getDataPtr(amax, amax_indices[i]));
    scale_dptr_list.push_back(getDataPtr(scale, scale_indices[i]));
    scale_inv_dptr_list.push_back(getDataPtr(scale_inv, scale_inv_indices[i]));
  }

  fused_multi_cast_transpose_base(input_list, scale_dptr_list, cast_output_list,
                                  transposed_output_list, amax_dptr_list, scale_inv_dptr_list,
                                  otype);

  return std::make_tuple(std::move(cast_output_list), std::move(transposed_output_list));
}

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto output = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  if (M == 0 || N == 0) return output;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

void fp8_transpose_noalloc(at::Tensor input, at::Tensor output, transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());
}

void fp8_transpose_noalloc_noop(at::Tensor input, at::Tensor output, at::Tensor noop,
                                transformer_engine::DType otype) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto noop_cu = makeTransformerEngineTensor(noop);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, M}, otype);

  nvte_transpose_with_noop(input_cu.data(), noop_cu.data(), output_cu.data(),
                           at::cuda::getCurrentCUDAStream());
}
