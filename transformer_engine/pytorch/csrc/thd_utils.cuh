/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_THD_UTILS_CUH_
#define TRANSFORMER_ENGINE_FUSED_ATTN_THD_UTILS_CUH_

#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>

struct LseCorrectionFunctor {
  __forceinline__ __device__ static void run(double *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    double val = lse[idx];
    float val_per_step = half_lse[half_idx];
    double max_scale = max(val, val_per_step);
    double min_scale = min(val, val_per_step);
    lse[idx] = max_scale + log(1.0 + exp(min_scale - max_scale));
  }
};

struct ReadLseFunctor {
  __forceinline__ __device__ static void run(float *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    half_lse[half_idx] = lse[idx];
  }
};

struct EmptyFunctor {
  __forceinline__ __device__ static void run(void *token, void *token_per_step, int idx) {}
};

struct CopyFunctor {
  __forceinline__ __device__ static void run(void *token, void *token_per_step, int idx) {
    reinterpret_cast<float4 *>(token)[idx] = reinterpret_cast<float4 *>(token_per_step)[idx];
  }
};

template <typename dtype>
struct AddFunctor {
  __forceinline__ __device__ static void run(dtype *token, dtype *token_per_step, int idx) {
    float4 d_ = reinterpret_cast<float4 *>(token)[idx];
    dtype *p_ = reinterpret_cast<dtype *>(&d_);

    float4 d = reinterpret_cast<float4 *>(token_per_step)[idx];
    dtype *p = reinterpret_cast<dtype *>(&d);

#pragma unroll
    for (int i = 0; i < sizeof(float4) / sizeof(dtype); i++) {
      p_[i] += p[i];
    }

    reinterpret_cast<float4 *>(token)[idx] = d_;
  }
};

namespace transformer_engine {
namespace fused_attn {

/***************************************************************************************************
 * Support THD format for Context Parallel: Binary search an array for a target value
 **************************************************************************************************/

__forceinline__ __device__ int binary_search(int target, int *array, int len) {
  int left = 1, right = len - 1;
  while (left < right) {
    int mid = (left + right) / 2;
    if (array[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Generate partitioned indices for input tokens
 **************************************************************************************************/
__global__ void thd_partition_indices_kernel(int *output, int *cu_seqlens, int batch,
                                             int total_tokens, int world_size, int rank) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    int seqlen = cu_seqlens[i];
    // Currently we assume that each sequence length is divisible by (world_size*2) since we have
    // to distribute each sequence evenly to different GPUs.
    assert(seqlen % (world_size * 2) == 0);
    cu_seqlens_s[i] = seqlen / world_size;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (int token_id = tid; token_id < total_tokens / world_size; token_id += num_threads) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
    int index = token_id - cu_seqlens_s[seq_id];
    int offset = index < seq_len / 2 ? rank : (world_size - 1) * 2 - rank;
    index += cu_seqlens_s[seq_id] * world_size + seq_len / 2 * offset;
    output[token_id] = index;
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/
__global__ void thd_read_half_tensor_kernel(void *half, void *tensor, int *cu_seqlens, int batch,
                                            int hidden_size_in_bytes, int half_idx,
                                            int dim_size_of_token) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / 2;
  }
  __syncthreads();

  int warpid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int laneid = threadIdx.x % 32;
  int num_warps = (blockDim.x * gridDim.x) / 32;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_float4s_per_token = hidden_size_in_bytes / sizeof(float4);

  size_t offset = static_cast<size_t>(dim_size_of_token) * hidden_size_in_bytes;
  half = reinterpret_cast<void *>(reinterpret_cast<char *>(half) + offset / 2 * blockIdx.y);
  tensor = reinterpret_cast<void *>(reinterpret_cast<char *>(tensor) + offset * blockIdx.y);

  for (int token_id = warpid; token_id < num_total_tokens; token_id += num_warps) {
    int seqid = binary_search(token_id, cu_seqlens_s, batch + 1);

    size_t offset_in_bytes = static_cast<size_t>(token_id) * hidden_size_in_bytes;
    float4 *cur_half_token =
        reinterpret_cast<float4 *>(reinterpret_cast<char *>(half) + offset_in_bytes);

    offset_in_bytes =
        (static_cast<size_t>(token_id) + cu_seqlens_s[seqid + half_idx]) * hidden_size_in_bytes;
    float4 *cur_token =
        reinterpret_cast<float4 *>(reinterpret_cast<char *>(tensor) + offset_in_bytes);

    for (int idx = laneid; idx < num_float4s_per_token; idx += 32) {
      cur_half_token[idx] = cur_token[idx];
    }
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

template <typename lse_dtype, bool lse_packed, typename Functor>
__global__ void thd_lse_kernel(lse_dtype *lse, float *half_lse, int *cu_seqlens, int batch,
                               int num_heads, int lse_seqlen, int second_half_lse_seqlen) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / 2;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  int num_total_tokens = cu_seqlens_s[batch];

  for (int token_id = tid; token_id < num_total_tokens; token_id += num_threads) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    for (int head_id = blockIdx.y; head_id < num_heads; head_id += gridDim.y) {
      size_t idx, half_idx;
      if constexpr (lse_packed) {
        idx = head_id * lse_seqlen + token_id + cu_seqlens_s[seq_id + 1];
        half_idx = head_id * second_half_lse_seqlen + token_id;
      } else {
        size_t row = static_cast<size_t>(seq_id) * num_heads + head_id;
        int col = token_id - cu_seqlens_s[seq_id];
        int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];

        idx = row * lse_seqlen + col + seq_len;
        half_idx = row * second_half_lse_seqlen + col;
      }

      Functor::run(lse, half_lse, idx, half_idx);
    }
  }
}

/***************************************************************************************************
 * Support BSHD, SBHD, and THD formats for Context Parallel: Out correction in forward
 **************************************************************************************************/

// Stores pointers to output and lse tensors for batch kernel launch.
template <int n>
struct TensorList {
  void *out[n];
  void *lse[n];
  int num;
};

struct IndexMapping {
  int out;
  int lse;
  int out_first_half;
  int lse_first_half;
  int out_second_half;
  int lse_second_half;
};

__device__
IndexMapping build_mapping_bshd(int index, int s_size, int h_size, int d_size) {
  int hd_size = h_size * d_size;
  int shd_size = s_size * hd_size;
  int b = index / shd_size;
  int s = index / hd_size % s_size;
  int h = index / d_size % h_size;
  
  IndexMapping mapping;
  mapping.out = index;
  mapping.out_first_half = index + index / shd_size * shd_size;
  mapping.out_second_half = index + index / shd_size * shd_size + shd_size;
  mapping.lse = b * (h_size * s_size) + h * s_size + s;  // lse is in BHS format
  mapping.lse_first_half = mapping.lse * 2 - mapping.lse % s_size;
  mapping.lse_second_half = mapping.lse_first_half + s_size;

  return mapping;
}

__device__
IndexMapping build_mapping_sbhd(int index, int s_size, int b_size, int h_size, int d_size) {
  int hd_size = h_size * d_size;
  int bhd_size = b_size * hd_size;
  int sbhd_size = s_size * bhd_size;
  int s = index / bhd_size;
  int b = index / hd_size % b_size;
  int h = index / d_size % h_size;

  IndexMapping mapping;
  mapping.out = index;
  mapping.out_first_half = index;
  mapping.out_second_half = index + sbhd_size;
  mapping.lse = b * (h_size * s_size) + h * s_size + s;  // lse is in BHS format
  mapping.lse_first_half = mapping.lse * 2 - mapping.lse % s_size;
  mapping.lse_second_half = mapping.lse_first_half + s_size;

  return mapping;
}

__device__
IndexMapping build_mapping_thd(int index, int h_size, int d_size, int *cu_seqlens, int batch, 
                               int lse_seqlen, bool packed_lse) {
  int hd_size = h_size * d_size;
  int t = index / hd_size;
  int h = index / d_size % h_size;
  int seq_id = binary_search(t, cu_seqlens, batch + 1);

  IndexMapping mapping;
  mapping.out = index;
  mapping.out_first_half = index + cu_seqlens[seq_id] * hd_size;
  mapping.out_second_half = index + cu_seqlens[seq_id + 1] * hd_size;
  if (packed_lse) {
    mapping.lse = h * lse_seqlen + t;
    mapping.lse_first_half = h * lse_seqlen * 2 + t + cu_seqlens[seq_id];
    mapping.lse_second_half = h * lse_seqlen * 2 + t + cu_seqlens[seq_id + 1];
  } else {
    int s = t - cu_seqlens[seq_id];
    mapping.lse = seq_id * (h_size * lse_seqlen) + h * lse_seqlen + s;
    mapping.lse_first_half = mapping.lse * 2 - s;
    mapping.lse_second_half = mapping.lse_first_half + cu_seqlens[seq_id + 1] - cu_seqlens[seq_id];
  }

  return mapping;
}

template <typename dtype, typename storage_type>
__device__ __forceinline__
void out_correction(storage_type *out, storage_type out_per_step, float lse, float lse_per_step) {
  dtype *p_out = reinterpret_cast<dtype *>(out);
  dtype *p_out_per_step = reinterpret_cast<dtype *>(&out_per_step);
  float lse_corrected = exp(lse_per_step - lse);
  for (int i = 0; i < sizeof(storage_type) / sizeof(dtype); i++) {
    p_out[i] += static_cast<float>(p_out_per_step[i]) * lse_corrected;
  }
}

template <typename dtype, bool causal, NVTE_QKV_Format out_format,
          bool softmax_lse_in_packed_format, int max_tensors, typename storage_type>
__global__ void fused_out_correction_kernel(dtype *out, TensorList<max_tensors> tensors, float *lse,
                                            int *cu_seqlens, int batch, int num_heads,
                                            int dim_per_head, int lse_seqlen, int num_total_tokens,
                                            int cp_size, int cp_rank, int finished_steps) {
  extern __shared__ int cu_seqlens_s[];
  if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
    for (int i = threadIdx.x; i < batch + 1; i += blockDim.x) {
      cu_seqlens_s[i] = cu_seqlens[i] / 2;
    }
    __syncthreads();
  }

  // num_valid_tokens <= num_total_tokens because there may be padded tokens.
  int num_valid_tokens;
  if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
    num_valid_tokens = cu_seqlens_s[batch] * 2;
  } else {
    num_valid_tokens = lse_seqlen * batch;
  }

  constexpr int elems_per_thread = sizeof(storage_type) / sizeof(dtype);
  int threads_per_token = num_heads * dim_per_head / elems_per_thread;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid / threads_per_token >= num_total_tokens / 2) {
    return;
  }
  if (tid / threads_per_token >= num_valid_tokens / 2) {
    // Padding with zeros for invalid areas of out tensor.
    int threads_do_padding = (num_total_tokens - num_valid_tokens) * threads_per_token / 2;
    reinterpret_cast<storage_type *>(out)[tid * 2] = {0};
    reinterpret_cast<storage_type *>(out)[tid * 2 + threads_do_padding] = {0};
    return;
  }

  int num_full_tensors;
  if constexpr (causal) {
    // If causal is True, some tensors have only the half size of out tensor:
    //     Give cp_size, cp_rank, there should be cp_size correction steps in total, in which the 
    //     [0, cp_rank + 1) steps are full tensors, and the [cp_rank + 1, cp_size) steps are half 
    //     tensors.
    //     Current kernel is responsible for the [finished_steps, finished_steps + tensors.num)
    //     steps.
    //     We want to compute the number of full tensors, so the left_boundary of full tensors is
    //        max(0, finished_steps) = finished_steps (because finished_steps is alwasys >=0),
    //     and the right_boundary of full tensors is
    //        min(cp_rank + 1, finished_steps + tensors.num).
    //     The number of full tensors is 0 if right boundary <= left_boundary, otherwise it's (right
    //     boundary - left_boundary).
    num_full_tensors = min(max(cp_rank + 1 - finished_steps, 0), tensors.num);
  } else {
    // If causal is False, all the tensors have the same size as the out tensor.
    num_full_tensors = tensors.num;
  }

  IndexMapping mapping;
  if constexpr (out_format == NVTE_QKV_Format::NVTE_BSHD) {
    mapping = build_mapping_bshd(tid, lse_seqlen / 2, num_heads, dim_per_head / elems_per_thread);
  } else if constexpr (out_format == NVTE_QKV_Format::NVTE_SBHD) {
    mapping = build_mapping_sbhd(
      tid, lse_seqlen / 2, batch, num_heads, dim_per_head / elems_per_thread);
  } else if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
    mapping = build_mapping_thd(tid, num_heads, dim_per_head / elems_per_thread, cu_seqlens_s,
                                batch, lse_seqlen / 2, softmax_lse_in_packed_format);
  }

  storage_type out_buffer;
  storage_type out_per_step_buffer;

  // Step1: Calculate the first half, only full tensors need to be concerned.
  {
    float lse_full = lse[mapping.lse_first_half];
    out_buffer = {0};
    for (int tensor_id = 0; tensor_id < num_full_tensors; tensor_id++) {
      float lse_per_step = reinterpret_cast<float *>(tensors.lse[tensor_id])[mapping.lse_first_half];
      out_per_step_buffer = reinterpret_cast<storage_type *>(tensors.out[tensor_id])[mapping.out_first_half];
      out_correction<dtype>(&out_buffer, out_per_step_buffer, lse_full, lse_per_step);
    }
    reinterpret_cast<storage_type *>(out)[mapping.out_first_half] = out_buffer;
  }

  // Step2: Calculate the second half, all tensors need to be concerned.
  {
    float lse_full = lse[mapping.lse_second_half];
    out_buffer = {0};
    // Step2.1 Calculate the second half of full tensors.
    for (int tensor_id = 0; tensor_id < num_full_tensors; tensor_id++) {
      float lse_per_step = reinterpret_cast<float *>(tensors.lse[tensor_id])[mapping.lse_second_half];
      out_per_step_buffer = reinterpret_cast<storage_type *>(tensors.out[tensor_id])[mapping.out_second_half];
      out_correction<dtype>(&out_buffer, out_per_step_buffer, lse_full, lse_per_step);
    }
    // Step2.2 Calculate the second half of non-full tensors.
    for (int tensor_id = num_full_tensors; tensor_id < tensors.num; tensor_id++) {
      float lse_per_step = reinterpret_cast<float *>(tensors.lse[tensor_id])[mapping.lse];
      out_per_step_buffer = reinterpret_cast<storage_type *>(tensors.out[tensor_id])[mapping.out];
      out_correction<dtype>(&out_buffer, out_per_step_buffer, lse_full, lse_per_step);
    }
    reinterpret_cast<storage_type *>(out)[mapping.out_second_half] = out_buffer;
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

template <typename dtype, typename Functor_0, typename Functor_1, int functor_idx, int group_size>
__global__ void thd_grad_correction_kernel(dtype *grad, dtype *grad_per_step, int *cu_seqlens,
                                           int batch, int hidden_size, int dim_size_of_token) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    if constexpr (functor_idx < 2) {
      cu_seqlens_s[i] = cu_seqlens[i] / 2;
    } else {
      cu_seqlens_s[i] = cu_seqlens[i];
    }
  }
  __syncthreads();

  int group_id = (blockIdx.x * blockDim.x + threadIdx.x) / group_size;
  int lane_id = threadIdx.x % group_size;
  int num_groups = (blockDim.x * gridDim.x) / group_size;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_inner_loops = hidden_size * sizeof(dtype) / sizeof(float4);

  size_t offset = static_cast<size_t>(dim_size_of_token) * hidden_size;
  if constexpr (functor_idx < 2) {
    grad_per_step = grad_per_step + offset / 2 * blockIdx.y;
  } else {
    grad_per_step = grad_per_step + offset * blockIdx.y;
  }
  grad = grad + offset * blockIdx.y;

  for (int token_id = group_id; token_id < num_total_tokens; token_id += num_groups) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);

    int token_offset;
    bool is_first_half;
    if constexpr (functor_idx < 2) {
      token_offset = cu_seqlens_s[seq_id + functor_idx];
      is_first_half = (functor_idx == 0);
    } else {
      token_offset = 0;
      int len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
      is_first_half = (token_id - cu_seqlens_s[seq_id]) < (len / 2);
    }

    dtype *token = &grad[(token_id + token_offset) * static_cast<size_t>(hidden_size)];
    dtype *token_per_step = &grad_per_step[token_id * static_cast<size_t>(hidden_size)];
    for (int idx = lane_id; idx < num_inner_loops; idx += group_size) {
      if (is_first_half) {
        Functor_0::run(token, token_per_step, idx);
      } else {
        Functor_1::run(token, token_per_step, idx);
      }
    }
  }
}

}  // namespace fused_attn
}  // namespace transformer_engine
#endif
