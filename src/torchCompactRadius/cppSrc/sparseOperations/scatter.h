#pragma once
#include <common.h>
#include <hashing.h>
#include <multiLevelMemory/mlmUtil.h>
#include <algorithm>
#include <optional>
#include <atomic>


namespace TORCH_EXTENSION_NAME{
    torch::Tensor
    scatter_sum(torch::Tensor src, torch::Tensor index, int64_t dim,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size);

    torch::Tensor
    scatter_mul(torch::Tensor src, torch::Tensor index, int64_t dim,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size);

    torch::Tensor
    scatter_mean(torch::Tensor src, torch::Tensor index, int64_t dim,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size);

    std::tuple<torch::Tensor, torch::Tensor>
    scatter_min(torch::Tensor src, torch::Tensor index, int64_t dim,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size);

    std::tuple<torch::Tensor, torch::Tensor>
    scatter_max(torch::Tensor src, torch::Tensor index, int64_t dim,
                std::optional<torch::Tensor> optional_out,
                std::optional<int64_t> dim_size);
}