#pragma once
#include "layer.hpp"

#include <es_la/dense.hpp>
#include <es_util/numeric.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <string>

class Conv_layer : public Trainable_layer
{
public:
	Conv_layer(std::size_t n_kernels, std::size_t kernel_size) : n_kernels_(n_kernels), kernel_size_(kernel_size)
	{}

	template<class Strategy, class Layer>
	void init(Strategy&& init_strategy, const Layer& prev_layer)
	{
		output_size_per_kernel_ = get_output_size(prev_layer.output_size());
		init_storage(n_kernels_, kernel_size_, init_strategy);
	}

	template<class In, class Out>
	void compute_output(const In& input, Out& output) const
	{
		output.resize(output_size(), input.cols());

		for (std::size_t col = 0; col < input.cols(); ++col)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel_; ++i)
				{
					double conv = 0;
					for (std::size_t j = 0; j < kernel_size_; ++j)
						conv += params_.weights(k, j) * input(i + j, col);
					output(i + k * output_size_per_kernel_, col) = std::tanh(conv + params_.biases[k]);
				}
	}

	template<class In, class Out, class Out_grad>
	void compute_gradient(const In& in, const Out& out, const Out_grad& out_grad, Parameters& params_grad)
	{
		assert(in.cols() == out.cols());

		for (std::size_t col = 0; col < in.cols(); ++col)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel_; ++i)
				{
					const auto d = 1 - es_util::sq(out(i + k * output_size_per_kernel_, col));
					params_grad.biases[k] += d * out_grad(i + k * output_size_per_kernel_, col);
				}

		for (std::size_t col = 0; col < in.cols(); ++col)
			for (std::size_t p = 0; p < kernel_size_; ++p)
				for (std::size_t q = 0; q < n_kernels_; ++q)
					for (std::size_t i = 0; i < output_size_per_kernel_; ++i)
					{
						const auto d = 1 - es_util::sq(out(i + q * output_size_per_kernel_, col));
						params_grad.weights(q, p) +=
							d * out_grad(i + q * output_size_per_kernel_, col) * in(i + p, col);
					}
	}

	std::size_t output_size() const
	{
		return output_size_per_kernel_ * n_kernels_;
	}

	std::size_t output_size_per_kernel() const
	{
		return output_size_per_kernel_;
	}

	std::size_t n_kernels() const
	{
		return n_kernels_;
	}

	virtual std::string name() const override
	{
		return "Convolution layer";
	}

	std::string info_string() const
	{
		std::string info = name() + '\n';
		info += "  Number of kernels: " + std::to_string(n_kernels_) + "\n";
		info += "  Kernel size: " + std::to_string(kernel_size_) + "\n";
		info += "  Number of trainable parameters: " + std::to_string(n_trainable_params()) + "\n";
		return info;
	}

private:
	std::size_t get_output_size(std::size_t input_size) const
	{
		assert(input_size >= kernel_size_);
		return input_size - kernel_size_ + 1;
	}

private:
	const std::size_t n_kernels_;
	const std::size_t kernel_size_;
	std::size_t output_size_per_kernel_ = 0;
};
