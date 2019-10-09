#pragma once
#include "layer.hpp"
#include <esl/dense.hpp>

#include <cassert>
#include <cstddef>

class Pooling_layer : public Layer
{
public:
	explicit Pooling_layer(std::size_t pooling_size) : pooling_size_(pooling_size)
	{}

	template<class Strategy, class Layer>
	void init(Strategy&&, const Layer& prev_layer)
	{
		const auto input_size_per_kernel = prev_layer.output_size_per_kernel();
		const auto output_size_per_kernel = get_output_size(input_size_per_kernel);
		n_kernels_ = prev_layer.n_kernels();
		output_size_ = output_size_per_kernel * n_kernels_;
	}

	template<class In, class Out>
	void compute_output(const In& in, Out& out) const
	{
		const auto input_size = in.rows();
		const auto input_size_per_kernel = input_size / n_kernels_;
		const auto output_size_per_kernel = output_size_ / n_kernels_;

		out.resize(output_size_, in.cols());

		for (std::size_t col = 0; col < in.cols(); ++col)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel; ++i)
				{
					auto max = in(i * pooling_size_ + k * input_size_per_kernel, col);
					for (std::size_t p = 1; p < pooling_size_; ++p)
					{
						auto v = in(p + i * pooling_size_ + k * input_size_per_kernel, col);
						max = std::max(max, v);
					}
					out(i + k * output_size_per_kernel, col) = max;
				}
	}

	template<class In, class Out, class In_grad, class Out_grad>
	void compute_gradient(const In& in, const Out& out, In_grad& in_grad, const Out_grad& out_grad) const
	{
		assert(in.cols() == out.cols());

		const auto input_size = in.rows();
		const auto input_size_per_kernel = input_size / n_kernels_;
		const auto output_size_per_kernel = output_size_ / n_kernels_;

		in_grad.resize(in.rows(), in.cols());
		in_grad = 0;

		for (std::size_t col = 0; col < in.cols(); ++col)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel; ++i)
				{
					std::size_t max_index = 0;
					auto max = in(i * pooling_size_ + k * input_size_per_kernel, col);
					for (std::size_t p = 1; p < pooling_size_; ++p)
					{
						auto v = in(p + i * pooling_size_ + k * input_size_per_kernel, col);
						if (v > max)
						{
							max_index = p;
							max = v;
						}
					}
					in_grad(max_index + i * pooling_size_ + k * input_size_per_kernel, col) = out_grad(i + k * output_size_per_kernel, col);
				}
	}

	std::size_t output_size() const
	{
		return output_size_;
	}

	virtual std::string name() const override
	{
		return "Pooling layer";
	}

	std::string info_string() const
	{
		std::string info = name() + '\n';
		info += "  Pooling size: " + std::to_string(pooling_size_) + "\n";
		return info;
	}

private:
	std::size_t get_output_size(std::size_t input_size) const
	{
		assert(input_size >= pooling_size_);
		return input_size / pooling_size_;
	}

private:
	const std::size_t pooling_size_;
	std::size_t output_size_ = 0;
	std::size_t n_kernels_ = 0;
};
