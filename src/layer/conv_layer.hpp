#pragma once
#include "layer.hpp"

#include <es_la/dense.hpp>
#include <es_util/numeric.hpp>

#include <mkl_types.h>
#include <mkl_vsl.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <string>

#define USE_MKL_CONV

class Conv_layer : public Trainable_layer
{
public:
	explicit Conv_layer(std::size_t n_kernels, std::size_t kernel_size) :
		n_kernels_(n_kernels), kernel_size_(kernel_size)
	{}

	template<class Strategy, class Layer>
	void init(Strategy&& init_strategy, const Layer& prev_layer)
	{
		output_size_per_kernel_ = get_output_size(prev_layer.output_size());
		init_storage(n_kernels_, kernel_size_, init_strategy);

		// params_.weights.resize(kernel_size_, n_kernels_);
		// init_strategy(params_.weights);

		// params_.biases.resize(n_kernels_);
		// init_strategy(params_.biases);
	}

	template<class In, class Out>
	void compute_output(const In& in, Out& out) const
	{
		const auto n = in.cols();
		out.resize(output_size(), in.cols());

#ifdef USE_MKL_CONV
		for (std::size_t k = 0; k < n_kernels_; ++k)
		{
			::VSLCorrTaskPtr task;
			[[maybe_unused]] const auto st = ::vsldCorrNewTaskX1D(&task, VSL_CORR_MODE_AUTO, kernel_size_, in.rows(),
				output_size_per_kernel_, params_.weights.row_view(k).data(), params_.weights.rows());
			assert(st == VSL_STATUS_OK);

			const MKL_INT start = 0;
			::vslCorrSetStart(task, &start);

			for (std::size_t j = 0; j < n; ++j)
				::vsldCorrExecX1D(
					task, in.col_view(j).data(), 1, out.col_view(j).data() + k * output_size_per_kernel_, 1);

			::vslCorrDeleteTask(&task);
		}

		for (std::size_t j = 0; j < n; ++j)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel_; ++i)
					out(i + k * output_size_per_kernel_, j) =
						std::tanh(out(i + k * output_size_per_kernel_, j) + params_.biases[k]);
#else
		for (std::size_t j = 0; j < n; ++j)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel_; ++i)
				{
					double conv = 0;
					for (std::size_t p = 0; p < kernel_size_; ++p)
						conv += params_.weights(k, p) * in(i + p, j);
					out(i + k * output_size_per_kernel_, j) = std::tanh(conv + params_.biases[k]);
				}
#endif
	}

	template<class In, class Out, class Out_grad>
	void compute_gradient(const In& in, const Out& out, const Out_grad& out_grad, Parameters& params_grad) const
	{
		assert(in.cols() == out.cols());

		const auto n = in.cols();

		es_la::Matrix_xd m(out.rows(), n);
		for (std::size_t j = 0; j < n; ++j)
			for (std::size_t i = 0; i < out.rows(); ++i)
				m(i, j) = (1 - es_util::sq(out(i, j))) * out_grad(i, j);

		for (std::size_t col = 0; col < n; ++col)
			for (std::size_t k = 0; k < n_kernels_; ++k)
				for (std::size_t i = 0; i < output_size_per_kernel_; ++i)
					params_grad.biases[k] += m(i + k * output_size_per_kernel_, col);

		for (std::size_t k = 0; k < n_kernels_; ++k)
			for (std::size_t col = 0; col < n; ++col)
				for (std::size_t i = 0; i < kernel_size_; ++i)
					for (std::size_t p = 0; p < output_size_per_kernel_; ++p)
						params_grad.weights(k, i) += m(p + k * output_size_per_kernel_, col) * in(p + i, col);

		// TODO : use MKL
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
