#pragma once
#include "layer.hpp"

#include <es_la/dense.hpp>
#include <es_util/numeric.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <string>

class Fc_layer : public Trainable_layer
{
public:
	Fc_layer(std::size_t n_nodes) : n_nodes_(n_nodes)
	{}

	template<class Strategy, class Layer>
	void init(Strategy&& init_strategy, const Layer& prev_layer)
	{
		init_storage(n_nodes_, prev_layer.output_size(), init_strategy);
	}

	template<class In, class Out>
	void compute_output(const In& in, Out& out) const
	{
		const auto n = in.cols();

		out.resize(n_nodes_, n);
		out = params_.weights * in;

		for (std::size_t j = 0; j < n; ++j)
			out.col_view(j) += params_.biases;

		for (std::size_t j = 0; j < n; ++j)
			for (std::size_t i = 0; i < n_nodes_; ++i)
				out(i, j) = std::tanh(out(i, j));
	}

	template<class In, class Out, class In_grad, class Out_grad>
	void compute_gradient(
		const In& in, const Out& out, In_grad& in_grad, const Out_grad& out_grad, Parameters& params_grad)
	{
		assert(out.cols() == in.cols());
		assert(out_grad.cols() == in.cols());

		const auto n = in.cols();
		in_grad.resize(in.rows(), n);

		const auto jacobian = compute_jacobian(out_grad, out);

		in_grad = params_.weights.tr_view() * jacobian;
		params_grad.weights += jacobian * in.tr_view();

		for (std::size_t j = 0; j < n; ++j)
			params_grad.biases += jacobian.col_view(j);
	}

	std::size_t output_size() const
	{
		return n_nodes_;
	}

	virtual std::string name() const override
	{
		return "Fully connected layer";
	}

	std::string info_string() const
	{
		std::string info = name() + '\n';
		info += "  Number of nodes: " + std::to_string(n_nodes_) + "\n";
		info += "  Number of trainable parameters: " + std::to_string(n_trainable_params()) + "\n";
		return info;
	}

private:
	template<class Out>
	es_la::Matrix_xd compute_jacobian(es_la::Matrix_xd out_grad, const Out& out) const
	{
		const auto n = out.cols();

		for (std::size_t j = 0; j < n; ++j)
			for (std::size_t i = 0; i < n_nodes_; ++i)
				out_grad(i, j) *= (1 - es_util::sq(out(i, j)));

		return out_grad;
	}

private:
	const std::size_t n_nodes_;
};
