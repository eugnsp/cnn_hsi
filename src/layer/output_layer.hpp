#pragma once
#include "layer.hpp"

#include <es_la/dense.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <string>

class Output_layer : public Trainable_layer
{
public:
	explicit Output_layer(std::size_t n_nodes) : n_nodes_(n_nodes)
	{}

	template<class Strategy, class Layer>
	void init(Strategy&& init_strategy, const Layer& prev_layer)
	{
		init_storage(n_nodes_, prev_layer.output_size(), init_strategy);
	}

	template<class Input, class Output>
	void compute_output(const Input& in, Output& out) const
	{
		const auto n = in.cols();

		out.resize(n_nodes_, n);
		out = params_.weights * in;

		for (std::size_t j = 0; j < n; ++j)
		{
			out.col_view(j) += params_.biases;

			double norm = 0;
			for (std::size_t i = 0; i < n_nodes_; ++i)
			{
				out(i, j) = std::exp(out(i, j));
				norm += out(i, j);
			}

			out.col_view(j) /= norm;
		}
	}

	template<class In, class Out, class In_grad, class Out_grad>
	void compute_gradient(
		const In& in, const Out& out, In_grad& in_grad, const Out_grad& out_grad, Parameters& params_grad) const
	{
		assert(in.cols() == out.cols());
		assert(out_grad.cols() == in.cols());

		const auto n = in.cols();
		in_grad.resize(in.rows(), n);

		es_la::Matrix_xd m = out_grad;
		for (std::size_t j = 0; j < n; ++j)
			for (std::size_t i = 0; i < n_nodes_; ++i)
				m(i, j) *= out(i, j);

		es_la::Matrix_xd m2 = m;
		for (std::size_t j = 0; j < n; ++j)
		{
			const auto sum = es_la::sum(m.col_view(j));
			m2.col_view(j) -= sum * out.col_view(j);
		}

		in_grad = params_.weights.tr_view() * m2;
		params_grad.weights += m2 * in.tr_view();

		for (std::size_t j = 0; j < n; ++j)
			params_grad.biases += m.col_view(j);
		auto m1 = (out * m.tr_view()).eval();
		for (std::size_t i = 0; i < n_nodes_; ++i)
			params_grad.biases -= m1.col_view(i);
	}

	virtual std::string name() const override
	{
		return "Output layer";
	}

	std::string info_string() const
	{
		std::string info = name() + '\n';
		info += "  Number of nodes: " + std::to_string(n_nodes_) + "\n";
		info += "  Number of trainable parameters: " + std::to_string(n_trainable_params()) + "\n";
		return info;
	}

private:
	const std::size_t n_nodes_;
};
