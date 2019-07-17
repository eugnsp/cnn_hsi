#pragma once
#include "../layer/parameters.hpp"
#include "../util/loss_fn_calculator.hpp"
#include "classifier.hpp"
#include "trainer.hpp"

#include <es_la/dense.hpp>
#include <es_util/tuple.hpp>
#include <es_util/type_traits.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <iostream>

template<class... Layers>
class Neural_network
{
	friend class internal::Trainer<Neural_network>;

public:
	static constexpr auto n_layers = sizeof...(Layers);
	static_assert(n_layers >= 2);

	using Layers_outputs = std::array<es_la::Matrix_xd, n_layers>;
	using Layers_parameters = std::tuple<typename Layers::Parameters...>;

private:
	using Layers_tuple = std::tuple<Layers...>;

	// A fictitious layer that is used to provide
	// the first real layer with the input data size
	struct Input_layer
	{
	public:
		explicit Input_layer(std::size_t output_size) : output_size_(output_size)
		{}

		std::size_t output_size() const
		{
			return output_size_;
		}

	private:
		const std::size_t output_size_;
	};

public:
	template<class... Ls>
	explicit Neural_network(Ls&&... layers) : layers_(std::forward<Ls>(layers)...)
	{}

	template<class... Ts>
	Neural_network(std::piecewise_construct_t, Ts&&... arg_tuples) :
		Neural_network(std::make_index_sequence<n_layers>{}, std::forward_as_tuple(std::forward<Ts>(arg_tuples)...))
	{}

	template<class Strategy>
	void init(Strategy&& init_strategy, std::size_t input_size)
	{
		input_size_ = input_size;

		std::get<0>(layers_).init(init_strategy, Input_layer{input_size});
		init_impl(init_strategy, std::make_index_sequence<n_layers - 1>{});
	}

	template<class In>
	void compute_outputs(const In& in, Layers_outputs& outs) const
	{
		std::get<0>(layers_).compute_output(in, outs[0]);
		compute_outputs_impl(outs, std::make_index_sequence<n_layers - 1>{});
	}

	template<class In>
	Layers_outputs compute_outputs(const In& in) const
	{
		Layers_outputs outs;
		compute_outputs(in, outs);
		return outs;
	}

	template<class In>
	void compute_gradients(
		const In& in, const Layers_outputs& outs, Layers_outputs& out_grads, Layers_parameters& param_grads) const
	{
		compute_gradients_impl(in, outs, out_grads, param_grads);
	}

	template<class In>
	es_la::Vector_x<std::size_t> classify(const In& in) const
	{
		assert(in.rows() == input_size_);
		return internal::Classifier{*this}(in);
	}

	template<class In, class Labels, class Callback_fn>
	es_la::Vector_xd train(
		const In& in, const Labels& labels, unsigned int n_iters, double rate, Callback_fn callback_fn)
	{
		assert(in.rows() == input_size_);
		return internal::Trainer{*this}(in, labels, n_iters, rate, callback_fn);
	}

	template<class In, class Labels>
	es_la::Vector_xd train(const In& in, const Labels& labels, unsigned int n_iters, double rate)
	{
		return train(in, labels, n_iters, rate, [](auto...) {});
	}

	std::string info_string() const
	{
		std::string info = "Neural network contains " + std::to_string(n_layers) + " layers:\n";
		std::size_t i = 1;

		es_util::tuple_for_each(
			[&info, &i](auto& layer) { info += std::to_string(i++) + ". " + layer.info_string(); }, layers_);

		return info;
	}

	template<class In, class Labels>
	void check_gradients(const In& in, const Labels& labels)
	{
		assert(in.cols() == labels.size());
		const auto n = labels.size();

		const Layers_outputs outs = compute_outputs(in);
		Layers_outputs out_grads;
		Layers_parameters param_grads;

		auto& last_grad = out_grads[n_layers - 1];
		last_grad.resize(outs[n_layers - 1].rows(), outs[n_layers - 1].cols());
		last_grad = 0;
		for (std::size_t col = 0; col < n; ++col)
			last_grad(labels[col], col) = -1 / outs.back()(labels[col], col);

		compute_gradients(in, outs, out_grads, param_grads);
		check_gradients_impl(Loss_fn_calculator{*this, in, labels}, param_grads);
	}

private:
	template<class Strategy, std::size_t... indices>
	void init_impl(Strategy&& init_strategy, std::index_sequence<indices...>)
	{
		(std::get<indices + 1>(layers_).init(init_strategy, std::get<indices>(layers_)), ...);
	}

	template<std::size_t... indices>
	void compute_outputs_impl(Layers_outputs& outs, std::index_sequence<indices...>) const
	{
		(std::get<indices + 1>(layers_).compute_output(outs[indices], outs[indices + 1]), ...);
	}

	template<std::size_t index = n_layers - 1, class In>
	void compute_gradients_impl(
		const In& in, const Layers_outputs& outs, Layers_outputs& out_grads, Layers_parameters& param_grads) const
	{
		if constexpr (std::is_same_v<std::tuple_element_t<index, Layers_parameters>, Empty_parameters>)
		{
			if constexpr (index > 0)
				std::get<index>(layers_).compute_gradient(
					outs[index - 1], outs[index], out_grads[index - 1], out_grads[index]);
		}
		else
		{
			std::get<index>(layers_).reset(std::get<index>(param_grads));
			if constexpr (index > 0)
				std::get<index>(layers_).compute_gradient(
					outs[index - 1], outs[index], out_grads[index - 1], out_grads[index], std::get<index>(param_grads));
			else
				std::get<index>(layers_).compute_gradient(
					in, outs[index], out_grads[index], std::get<index>(param_grads));
		}

		if constexpr (index > 0)
			compute_gradients_impl<index - 1>(in, outs, out_grads, param_grads);
	}

	template<std::size_t index = n_layers - 1>
	void add_gradients(double alpha, const Layers_parameters& param_grads)
	{
		if constexpr (!std::is_same_v<std::tuple_element_t<index, Layers_parameters>, Empty_parameters>)
			std::get<index>(layers_).add_params(alpha, std::get<index>(param_grads));

		if constexpr (index > 0)
			add_gradients<index - 1>(alpha, param_grads);
	}

	template<std::size_t index = n_layers - 1, class Loss_fn>
	void check_gradients_impl(const Loss_fn& loss_fn, Layers_parameters param_grads)
	{
		std::cout << "Checking " << std::get<index>(layers_).name() << std::endl;
		if constexpr (!std::is_same_v<std::tuple_element_t<index, Layers_parameters>, Empty_parameters>)
			std::get<index>(layers_).check_gradient(loss_fn, std::get<index>(param_grads));

		if constexpr (index > 0)
			check_gradients_impl<index - 1>(loss_fn, param_grads);
	}

private:
	template<std::size_t... indices, class Tuple>
	Neural_network(std::index_sequence<indices...>, Tuple&& tuple) :
		layers_(std::make_from_tuple<std::tuple_element_t<indices, Layers_tuple>>(
			std::get<indices>(std::forward<Tuple>(tuple)))...)
	{}

private:
	Layers_tuple layers_;
	std::size_t input_size_;
};

template<class... Layers>
auto make_neural_network(Layers&&... layers)
{
	return Neural_network<es_util::Remove_cv_ref<Layers>...>(std::forward<Layers>(layers)...);
}
