#pragma once
#include <es_la/dense.hpp>
#include <es_util/thread.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <thread>
#include <vector>

namespace internal
{
template<class Network>
class Trainer
{
public:
	Trainer(Network& network) : network_(network)
	{}

	template<class In, class Labels, class Callback_fn>
	es_la::Vector_xd operator()(
		const In& in, const Labels& labels, unsigned int n_iters, double rate, Callback_fn callback_fn)
	{
		assert(in.cols() == labels.size());

		const auto n_workers = std::max(1u, std::thread::hardware_concurrency());
		const auto n_samples = labels.size();
		const auto n_samples_per_worker = (n_samples + n_workers - 1) / n_workers;

		std::vector<std::thread> workers;

		std::vector<typename Network::Layers_parameters> wrk_param_grads(n_workers);
		es_la::Vector_xd wrk_loss_function(n_workers);
		es_la::Vector_xd loss_function(n_iters);

		es_util::Barrier barrier(n_workers, [&, this](std::size_t it)
		{
			for (auto& p : wrk_param_grads)
				network_.add_gradients(-rate / n_samples, p);

			loss_function[it] = 0;
			for (std::size_t i = 0; i < wrk_loss_function.size(); ++i)
				loss_function[it] += wrk_loss_function[i];
			loss_function[it] /= n_samples;

			callback_fn(it, loss_function[it]);
		});

		for (unsigned int i = 0; i < n_workers; ++i)
		{
			const auto first = i * n_samples_per_worker;
			const auto n = std::min(n_samples - first, n_samples_per_worker);

			workers.emplace_back([&, n, first, i, this]()
			{
				train_step(in.cols_view(first, n), labels.rows_view(first, n), wrk_param_grads[i], wrk_loss_function[i],
					barrier, n_iters);
			});
		}

		for (auto& w : workers)
			w.join();

		return loss_function;
	}

private:
	template<class In, class Labels, class Barrier>
	void train_step(In in, Labels labels, typename Network::Layers_parameters& param_grads, double& loss_function,
		Barrier& barrier, unsigned int n_iters)
	{
		const auto n = labels.size();
		assert(in.cols() == n);

		typename Network::Layers_outputs outs;
		typename Network::Layers_outputs out_grads;

		for (unsigned int it = 0; it < n_iters; ++it)
		{
			network_.compute_outputs(in, outs);

			auto& last_grad = out_grads.back();
			last_grad.resize(outs.back().rows(), outs.back().cols());
			last_grad = 0;
			for (std::size_t j = 0; j < n; ++j)
				last_grad(labels[j], j) = -1 / outs.back()(labels[j], j);

			network_.compute_gradients(in, outs, out_grads, param_grads);

			loss_function = 0;
			for (std::size_t j = 0; j < n; ++j)
				loss_function -= std::log(outs.back()(labels[j], j));

			barrier.wait();
		}
	}

private:
	Network& network_;
};
} // namespace internal
