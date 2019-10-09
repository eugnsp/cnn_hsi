#pragma once
#include <esl/dense.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <thread>
#include <vector>

namespace internal
{
template<class Network>
class Classifier
{
public:
	Classifier(const Network& network) : network_(network)
	{}

	template<class In>
	esl::Vector_x<std::size_t> operator()(const In& in) const
	{
		const auto n_workers = std::max(1u, std::thread::hardware_concurrency());
		const auto n_samples = in.cols();
		const auto n_samples_per_worker = (n_samples + n_workers - 1) / n_workers;

		esl::Vector_x<std::size_t> labels(n_samples);
		std::vector<std::thread> workers;

		for (unsigned int i = 0; i < n_workers; ++i)
		{
			const auto first = i * n_samples_per_worker;
			const auto n = std::min(n_samples - first, n_samples_per_worker);
			workers.emplace_back(
				[this, &in, &labels, first, n]() { classify(in.cols_view(first, n), labels.rows_view(first, n)); });
		}

		for (auto& w : workers)
			w.join();

		return labels;
	}

private:
	template<class In, class Labels>
	void classify(In in, Labels labels) const
	{
		const auto n = labels.size();
		assert(in.cols() == n);

		const auto outputs = network_.compute_outputs(in);
		const auto& output = outputs.back();
		assert(output.cols() == n);

		for (std::size_t j = 0; j < n; ++j)
		{
			std::size_t max_index = 0;
			double max_value = 0;
			for (std::size_t i = 0; i < output.rows(); ++i)
				if (output(i, j) > max_value)
				{
					max_index = i;
					max_value = output(i, j);
				}

			labels[j] = max_index;
		}
	}

private:
	const Network& network_;
};
} // namespace internal
