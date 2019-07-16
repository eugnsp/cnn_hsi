#pragma once

#include <cmath>
#include <cstddef>

template<class Neural_network, class In, class Labels>
class Loss_fn_calculator
{
public:
	Loss_fn_calculator(const Neural_network& network, const In& in, const Labels& labels) :
		network_(network), in_(in), labels_(labels)
	{}

	double operator()() const
	{
		const auto outs = network_.compute_outputs(in_);
		double loss = 0;
		for (std::size_t i = 0; i < labels_.size(); ++i)
			loss -= std::log(outs.back()(labels_[i], i));
		return loss;
	}

private:
	const Neural_network& network_;
	const In& in_;
	const Labels& labels_;
};
