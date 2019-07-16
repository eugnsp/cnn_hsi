#pragma once
#include <es_la/dense.hpp>

#include <cstddef>

struct Empty_parameters
{};

struct Trainable_parameters
{
	es_la::Matrix_xd weights;
	es_la::Vector_xd biases;

	void reset()
	{
		weights = 0;
		biases = 0;
	}
};
