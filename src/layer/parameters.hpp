#pragma once
#include <esl/dense.hpp>

#include <cstddef>

struct Empty_parameters
{};

struct Trainable_parameters
{
	esl::Matrix_xd weights;
	esl::Vector_xd biases;

	void reset()
	{
		weights = 0;
		biases = 0;
	}
};
