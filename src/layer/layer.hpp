#pragma once
#include "parameters.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

class Layer
{
public:
	using Parameters = Empty_parameters;

public:
	virtual std::string name() const = 0;
};

class Trainable_layer : Layer
{
public:
	using Parameters = Trainable_parameters;

public:
	void reset(Parameters& params) const
	{
		params.weights.resize(params_.weights.rows(), params_.weights.cols());
		params.biases.resize(params_.biases.size());

		params.weights = 0;
		params.biases = 0;
	}

	void add_params(double alpha, const Parameters& params)
	{
		params_.weights += alpha * params.weights;
		params_.biases += alpha * params.biases;
	}

	template<class Loss_fn, class Grad>
	void check_gradient(const Loss_fn& loss_fn, const Grad& grad, double d = 1e-4)
	{
		const auto loss0 = loss_fn();
		for (std::size_t col = 0; col < params_.weights.cols(); ++col)
			for (std::size_t row = 0; row < params_.weights.rows(); ++row)
			{
				params_.weights(row, col) += d;
				const auto fd_grad = (loss_fn() - loss0) / d;
				params_.weights(row, col) -= d;

				const auto an_grad = grad.weights(row, col);

				if (row == 2 && col == 1)
				{
					std::cout << "Expected value: " << an_grad << '\n'
							  << "Finite difference value: " << fd_grad << std::endl;
				}

				const auto beta = std::abs(fd_grad - an_grad) / (std::abs(fd_grad) + std::abs(an_grad));
				// if (beta > 100 * d && std::abs(an_grad) > 100 * d && std::abs(fd_grad) > 100 * d)
				// {
				// 	std::cout << "Expected value: " << an_grad << '\n'
				// 			  << "Finite difference value: " << fd_grad << std::endl;
				// 	throw std::runtime_error(name() + ": bad weights gradient");
				// }
			}

		for (std::size_t row = 0; row < params_.weights.rows(); ++row)
		{
			params_.biases[row] += d;
			const auto fd_grad = (loss_fn() - loss0) / d;
			params_.biases[row] -= d;

			const auto an_grad = grad.biases[row];
			const auto beta = std::abs(fd_grad - an_grad) / (std::abs(fd_grad) + std::abs(an_grad));
			// if (beta > 100 * d && std::abs(an_grad) > 100 * d && std::abs(fd_grad) > 100 * d)
			// {
			// 	std::cout << "Expected value: " << an_grad << '\n'
			// 			  << "Finite difference value: " << fd_grad << std::endl;
			// 	throw std::runtime_error(name() + ": bad biases gradient");
			// }
		}
	}

protected:
	void init_storage(std::size_t n_rows, std::size_t n_cols)
	{
		params_.weights.resize(n_rows, n_cols);
		params_.biases.resize(n_rows);
	}

	std::size_t n_trainable_params() const
	{
		return params_.weights.size() + params_.biases.size();
	}

protected:
	Parameters params_;
};
