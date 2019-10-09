#pragma once
#include <esl/dense.hpp>

#include <cstddef>
#include <random>

class Random_init
{
public:
	Random_init(double max) : max_(max)
	{
// 		std::random_device rd;
// 		generator_.seed(rd());
		generator_.seed(0);
	}

	template<std::size_t rows, std::size_t cols>
	void operator()(esl::Matrix_d<rows, cols>& matrix)
	{
		std::uniform_real_distribution<double> distr(-max_, max_);
		matrix = esl::Random_matrix(matrix.rows(), matrix.cols(), distr, generator_);
	}

private:
	const double max_;
	std::mt19937 generator_;
};
