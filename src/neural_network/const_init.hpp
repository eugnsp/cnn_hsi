#pragma once
#include <es_la/dense.hpp>

#include <cstddef>

class Const_init
{
public:
	Const_init(double value) : value_(value)
	{}

	template<std::size_t rows, std::size_t cols>
	void operator()(es_la::Matrix_d<rows, cols>& matrix)
	{
		matrix = value_;
	}

private:
	const double value_;
};