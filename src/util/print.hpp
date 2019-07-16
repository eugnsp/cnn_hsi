#pragma once
#include <cstddef>
#include <iostream>

template<class Expr>
void print(const Expr& expr)
{
	for (std::size_t row = 0; row < expr.rows(); ++row)
	{
		for (std::size_t col = 0; col < expr.cols(); ++col)
			std::cout << expr(row, col) << ' ';
		std::cout << std::endl;
	}
}
