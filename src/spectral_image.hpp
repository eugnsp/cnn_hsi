#pragma once
#include <es_la/dense.hpp>

#include <cstddef>
#include <fstream>
#include <string>

struct Spectral_image
{
	std::size_t rows;
	std::size_t cols;
	std::size_t spectrum_size;
	es_la::Matrix_xd data;
};

Spectral_image read_image(const std::string& file_name)
{
	std::ifstream file;
	file.exceptions(std::ifstream::badbit | std::ifstream::failbit);
	file.open(file_name);

	Spectral_image image;
	file >> image.spectrum_size >> image.rows >> image.cols;

	image.data.resize(image.spectrum_size, image.rows * image.cols);
	for (std::size_t col = 0; col < image.cols; ++col)
		for (std::size_t row = 0; row < image.rows; ++row)
		{
			const auto index = row + col * image.rows;
			for (std::size_t s = 0; s < image.spectrum_size; ++s)
				file >> image.data(s, index);
		}

	return image;
}

