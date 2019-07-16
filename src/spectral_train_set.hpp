#pragma once
#include <es_la/dense.hpp>

#include <cassert>
#include <cstddef>
#include <fstream>
#include <string>

struct Spectral_train_set
{
	std::size_t size;
	std::size_t spectrum_size;
	std::size_t n_label_values;
	es_la::Vector_x<std::size_t> labels;
	es_la::Matrix_xd data;
};

Spectral_train_set read_train_set(const std::string& data_file_name, const std::string& labels_file_name)
{
	std::ifstream data_file;
	data_file.exceptions(std::ifstream::badbit | std::ifstream::failbit);
	data_file.open(data_file_name);

	Spectral_train_set train_set;
	data_file >> train_set.spectrum_size >> train_set.size;

	train_set.data.resize(train_set.spectrum_size, train_set.size);
	for (std::size_t s = 0; s < train_set.spectrum_size; ++s)
		for (std::size_t i = 0; i < train_set.size; ++i)
			data_file >> train_set.data(s, i);

	std::ifstream labels_file;
	labels_file.exceptions(std::ifstream::badbit | std::ifstream::failbit);
	labels_file.open(labels_file_name);

	std::size_t labels_size;
	labels_file >> train_set.n_label_values >> labels_size;
	assert(labels_size == train_set.size);

	train_set.labels.resize(labels_size);
	for (std::size_t i = 0; i < labels_size; ++i)
		labels_file >> train_set.labels[i];

	return train_set;
}
