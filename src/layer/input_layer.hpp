#pragma once
#include <es_la/dense.hpp>

#include <iostream>

class Input_layer
{
public:
	Input_layer()
	{
		output_.resize(204);
	}

	template<class Strategy>
	void init(Strategy&& init_strategy)
	{
		// HACK
		init_strategy(output_);
	}

	template<class Vector>
	void set(const Vector& values)
	{
		output_ = values;
	}

	const es_la::Vector_xd& values() const
	{
		return output_;
	}

	std::string info_string() const
	{
		std::string info = "Input connected layer\n";
		info += "  Number of nodes: " + std::to_string(output_.size()) + "\n";
		return info;
	}

	// TODO : remove
	void init_gradients()
	{}

	// TODO : remove
	void add_gradients(double)
	{}

private:
	es_la::Vector_xd output_;
};
