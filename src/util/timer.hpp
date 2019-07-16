#pragma once
#include <chrono>
#include <ratio>

class Timer
{
private:
	using Clock = std::chrono::high_resolution_clock;

	template<typename Ratio = std::ratio<1>>
	auto interval() const
	{
		return std::chrono::duration<double, Ratio>(stop_ - start_).count();
	}

public:
	void start()
	{
		start_ = Clock::now();
	}

	void stop()
	{
		stop_ = Clock::now();
	}

	auto nsec() const
	{
		return interval<std::nano>();
	}

	auto usec() const
	{
		return interval<std::micro>();
	}

	auto msec() const
	{
		return interval<std::milli>();
	}

	auto sec() const
	{
		return interval();
	}

	auto min() const
	{
		return interval() / 60;
	}

private:
	std::chrono::time_point<Clock> start_, stop_;
};
