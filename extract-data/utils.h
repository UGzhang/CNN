#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace utils
{
	extern std::vector<std::vector<double>> to_categorical(std::vector<uint8_t> vec);
	extern std::vector<std::vector<double>> normalize_image_set(const std::vector<std::vector<uint8_t>>& images);
	extern std::vector<double> normal_distribution_vector(int size, double mean, double sigma);
	extern void print_duration(long long milliseconds);

	template <typename X, typename Y>
	static auto random_subset(const std::vector<X>& x, const std::vector<Y>& y, size_t count = 0)
	-> std::tuple<std::vector<X>, std::vector<Y>>
	{
		if (count == 0)
			count = x.size();
		if (count > x.size() || count > y.size())
			throw std::runtime_error("subset size is too big");

		auto indices = vector<int>(x.size());
		for (auto i = 0u; i < indices.size(); i++)
		{
			indices[i] = i;
		}

		auto generator = std::mt19937(std::random_device{}());
		std::shuffle(begin(indices), end(indices), generator);

		auto random_x = std::vector<X>(count);
		auto random_y = std::vector<Y>(count);
		for (auto i = 0u; i < count; i++)
		{
			random_x[i] = x[indices[i]];
			random_y[i] = y[indices[i]];
		}

		return {random_x, random_y};
	}
}
