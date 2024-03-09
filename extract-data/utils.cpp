#include "utils.h"
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace utils
{
	using namespace std;

	vector<vector<double>> to_categorical(vector<uint8_t> vec)
	{
		auto unique = vector<int>{0,1,2,3,4,5,6,7,8,9};
//		for (const auto& val : vec)
//		{
//			if (find(begin(unique), end(unique), val) == end(unique))
//			{
//				unique.push_back(val);
//			}
//		}
//		sort(begin(unique), end(unique));




		auto result = vector<vector<double>>{};
		result.reserve(vec.size());
		for (const auto& val : vec)
		{
			auto item = vector<double>(unique.size(), 0.0);
			const auto idx = distance(begin(unique), find(begin(unique), end(unique), val));
			item[idx] = 1.0;
			result.push_back(item);
		}

		return result;
	}

	vector<vector<double>> normalize_image_set(const vector<vector<uint8_t>>& images)
	{
		auto normalized = vector<vector<double>>(images.size());
#pragma omp parallel for
		for (auto i = 0; i < images.size(); i++)
		{
			normalized[i] = vector<double>(images[i].size());
			for (auto j = 0; j < images[i].size(); j++)
			{
				normalized[i][j] = images[i][j] / 255.0;
			}
		}

		return normalized;
	}

	vector<double> normal_distribution_vector(const int size, const double mean, const double sigma)
	{
		std::default_random_engine generator;
		std::normal_distribution<double> distribution(mean, sigma);

		auto normal = vector<double>(size);
		for (auto& value : normal)
		{
			value = distribution(generator);
		}

		return normal;
	}

	void print_duration(const long long milliseconds)
	{
		std::cout << "Time: "
			<< std::right
			<< std::setfill('0')
			<< std::setw(2)
			<< milliseconds / (1000 * 60)
			<< ":"
			<< std::setw(2)
			<< (milliseconds / 1000) % 60
			<< "."
			<< std::setw(3)
			<< milliseconds % 1000
			<< '\n';
	}
}
