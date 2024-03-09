#include "fc_layer.h"
#include "../extract-data/utils.h"

void fc_layer::init_weights(const int left_size, const int right_size, const double mean, const double sigma)
{
	weights_ = std::vector<std::vector<variable>>(right_size);
	derivative_ = std::vector<double>(right_size, 0.0);
#pragma omp parallel for
	for (auto i = 0; i < right_size; i++)
	{
		const auto random_vec = utils::normal_distribution_vector(left_size, mean, sigma);
		weights_[i] = std::vector<variable>(left_size);
		for (auto j = 0; j < left_size; j++)
		{
			weights_[i][j] = {random_vec[j], 0};
		}
	}
}

fc_layer::fc_layer(const int prev_size, const int size)
	: layer(size)
{
	if (prev_size == 0u) return;
	init_weights(prev_size, size);
}

void fc_layer::adjust_weights(const double learning_rate)
{
#pragma omp parallel for
	for (auto i = 0; i < weights_.size(); i++)
	{
		for (auto j = 0; j < weights_[i].size(); j++)
		{
			weights_[i][j].value -= learning_rate * weights_[i][j].delta;
			weights_[i][j].delta = 0.0;
		}
	}
	bias_.value -= learning_rate * bias_.delta;
	bias_.delta = 0.0;
}
#include <iostream>
void fc_layer::forward(const layer& prev)
{
#pragma omp parallel for

    double minValue = 0;
	for (auto i = 0; i < neurons_.size(); i++)
	{
		neurons_[i] = bias_.value;
		for (auto j = 0; j < prev.size(); j++)
		{
			neurons_[i] += prev[j] * weights_[i][j].value;

		}
	}
	apply_activation_function();
}

void fc_layer::backward(const layer& prev, const fc_layer& next)
{
	compute_derivatives(next);
#pragma omp parallel for
	for (auto i = 0; i < weights_.size(); i++)
	{
		for (auto j = 0; j < weights_[i].size(); j++)
		{
			weights_[i][j].delta += derivative_[i] * prev[j];
		}
		bias_.delta += derivative_[i];
	}
}
