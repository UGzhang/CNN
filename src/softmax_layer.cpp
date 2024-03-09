#include "softmax_layer.h"
#include <cmath>

void softmax_layer::apply_activation_function()
{
	auto sum = 0.0;

#pragma omp parallel for reduction(+:sum)
	for (auto i = 0; i < neurons_.size(); i++)
	{
		neurons_[i] = exp(neurons_[i]);
		sum += neurons_[i];
	}

#pragma omp parallel for
	for (auto i = 0; i < neurons_.size(); i++)
	{
		neurons_[i] /= sum;
	}
}

void softmax_layer::compute_derivatives(const fc_layer& next)
{
#pragma omp parallel for
	for (auto i = 0; i < neurons_.size(); i++)
	{
		derivative_[i] = neurons_[i] - next[i];
	}
}

softmax_layer::softmax_layer(const size_t prev_size, const size_t size)
	: fc_layer{prev_size, size}
{
}
