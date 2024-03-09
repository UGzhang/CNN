#include "relu_layer.h"
#include <cmath>

void relu_layer::apply_activation_function()
{
#pragma omp parallel for
	for (auto i = 0; i < neurons_.size(); i++)
	{
//		neurons_[i] = tanh(neurons_[i]);
        neurons_[i] = std::max(0.0, neurons_[i]);
	}
}

void relu_layer::compute_derivatives(const fc_layer& next)
{
#pragma omp parallel for
	for (auto i = 0; i < neurons_.size(); i++)
	{
        derivative_[i] = 0.;
        if(neurons_[i] > 0.){
            for (auto k = 0; k < next.size(); k++)
            {
                derivative_[i] += next.weight(k, i) * next.derivative(k);
            }
        }


	}
}

relu_layer::relu_layer(const int prev_size, const int size)
	: fc_layer{prev_size, size}
{
}
