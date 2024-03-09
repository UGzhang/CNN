#pragma once
#include "layer.h"

class fc_layer : public layer
{
	struct variable
	{
		double value;
		double delta;
	};

	virtual void apply_activation_function() = 0;
	virtual void compute_derivatives(const fc_layer& next) = 0;
	void init_weights(int left_size, int right_size, double mean = 0.0, double sigma = 0.3 );

protected:
	std::vector<double> derivative_;
	std::vector<std::vector<variable>> weights_;
	variable bias_ = {0, 0};

public:
	explicit fc_layer(int prev_size, int size);

	void adjust_weights(double learning_rate);
	void forward(const layer& prev);
	void backward(const layer& prev, const fc_layer& next);

	double weight(const int cur, const int prev) const { return weights_[cur][prev].value; }
	double derivative(const int idx) const { return derivative_[idx]; }
};
