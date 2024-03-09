#pragma once
#include "fc_layer.h"

class relu_layer : public fc_layer
{
	void apply_activation_function() override;
	void compute_derivatives(const fc_layer& next) override;

public:
	explicit relu_layer(size_t prev_size, size_t size);
};
