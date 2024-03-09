#pragma once
#include <vector>

class layer
{
protected:
	std::vector<double> neurons_;

public:
	explicit layer(const size_t size) : neurons_(size)
	{
	}

	explicit layer(std::vector<double> values) : neurons_{std::move(values)}
	{
	}

	layer(const layer& other) = default;
	layer(layer&& other) noexcept = default;
	layer& operator=(const layer& other) = default;
	layer& operator=(layer&& other) noexcept = default;
	virtual ~layer() = default;

	const auto& neurons() const noexcept { return neurons_; }
	double operator[](const int index) const { return neurons_[index]; }
	double& operator[](const int index) { return neurons_[index]; }
	auto size() const noexcept { return neurons_.size(); }
};
