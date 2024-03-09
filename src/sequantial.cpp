#include "sequantial.h"
#include "../extract-data/utils.h"
#include "fc_layer_empty.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>

bool sequential::is_accurate(const vec& target, const vec& output)
{
	const auto max = std::max_element(begin(output), end(output));
	return target[std::distance(begin(output), max)];
}

sequential::sequential(const size_t input_size, const size_t hidden_size, const size_t output_size)
	: input_{input_size}, hidden_{input_size, hidden_size}, output_{hidden_size, output_size}
{
}

void sequential::forward_pass(vec x)
{
	input_ = layer{std::move(x)};
	hidden_.forward(input_);
	output_.forward(hidden_);
}

void sequential::backward_pass(vec y)
{
	output_.backward(hidden_, fc_layer_empty{std::move(y)});
	hidden_.backward(input_, output_);
}

void sequential::adjust_weights(const double learning_rate)
{
	hidden_.adjust_weights(learning_rate);
	output_.adjust_weights(learning_rate);
}

double sequential::cross_entropy(const vec& target, const vec& output)
{
	auto sum = 0.0;

#pragma omp parallel for reduction(+:sum)
	for (auto i = 0; i < target.size(); i++)
	{
		sum += target[i] * log(output[i]);
	}

	return -sum;
}

std::tuple<double, double> sequential::evaluate(
        const matrix& x,const matrix& y,
        size_t batch_size, std::string& log_file)
{
	auto loss = 0.0;
	auto accuracy = 0.0;
    int batch = 0;
    std::ofstream output(log_file);
	for (auto i = 0; i < x.size(); i++)
	{
		forward_pass(x[i]);
        auto label = y[i];
        auto pred = output_.neurons();
		loss += cross_entropy(label, pred);
		accuracy += is_accurate(label, pred);

        if (i % batch_size == 0){
            output << "Current batch: " << batch << "\n";
            batch++;
        }
        auto labelIter = std::find(label.begin(), label.end(),1);
        auto posLabel = std::distance(label.begin(), labelIter);

        auto predIter = std::max_element(pred.begin(), pred.end());
        auto posPred = std::distance(pred.begin(), predIter);
        output << " - image " << i << ": Prediction=" << posPred << ". Label=" << posLabel << std::endl;

	}
	return {loss / x.size(), accuracy / x.size()};
}

void sequential::fit(const matrix& x_train, const matrix& y_train,
                     const matrix& x_test, const matrix& y_test,
                     std::string& log_file,
                     size_t epoch_count, double learning_rate, size_t batch_size)
{
	const auto train_size = static_cast<int>(x_train.size());
	const auto batch_count = static_cast<int>(std::ceil(train_size / double(batch_size)));

	const auto start = std::chrono::system_clock::now();
    learning_rate *= 200;
	for (auto epoch = 1; epoch <= epoch_count; epoch++)
	{
		for (auto batch = 0; batch < batch_count; batch++)
		{
			for (auto test = 0; test < batch_size && batch * batch_size + test < train_size; test++)
			{
				forward_pass(x_train[batch * batch_size + test]);
				backward_pass(y_train[batch * batch_size + test]);
			}
			adjust_weights(learning_rate / double(batch_size));
			std::cout << "\repoch #" << std::setw(2) << std::left << epoch
				<< " batch #" << batch + 1 << "/" << batch_count;
		}

		const auto [loss, acc] = evaluate(x_test, y_test, batch_size, log_file);
		std::cout << "\repoch #" << std::setw(2) << std::left << epoch
			<< " loss: " << loss << ", acc: " << acc << '\n';
	}

	const auto finish = std::chrono::system_clock::now();

	utils::print_duration(std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
}
