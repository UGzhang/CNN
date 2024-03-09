#include "../extract-data/utils.h"
#include "../extract-data/mnist_reader.h"
#include "sequantial.h"
#include <filesystem>
#include <iostream>
#include "operate_config.h"

using std::cout;
namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    if(argc != 2) exit(0);
    std::string Config = argv[1];

    ConfigHandle.init(Config);

    int num_epochs = ConfigHandle.read("num_epochs", 0);
    int batch_size = ConfigHandle.read("batch_size", 0);
    int hidden_size = ConfigHandle.read("hidden_size", 0);
    double learning_rate = ConfigHandle.read("learning_rate", 0.0);
    std::string rel_path_train_images = ConfigHandle.read("rel_path_train_images", std::string{});
    std::string rel_path_train_labels = ConfigHandle.read("rel_path_train_labels", std::string{});
    std::string rel_path_test_images = ConfigHandle.read("rel_path_test_images", std::string{});
    std::string rel_path_test_labels = ConfigHandle.read("rel_path_test_labels", std::string{});
    std::string rel_path_log_file = ConfigHandle.read("rel_path_log_file", std::string{});


		cout << "Loading data...\n";

//		auto [train_images, train_labels] = mnist_reader::read(
//			rel_path_train_images,
//			rel_path_train_labels
//		);

        auto train_images = mnist_reader::readImage(rel_path_train_images);
        auto train_labels = mnist_reader::readLable(rel_path_train_labels);


//		auto [test_images, test_labels] = mnist_reader::read(
//			rel_path_test_images,
//			rel_path_test_labels
//		);

    auto test_images = mnist_reader::readImage(rel_path_test_images);
    auto test_labels = mnist_reader::readLable(rel_path_test_labels);

		cout << "Loaded.\n\n";

		//const auto [random_images, random_labels] = utils::random_subset(train_images, train_labels, 60000);

		const auto x_train = utils::normalize_image_set(train_images);
		const auto y_train = utils::to_categorical(train_labels);

		const auto x_test = utils::normalize_image_set(test_images);
		const auto y_test = utils::to_categorical(test_labels);

		const int input_size = x_train[0].size();
		const int num_classes = y_train[0].size();

		auto model = sequential{input_size, hidden_size, num_classes};
		model.fit(x_train, y_train, x_test, y_test,  rel_path_log_file, num_epochs, learning_rate, batch_size);

	return 0;
}
