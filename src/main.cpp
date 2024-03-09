#include <filesystem>
#include <iostream>
#include "operate_config.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "layer.h"
#include "fully_connected.h"
#include "relu.h"
#include "softmax.h"
#include "loss.h"
#include "cross_entropy_loss.h"
#include "mnist.h"
#include "network.h"
#include "sgd.h"

using std::cout;

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

    MNIST dataset{};
    dataset.readData(rel_path_train_images, true);
    dataset.readData(rel_path_test_images, false);
    dataset.readLabel(rel_path_train_labels, true);
    dataset.readLabel(rel_path_test_labels, false);

    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

    Network fcnn;
    Layer* fc1 = new FullyConnected(dim_in,hidden_size);
    Layer* relu1 = new ReLU;
    Layer* fc2 = new FullyConnected(hidden_size,10);
    Layer* softmax = new Softmax;
    fcnn.add_layer(fc1);
    fcnn.add_layer(relu1);
    fcnn.add_layer(fc2);
    fcnn.add_layer(softmax);

    Loss* loss = new CrossEntropy;
    fcnn.add_loss(loss);
    // train & test
    SGD opt(learning_rate, 5e-4, 0.9, true);
    // SGD opt(0.001);
    for (int epoch = 0; epoch < num_epochs; epoch ++) {
        shuffle_data(dataset.train_data, dataset.train_labels);
        for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
            int ith_batch = start_idx / batch_size;
            Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                                      std::min(batch_size, n_train - start_idx));
            Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                            std::min(batch_size, n_train - start_idx));
            Matrix target_batch = one_hot_encode(label_batch, 10);
//            if (false && ith_batch % 10 == 1) {
//                std::cout << ith_batch << "-th grad: " << std::endl;
//                fcnn.check_gradient(x_batch, target_batch, 10);
//            }
            fcnn.forward(x_batch);
            fcnn.backward(x_batch, target_batch);
            // display
            if (ith_batch % 100 == 0 && n_train == 60000) {
                std::cout << ith_batch << "-th batch, loss: " << fcnn.get_loss()
                          << std::endl;
            }
            // optimize
            fcnn.update(opt);
        }
        // test
        fcnn.forward(dataset.test_data);
        float acc = compute_accuracy(fcnn.output(), dataset.test_labels, rel_path_log_file);

        if( n_train == 60000){
            std::cout << std::endl;
            std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
            std::cout << std::endl;
        }

    }
    return 0;
}
