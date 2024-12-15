#include <filesystem>
#include <iostream>
#include "operate_config.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <memory>

#include "param.h"
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

void initDataSet(const char* config, Param& param, MNIST& dataset);
void initNetwork(const Param& param, Network& fnn);
void train(const Param& param, MNIST& dataset, Network& fnn);
void test(const Param& param, MNIST& dataset, Network& fnn);

int main(int argc, char* argv[])
{
    if(argc != 2 && argc != 4) exit(0);

    if(argc == 2){

        MNIST dataset{};
        Param param{};
        initDataSet(argv[1], param, dataset);

        Network fnn;
        initNetwork(param, fnn);

        train(param, dataset, fnn);
        test(param, dataset, fnn);


    } else{

        const std::string& inputPath =  argv[1];
        const std::string& outputPath = argv[2];
        const int index = atoi(argv[3]);
        std::ofstream output(outputPath);
        MNIST dataset{};
        if(inputPath.find("idx3") != std::string::npos){
            dataset.readData(inputPath, true);
            int dim_in = dataset.train_data.rows();
            output << 2 << "\n";
            output << 28 << "\n";
            output << 28 << "\n";
            auto img = dataset.train_data.col(index);

            for (int i = 0; i < dim_in; ++i) {
                output << img(i,0) << "\n";
            }
            output << std::endl;

        } else{
            dataset.readLabel(inputPath, true);

            int label = dataset.train_labels(0,index);
            Matrix y_onehot = Matrix::Zero(10, 1);
            y_onehot(label,0) = 1;
            output << 1 << "\n";
            output << 10 << "\n";

            output << y_onehot;
            output << std::endl;
        }
    }

    return 0;
}



void train(const Param& param, MNIST& dataset, Network& fnn)
{
    SGD opt(param.learning_rate, 5e-4, 0.9, true);
    for (int epoch = 0; epoch < param.num_epochs; epoch ++) {
        shuffle_data(dataset.train_data, dataset.train_labels);
        for (int start_idx = 0; start_idx < param.n_train; start_idx += param.batch_size) {

            Matrix x_batch = dataset.train_data.block(0, start_idx, param.dim_in,
                                                      std::min(param.batch_size, param.n_train - start_idx));
            Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                            std::min(param.batch_size, param.n_train - start_idx));
            Matrix target_batch = one_hot_encode(label_batch, param.dim_out);

//            int ith_batch = start_idx / param.batch_size;
//            if (ith_batch % 10 == 0) {
//                std::cout << ith_batch << "-th grad: " << std::endl;
//                fcnn.check_gradient(x_batch, target_batch, 10);
//            }
            fnn.forward(x_batch);
            fnn.backward(x_batch, target_batch);
            // display
//                if (ith_batch % 100 == 0 && n_train == 60000) {
//                    std::cout << ith_batch << "-th batch, loss: " << fcnn.get_loss()
//                              << std::endl;
//                }
            // optimize
            fnn.update(opt);
        }


    }
}

void test(const Param& param, MNIST& dataset, Network& fnn){
    fnn.forward(dataset.test_data);
    float acc = compute_accuracy(fnn.output(), dataset.test_labels, param.rel_path_log_file);
    std::cout << " test acc: " << acc << std::endl;
}



void initNetwork(const Param& param, Network& fnn){
    Layer* fc1 = new FullyConnected(param.dim_in,param.hidden_size);
    Layer* relu1 = new ReLU;
    Layer* fc2 = new FullyConnected(param.hidden_size,10);
    Layer* softmax = new Softmax;
    fnn.add_layer(fc1);
    fnn.add_layer(relu1);
    fnn.add_layer(fc2);
    fnn.add_layer(softmax);

    Loss* loss = new CrossEntropy;
    fnn.add_loss(loss);
}

void initDataSet(const char* config, Param& param, MNIST& dataset){
    ConfigHandle.init(config);

    param.num_epochs = ConfigHandle.read("num_epochs", 0);
    param.batch_size = ConfigHandle.read("batch_size", 0);
    param.hidden_size = ConfigHandle.read("hidden_size", 0);
    param.learning_rate = ConfigHandle.read("learning_rate", 0.0);
    param.rel_path_train_images = ConfigHandle.read("rel_path_train_images", std::string{});
    param.rel_path_train_labels = ConfigHandle.read("rel_path_train_labels", std::string{});
    param.rel_path_test_images = ConfigHandle.read("rel_path_test_images", std::string{});
    param.rel_path_test_labels = ConfigHandle.read("rel_path_test_labels", std::string{});
    param.rel_path_log_file = ConfigHandle.read("rel_path_log_file", std::string{});

    dataset.readData(param.rel_path_train_images, true);
    dataset.readData(param.rel_path_test_images, false);
    dataset.readLabel(param.rel_path_train_labels, true);
    dataset.readLabel(param.rel_path_test_labels, false);

    param.n_train = dataset.train_data.cols();
    param.dim_in = dataset.train_data.rows();
    param.dim_out = 10;

#ifdef DEBUG
    std::cout << "mnist train number: " << param.n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
#endif
}