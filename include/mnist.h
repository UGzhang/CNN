#ifndef SRC_MNIST_H_
#define SRC_MNIST_H_

#include <fstream>
#include <iostream>
#include <string>
#include "./utils.h"

class MNIST {

 public:
  Matrix train_data;
  Matrix train_labels;
  Matrix test_data;
  Matrix test_labels;

  void read_mnist_data(std::string filename, Matrix& data);
  void read_mnist_label(std::string filename, Matrix& labels);

  void readData(const std::string& path, bool isTrain);
  void readLabel(const std::string& path, bool isTrain);
};

#endif  // SRC_MNIST_H_
