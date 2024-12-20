#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include "Eigen/Core"
#include <algorithm>
#include <iostream>
#include <random>
#include <fstream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<double, 1, Eigen::Dynamic> RowVector;

static std::default_random_engine generator;

// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(double* arr, int n, double mu, double sigma) {
  std::normal_distribution<double> distribution(mu, sigma);
  for (int i = 0; i < n; i ++) {
    arr[i] = distribution(generator);
  }
}

// shuffle cols of matrix
inline void shuffle_data(Matrix& data, Matrix& labels) {
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.cols());
  perm.setIdentity();
  std::random_shuffle(perm.indices().data(), perm.indices().data()
                      + perm.indices().size());
  data = data * perm;  // permute columns
  labels = labels * perm;
}

// encode discrete values to one-hot values
inline Matrix one_hot_encode(const Matrix& y, int n_value) {
  int n = y.cols();
  Matrix y_onehot = Matrix::Zero(n_value, n);
  for (int i = 0; i < n; i ++) {
    y_onehot(int(y(i)), i) = 1;
  }
  return y_onehot;
}

// classification accuracy
inline double compute_accuracy(const Matrix& preditions, const Matrix& labels, const std::string& output) {
  int n = preditions.cols();
  double acc = 0;
  int batch = 0;

  std::ofstream log(output);
  for (int i = 0; i < n; i ++) {
      if(i %  labels.size() == 0){
          log << "Current batch: " << batch << "\n";
          batch++;
      }

    Matrix::Index max_index;
    preditions.col(i).maxCoeff(&max_index);
    acc += int(max_index) == labels(i);
      log << " - image " << i << ": Prediction=" << int(max_index) << ". Label=" << labels(i) << std::endl;


  }
  return acc / n;
}

#endif  // SRC_UTILS_H_
