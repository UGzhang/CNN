#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include "./utils.h"

class Optimizer {
 protected:
  double lr;  // learning rate
  double decay;  // weight decay factor (default: 0)

 public:
  explicit Optimizer(double lr = 0.01, double decay = 0.0) :
                     lr(lr), decay(decay) {}
  virtual ~Optimizer() {}

  virtual void update(Vector::AlignedMapType& w,
                      Vector::ConstAlignedMapType& dw) = 0;
};

#endif  // SRC_OPTIMIZER_H_
