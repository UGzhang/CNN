#ifndef SRC_LOSS_H_
#define SRC_LOSS_H_

#include "./utils.h"

class Loss {
 protected:
  double loss;  // value of loss
  Matrix grad_bottom;  // gradient w.r.t input

 public:
  virtual ~Loss() {}

  virtual void evaluate(const Matrix& pred, const Matrix& target) = 0;
  virtual double output() { return loss; }
  virtual const Matrix& back_gradient() { return grad_bottom; }
};

#endif  // SRC_LOSS_H_
