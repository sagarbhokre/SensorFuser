#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define THRESHOLD (0.00001)

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /* predict the state  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::ComputeEstimate(VectorXd &z_pred, const VectorXd &z)
{
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  ComputeEstimate(z_pred, z);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  float px = x_(0), py = x_(1), vx = x_(2), vy = x_(3);

  VectorXd z_pred  = VectorXd(3);

  /* Handle division by 0 possibility */
  if((px < THRESHOLD) && (py < THRESHOLD)) {
    z_pred << 1, 0, 1;
  }
  else {
    float t1 = sqrt(px*px + py*py);
    z_pred << t1, atan2(py,px), (px*vx + py*vy)/t1;
  }
  ComputeEstimate(z_pred, z);
}
