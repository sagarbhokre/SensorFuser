#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define DEBUG (0)
#define ENABLE_RADAR (1)
#define ENABLE_LASER (1)
#define THRESHOLD (0.0001)
/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.03, 0, 0,
              0, 0.0000009, 0,
              0, 0, 0.09;

  H_laser_ << 1,0,0,0,
              0,1,0,0;

  /* Set the process and measurement noises */
  noise_ax =   1.90;
  noise_ay =   1.90;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;
  if(DEBUG) cout << "dt: " << dt << endl;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement

    /** Initialize the state ekf_.x_ with the first measurement */

    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    ekf_.F_ = MatrixXd(4,4);
    /* Create the covariance matrix */
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      if((measurement_pack.raw_measurements_[0] < THRESHOLD) &&
         (measurement_pack.raw_measurements_[1] < THRESHOLD) &&
         (measurement_pack.raw_measurements_[2] < THRESHOLD)) {
        return;
      }

      /* Convert radar from polar to cartesian coordinates and initialize state */
      double r = measurement_pack.raw_measurements_[0];
      double th = measurement_pack.raw_measurements_[1];
      double rd = measurement_pack.raw_measurements_[2];
      ekf_.x_ << r*cos(th), r*sin(th), rd*cos(th) , rd*sin(th);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      if((measurement_pack.raw_measurements_[0] < THRESHOLD) &&
         (measurement_pack.raw_measurements_[1] < THRESHOLD)) {
        return;
      }

      /* Initialize state */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /* Update the state transition matrix F according to the new elapsed time.*/
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // Set the process covariance matrix Q
  float sqr_dt = dt*dt;
  float cub_dt = pow(dt, 3);
  float qua_dt = pow(dt, 4);
  ekf_.Q_ = MatrixXd(4, 4);

  ekf_.Q_ << qua_dt/4*noise_ax, 0,                  cub_dt/2*noise_ax,  0,
             0,                 qua_dt/4*noise_ay,  0,                  cub_dt/2*noise_ay,
             cub_dt/2*noise_ax, 0,                  sqr_dt*noise_ax,    0,
             0,                 cub_dt/2*noise_ay,  0,                  sqr_dt*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    if(!ENABLE_RADAR) return;
    // Radar updates
    Tools tools;
    ekf_.H_  = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_  = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    if(!ENABLE_LASER) return;
    // Laser updates
    ekf_.H_  = H_laser_;
    ekf_.R_  = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  if(DEBUG) {
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
  }
}
