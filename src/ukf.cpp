#include "kalman_positioning/ukf.hpp"
#include <iostream>
#include <map>

/**
 * STUDENT ASSIGNMENT: Unscented Kalman Filter Implementation
 * 
 * This file contains placeholder implementations for the UKF class methods.
 * Students should implement each method according to the UKF algorithm.
 * 
 * Reference: Wan, E. A., & Van Der Merwe, R. (2000). 
 * "The Unscented Kalman Filter for Nonlinear Estimation"
 */

// ============================================================================
// CONSTRUCTOR
// ============================================================================

/**
 * @brief Initialize the Unscented Kalman Filter
 * 
 * STUDENT TODO:
 * 1. Initialize filter parameters (alpha, beta, kappa, lambda)
 * 2. Initialize state vector x_ with zeros
 * 3. Initialize state covariance matrix P_ 
 * 4. Set process noise covariance Q_
 * 5. Set measurement noise covariance R_
 * 6. Calculate sigma point weights for mean and covariance
 */
UKF::UKF(double process_noise_xy, double process_noise_theta,
         double measurement_noise_xy, int num_landmarks)
    : nx_(5), nz_(2) {
    
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    x_ = Eigen::VectorXd::Zero(nx_);
    

    Q_ = Eigen::MatrixXd::Zero(nx_, nx_);
    Q_(0, 0) = process_noise_xy;
    Q_(1, 1) = process_noise_xy;
    Q_(2, 2) = process_noise_theta;
    Q_(3, 3) = process_noise_xy;  
    Q_(4, 4) = process_noise_xy; 
    

    R_ = Eigen::MatrixXd::Zero(nz_, nz_);
    R_(0, 0) = measurement_noise_xy;
    R_(1, 1) = measurement_noise_xy;
    
    P_ = Eigen::MatrixXd::Identity(nx_, nx_);
    
    lambda_ = ALPHA * ALPHA * (nx_ + KAPPA) - nx_;
    gamma_ = std::sqrt(nx_ + lambda_);
    
    int sigma_points_len = 2 * nx_ + 1;
    Wm_.resize(sigma_points_len);
    Wc_.resize(sigma_points_len);
    
    Wm_[0] = lambda_ / (nx_ + lambda_);
    Wc_[0] = lambda_ / (nx_ + lambda_) + (1.0 - ALPHA * ALPHA + BETA);
    
    double weight = 1.0 / (2.0 * (nx_ + lambda_));
    for (int i = 1; i < sigma_points_len; ++i) {
        Wm_[i] = weight;
        Wc_[i] = weight;
    }
}

// ============================================================================
// SIGMA POINT GENERATION
// ============================================================================

/**
 * @brief Generate sigma points from mean and covariance
 * 
 * STUDENT TODO:
 * 1. Start with the mean as the first sigma point
 * 2. Compute Cholesky decomposition of covariance
 * 3. Generate 2*n symmetric sigma points around the mean
 */
std::vector<Eigen::VectorXd> UKF::generateSigmaPoints(const Eigen::VectorXd& mean,
                                                       const Eigen::MatrixXd& cov) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    std::vector<Eigen::VectorXd> sigma_points;
    sigma_points.push_back(mean);
    
    // Regularize covariance to ensure positive definiteness
    Eigen::MatrixXd cov_reg = cov;
    double jitter = 1e-9;
    cov_reg += Eigen::MatrixXd::Identity(nx_, nx_) * jitter;
    
    Eigen::LLT<Eigen::MatrixXd> llt(cov_reg);
    Eigen::MatrixXd L = llt.matrixL();
    
    for (int i = 0; i < nx_; ++i) {
        sigma_points.push_back(mean + gamma_ * L.col(i));
        sigma_points.push_back(mean - gamma_ * L.col(i));
    }
    
    return sigma_points;
}

// ============================================================================
// PROCESS MODEL
// ============================================================================

/**
 * @brief Apply motion model to a state vector
 * 
 * STUDENT TODO:
 * 1. Updates position: x' = x + dx, y' = y + dy
 * 2. Updates orientation: theta' = theta + dtheta (normalized)
 * 3. Updates velocities: vx' = dx/dt, vy' = dy/dt
 */
Eigen::VectorXd UKF::processModel(const Eigen::VectorXd& state, double dt,
                                  double dx, double dy, double dtheta) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    Eigen::VectorXd new_state = state;
    new_state(0) += dx;
    new_state(1) += dy;
    new_state(2) += dtheta;
    
    // Guard against dt == 0 or very small dt
    double safe_dt = std::max(dt, 1e-6);
    new_state(3) = dx / safe_dt;
    new_state(4) = dy / safe_dt;
    
    new_state(2) = normalizeAngle(new_state(2));

    return new_state;
}

// ============================================================================
// MEASUREMENT MODEL
// ============================================================================

/**
 * @brief Predict measurement given current state and landmark
 * 
 * STUDENT TODO:
 * 1. Calculate relative position: landmark - robot position
 * 2. Transform to robot frame using robot orientation
 * 3. Return relative position in robot frame
 */
Eigen::Vector2d UKF::measurementModel(const Eigen::VectorXd& state, int landmark_id) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    // Check if landmark exists
    if (landmarks_.find(landmark_id) == landmarks_.end()) {
        return Eigen::Vector2d::Zero();
    }
    
    // Find landmark position (ℓₓ, ℓᵧ) using landmark_id
    double lx = landmarks_[landmark_id].first;
    double ly = landmarks_[landmark_id].second;
    
    
    double rx = lx - state(0);
    double ry = ly - state(1);
    

    double theta = state(2);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    
    Eigen::Vector2d measurement;
    measurement(0) = rx * cos_theta + ry * sin_theta;
    measurement(1) = -rx * sin_theta + ry * cos_theta;
    
    return measurement;
}

// ============================================================================
// ANGLE NORMALIZATION
// ============================================================================

double UKF::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// ============================================================================
// PREDICTION STEP
// ============================================================================

/**
 * @brief Kalman Filter Prediction Step (Time Update)
 * 
 * STUDENT TODO:
 * 1. Generate sigma points from current state and covariance
 * 2. Propagate each sigma point through motion model
 * 3. Calculate mean and covariance of predicted sigma points
 * 4. Add process noise
 * 5. Update state and covariance estimates
 */
void UKF::predict(double dt, double dx, double dy, double dtheta) {
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints(x_, P_);
    
        std::vector<Eigen::VectorXd> new_sigma_points;
        for (const auto& sigma_point : sigma_points) {
            new_sigma_points.push_back(processModel(sigma_point, dt, dx, dy, dtheta));
        }
        
        Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(nx_);
        // Normal components (x, y, vx, vy) - weighted sum
        for (size_t i = 0; i < new_sigma_points.size(); ++i) {
            x_pred(0) += Wm_[i] * new_sigma_points[i](0);
            x_pred(1) += Wm_[i] * new_sigma_points[i](1);
            x_pred(3) += Wm_[i] * new_sigma_points[i](3);
            x_pred(4) += Wm_[i] * new_sigma_points[i](4);
        }
        // Circular mean for theta (angle averaging on unit circle)
        double sin_sum = 0.0, cos_sum = 0.0;
        for (size_t i = 0; i < new_sigma_points.size(); ++i) {
            sin_sum += Wm_[i] * std::sin(new_sigma_points[i](2));
            cos_sum += Wm_[i] * std::cos(new_sigma_points[i](2));
        }
        x_pred(2) = std::atan2(sin_sum, cos_sum);
        
        
        Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(nx_, nx_);
        for (size_t i = 0; i < new_sigma_points.size(); ++i) {
            Eigen::VectorXd diff = new_sigma_points[i] - x_pred;
            diff(2) = normalizeAngle(diff(2));
            P_pred += Wc_[i] * diff * diff.transpose();
        }
        
        P_pred += Q_;
        
        x_ = x_pred;
        P_ = P_pred;
}

// ============================================================================
// UPDATE STEP
// ============================================================================

/**
 * @brief Kalman Filter Update Step (Measurement Update)
 * 
 * STUDENT TODO:
 * 1. Generate sigma points
 * 2. Transform through measurement model
 * 3. Calculate predicted measurement mean
 * 4. Calculate measurement and cross-covariance
 * 5. Compute Kalman gain
 * 6. Update state with innovation
 * 7. Update covariance
 */
void UKF::update(const std::vector<std::tuple<int, double, double, double>>& landmark_observations) {
    if (landmark_observations.empty()) {
        return;
    }
    
    // STUDENT IMPLEMENTATION STARTS HERE
    // ========================================================================
    
    // Process each landmark observation
    for (const auto& observation : landmark_observations) {
        int landmark_id = std::get<0>(observation);
        double z_x = std::get<1>(observation);  // Observed measurement x
        double z_y = std::get<2>(observation);  // Observed measurement y
        // noise_cov is available but we use R_ instead
        
        // Skip if landmark doesn't exist
        if (landmarks_.find(landmark_id) == landmarks_.end()) {
            continue;
        }
        
        // 1. Generate sigma points and transform through measurement model
        std::vector<Eigen::VectorXd> sigma_points = generateSigmaPoints(x_, P_);
        std::vector<Eigen::Vector2d> predicted_measurements;
        for (const auto& sigma_point : sigma_points) {
            predicted_measurements.push_back(measurementModel(sigma_point, landmark_id));
        }
        
        // 2. Calculate ẑ, P_zz, and P_xz
        // Predicted measurement mean: ẑ = Σ(Wm_i * z_pred_i)
        Eigen::Vector2d z_pred = Eigen::Vector2d::Zero();
        for (size_t i = 0; i < predicted_measurements.size(); ++i) {
            z_pred += Wm_[i] * predicted_measurements[i];
        }
        
        // Measurement covariance: P_zz = Σ(Wc_i * (z_pred_i - ẑ) * (z_pred_i - ẑ)^T) + R
        Eigen::Matrix2d P_zz = Eigen::Matrix2d::Zero();
        for (size_t i = 0; i < predicted_measurements.size(); ++i) {
            Eigen::Vector2d diff = predicted_measurements[i] - z_pred;
            P_zz += Wc_[i] * diff * diff.transpose();
        }
        P_zz += R_;
        
        // Cross-covariance: P_xz = Σ(Wc_i * (chi_i - x_) * (z_pred_i - ẑ)^T)
        Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(nx_, nz_);
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            Eigen::VectorXd state_diff = sigma_points[i] - x_;
            // Normalize angle difference for theta component
            state_diff(2) = normalizeAngle(state_diff(2));
            Eigen::Vector2d meas_diff = predicted_measurements[i] - z_pred;
            P_xz += Wc_[i] * state_diff * meas_diff.transpose();
        }
        
        // 3. Compute Kalman gain: K = P_xz * P_zz^(-1)
        // Use robust LDLT solve instead of explicit inverse for numerical stability
        Eigen::MatrixXd K = P_xz * P_zz.ldlt().solve(Eigen::MatrixXd::Identity(nz_, nz_));
        
        // 4. Update state and covariance
        // Actual measurement
        Eigen::Vector2d z(z_x, z_y);
        
        // Innovation
        Eigen::Vector2d innovation = z - z_pred;
        
        // Log innovation for debugging
        std::cout << "UKF Update - Landmark " << landmark_id << ": "
                  << "z_pred=(" << z_pred(0) << ", " << z_pred(1) << "), "
                  << "z=(" << z(0) << ", " << z(1) << "), "
                  << "innovation=(" << innovation(0) << ", " << innovation(1) << "), "
                  << "K_max=" << K.cwiseAbs().maxCoeff() << std::endl;
        
        // Update state: x_ = x_ + K * (z - ẑ)
        Eigen::VectorXd state_before = x_;
        x_ += K * innovation;
        // Normalize angle
        x_(2) = normalizeAngle(x_(2));
        
        // Log state change
        Eigen::VectorXd state_change = x_ - state_before;
        std::cout << "UKF Update - State change: dx=" << state_change(0) 
                  << ", dy=" << state_change(1) 
                  << ", dtheta=" << state_change(2) << std::endl;
        
        // Update covariance: P_ = P_ - K * P_zz * K^T
        P_ -= K * P_zz * K.transpose();
    }
}

// ============================================================================
// LANDMARK MANAGEMENT
// ============================================================================

void UKF::setLandmarks(const std::map<int, std::pair<double, double>>& landmarks) {
    landmarks_ = landmarks;
}

bool UKF::hasLandmark(int id) const {
    return landmarks_.find(id) != landmarks_.end();
}
