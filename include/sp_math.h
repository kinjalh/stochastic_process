#pragma once
#include <vector>

#include <Eigen/Dense>

namespace sp 
{
bool approx_eq(float a, float b);
bool approx_eq(float a, float b, float eps);
int find_nearest_tval(const std::vector<std::pair<float, float>> &x_t, float t);
float compute_derivative(const std::vector<std::pair<float, float>> &x_t,
    float t, int n_pts);

float compute_drift(const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end);

float compute_volatility(const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end);

std::vector<std::pair<float, int>> drift_dist(
    const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end, float bin_size);

Eigen::MatrixXf drift_transitions(
    const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end, float bin_size);
} // namespace sp
