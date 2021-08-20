#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "sp_math.h"

namespace sp
{
bool approx_eq(float a, float b)
{
    return approx_eq(a, b, std::numeric_limits<float>::epsilon());
}

bool approx_eq(float a, float b, float eps)
{
    return std::abs(a - b) <= eps;
}

// x_t contains pairs of (t, x) values where x = f(t) -> must be ordered in
// ascending t values
// t is the value that we want to differentiate at -> find dx/dt
// n_pts is the number of points on either side of t to sample
float compute_derivative(
    const std::vector<std::pair<float, float>> &x_t, float t, int n_pts)
{
    // first find points as close as possible to t
    int idx_nearest = find_nearest_tval(x_t, t);

    int idx_start = 0;
    int idx_end = 0;
    if (n_pts > 0)
    {
        idx_start = std::max(0, idx_nearest - n_pts);
        idx_end = std::min((int) x_t.size() - 1, idx_nearest + n_pts);
    }
    else if (x_t[idx_nearest].first > t)
    {
        idx_start = std::max(0, idx_nearest - 1);
        idx_end = idx_nearest;
    }
    else
    {
        idx_start = idx_nearest;
        idx_end = std::min((int) x_t.size() - 1, idx_nearest + 1);
    }
    
    float dt_total = x_t[idx_end].first - x_t[idx_start].first;
    float avg = 0.0;
    for (int i = idx_start; i < idx_end; i++)
    {
        float dx = x_t[i + 1].second - x_t[i].second;
        avg += dx / dt_total; 
    }

    return avg;
}

// finds the index of the x_t entry with closest t value to t
// runs in logarithmic time
// if x_t is empty, returns -1
int find_nearest_tval(const std::vector<std::pair<float, float>> &x_t,
    float t)
{
    if (x_t.empty())
    {
        return -1;
    }

    int idx_left = 0;
    int idx_right = x_t.size() - 1;
    float min_err_abs = std::numeric_limits<float>::max();
    int idx_res = (idx_left + idx_right) / 2;
    while (idx_left <= idx_right)
    {
        int idx_mid = (idx_left + idx_right) / 2;

        float err = x_t[idx_mid].first - t;
        if (std::abs(err) < min_err_abs)
        {
            min_err_abs = std::abs(err);
            idx_res = idx_mid;
        }

        if (err > 0)
        {
            // t is less than our current value
            idx_right = idx_mid - 1;
        }
        else
        {
            // t is greater than our current value
            idx_left = idx_mid + 1;
        }
    }

    return idx_res;
}

float compute_drift(const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end)
{
    if (idx_start == idx_end)
    {
        return 0;
    }
    float drift = (log(data[idx_end].second) - log(data[idx_start].second)) /
        (data[idx_end].first - data[idx_start].first);
    return drift;
}

float compute_volatility(const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end)
{
    float mean = 0;
    for (int i = idx_start; i < idx_end; i++)
    {
        float d_s = data[i + 1].second - data[i].second;
        float d_t = data[i + 1].first - data[i].first;
        mean += std::pow(d_s, 2) / (std::pow(data[i].second, 2) * d_t);
    }
    mean /= data.size() - 1;
    return std::sqrt(mean);
}

std::vector<std::pair<float, int>> drift_dist(
    const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end, float bin_size)
{
    std::vector<float> drift_vals(idx_end - idx_start);
    for (int i = idx_start; i < idx_end; i++)
    {
        float drift = compute_drift(data, i, i + 1);
        drift_vals[i - idx_start] = drift;
    }

    float drift_min = *(std::min_element(drift_vals.begin(), drift_vals.end()));
    float drift_max = *(std::max_element(drift_vals.begin(), drift_vals.end()));

    float range_min;
    for (range_min = 0; range_min > drift_min; range_min -= bin_size);
    float range_max;
    for (range_max = 0; range_max < drift_max; range_max += bin_size);

    std::vector<std::pair<float, int>> hist;
    for (float b_min = range_min; b_min < range_max; b_min += bin_size)
    {
        hist.push_back({b_min, 0});
    }

    for (float d : drift_vals)
    {
        int idx = (int)((d - range_min) / bin_size);
        hist[idx].second++;
    }

    return hist;
}

Eigen::MatrixXf drift_transitions(
    const std::vector<std::pair<float, float>> &data,
    int idx_start, int idx_end, float bin_size)
{
    std::vector<float> drift_vals(idx_end - idx_start);
    for (int i = idx_start; i < idx_end; i++)
    {
        float drift = compute_drift(data, i, i + 1);
        drift_vals[i - idx_start] = drift;
    }

    float drift_min = *(std::min_element(drift_vals.begin(), drift_vals.end()));
    float drift_max = *(std::max_element(drift_vals.begin(), drift_vals.end()));

    std::cout << "drift min = " << drift_min << ", drift max = " << drift_max <<
        std::endl;

    int n_bins = 0;
    float range_min;
    for (range_min = 0; range_min > drift_min; range_min -= bin_size)
    {
        n_bins++;
    }
    float range_max;
    for (range_max = 0; range_max < drift_max; range_max += bin_size)
    {
        n_bins++;
    }
    std::cout << "n_bins = " << n_bins << std::endl;
    for (float d : drift_vals)
    {
        std::cout << d << " ";
    }
    std::cout << std::endl;
    std::cout << "BINS----------------------" << std::endl;
    int bin_counter = 0;
    for (float bin = range_min; bin < range_max; bin += bin_size)
    {
        std::cout << "bin " << bin_counter << ", from " << bin << " to " << (bin + bin_size) << std::endl;
        bin_counter++;
    }
    std::cout << std::endl;

    std::cout << "drift vals len = " << drift_vals.size() << std::endl;
    std::cout << "MAT INDS-----------------" << std::endl;
    Eigen::MatrixXf transitions = Eigen::MatrixXf::Zero(n_bins, n_bins);
    for (int i = 0; i < drift_vals.size() - 1; i++) {
        std::cout << "i = " << i << std::endl;
        float drift_init = drift_vals[i];
        int st_init = (int)((drift_init - range_min) / bin_size);
        float drift_final = drift_vals[i + 1];
        int st_final = (int)((drift_final - range_min) / bin_size);
        std::cout << "drifts: " << drift_init << ", " << drift_final << "\tind:"
            << st_init << ", " << st_final << std::endl;
        transitions(st_init, st_final)++;
    }
    std::cout << transitions << std::endl;
    return transitions;
}

} // namespace sp
