#pragma once

#include <vector>
#include <random>

namespace sp
{

class GBM
{
private:
    float drift_;
    float volatility_;
    float dt_;
    float s_curr_;
    std::vector<float> s_vals_;
    std::default_random_engine rand_eng_;
    std::normal_distribution<float> norm_dist_;
public:
    GBM(float drift, float volatility, float dt, float s_init);
    ~GBM();

    void step();
    void n_step(int n);
    std::vector<float> get_gmb_path();
};

} // namespace sp

