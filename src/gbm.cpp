#include <time.h>
#include <iostream>

#include "gbm.h"

namespace sp
{

GBM::GBM(float drift, float volatility, float dt, float s_init) :
    drift_(drift), volatility_(volatility), dt_(dt), s_curr_(s_init),
    rand_eng_(), norm_dist_(std::normal_distribution<float>(0, std::sqrt(dt)))
{
    rand_eng_.seed(time(NULL));
    s_vals_.push_back(s_init);
}

GBM::~GBM()
{   
}

void GBM::step()
{
    // std::default_random_engine rand_gen;
    // std::normal_distribution<float> n_dist(0, 0.05);

    float d_w = norm_dist_(rand_eng_);
    // std::cout << d_w << std::endl;
    float d_s = (drift_ * dt_ * s_curr_) + (volatility_ * s_curr_ * d_w);
    s_curr_ += d_s;
    s_vals_.push_back(s_curr_);
}

void GBM::n_step(int n)
{
    for (int i = 0; i < n; i++)
    {
        step();
    }
}

std::vector<float> GBM::get_gmb_path()
{
    return s_vals_;
}

} // namespace sp

