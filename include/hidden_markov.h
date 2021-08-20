#pragma once

#include <vector>

#include <Eigen/Dense>

namespace sp
{

class HMarkov
{
private:
    Eigen::MatrixXf trans_mat_;
    Eigen::MatrixXf emiss_mat_;
public:
    HMarkov();
    HMarkov(const Eigen::MatrixXf &trans_mat, const Eigen::MatrixXf &emiss_mat);
    HMarkov(const HMarkov &other);
    virtual ~HMarkov();

    virtual float forward_algo(const std::vector<int> obs_seq,
        const Eigen::MatrixXf &init_dist);
};

} // namespace sp
