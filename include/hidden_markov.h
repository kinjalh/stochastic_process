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

    virtual float forward(const std::vector<int> &obs_seq,
        const Eigen::MatrixXf &init_dist);
    virtual std::vector<int> viterbi(const std::vector<int> &obs_seq,
        const Eigen::MatrixXf &init_dist);
public:
    HMarkov();
    HMarkov(const Eigen::MatrixXf &trans_mat, const Eigen::MatrixXf &emiss_mat);
    HMarkov(const HMarkov &other);
    virtual ~HMarkov();

    virtual float seq_prob(const std::vector<int> &obs_seq,
        const Eigen::MatrixXf &init_dist);
    virtual std::vector<int> best_st_seq(const std::vector<int> &obs_seq,
        const Eigen::MatrixXf &init_dist);
};

} // namespace sp
