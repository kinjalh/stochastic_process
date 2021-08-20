#pragma once

#include <vector>

#include <Eigen/Dense>

namespace sp
{
constexpr float epsilon = 1e-5;

class Markov
{
private:
    Eigen::MatrixXf trans_mat_;
public:
    Markov();
    Markov(const Eigen::MatrixXf &trans_mat);
    Markov(const Markov &m);
    virtual ~Markov();

    virtual std::vector<int> gen_sequence(int init_st, int n_steps);    
    virtual Eigen::MatrixXf compute_probs(const Eigen::MatrixXf &init_dist,
        int n_steps);
    virtual Eigen::MatrixXf stationary_dist();
    virtual void set_transition_probs(const std::vector<int> &event_seq);
};
} // namespace sp
