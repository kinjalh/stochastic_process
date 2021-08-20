#include <iostream>
#include <stdlib.h>
#include <time.h>

#include <unsupported/Eigen/MatrixFunctions>

#include "markov.h"
#include "sp_math.h"

namespace sp
{

Markov::Markov()
{
}

Markov::Markov(const Eigen::MatrixXf &trans_mat) : trans_mat_(trans_mat)
{
}

Markov::Markov(const Markov &m) : trans_mat_(Eigen::MatrixXf(m.trans_mat_))
{
}

Markov::~Markov()
{
}

std::vector<int> Markov::gen_sequence(int init_st, int n_steps)
{
    std::vector<int> states(n_steps + 1);
    states[0] = init_st;
    srand(time(NULL));

    int curr_st = init_st;
    for (int i = 1; i < n_steps; i++)
    {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        int next_st = 0;
        float sum = 0;
        while (sum < r)
        {
            sum += trans_mat_(curr_st, next_st);
            if (sum < r)
            {
                next_st++;
            }
        }
        states[i] = next_st;
        curr_st = next_st;
    }

    return states;
}

Eigen::MatrixXf Markov::compute_probs(const Eigen::MatrixXf &init_dist,
    int n_steps)
{
    return init_dist * trans_mat_.pow(n_steps);
}

Eigen::MatrixXf Markov::stationary_dist()
{
    Eigen::EigenSolver<Eigen::MatrixXf> eig_solve(trans_mat_.transpose());
    Eigen::VectorXcf eig_vals = eig_solve.eigenvalues();
    Eigen::MatrixXcf eig_vects = eig_solve.eigenvectors();

    for (int i = 0; i < eig_vals.rows(); i++)
    {
        if (approx_eq(1, eig_vals.coeff(i, 0).real(), epsilon))
        {
            Eigen::MatrixXf res = eig_vects.col(i).real().transpose();
            return res / res.sum();
        }
    }

    return Eigen::MatrixXf::Zero(trans_mat_.rows(), 1);
}

void Markov::set_transition_probs(const std::vector<int> &event_seq)
{
    int n_events = *std::max_element(event_seq.begin(), event_seq.end()) + 1;
    Eigen::MatrixXi trans_counts = Eigen::MatrixXi::Zero(n_events, n_events);
    for (int i = 0; i < event_seq.size() - 1; i++)
    {
        int st_init = event_seq[i];
        int st_fin = event_seq[i + 1];
        trans_counts(st_init, st_fin)++;
    }

    trans_mat_ = Eigen::MatrixXf::Zero(n_events, n_events);
    for (int i = 0; i < n_events; i++)
    {
        trans_mat_.row(i) = trans_counts.row(i).cast<float>() /
            trans_counts.row(i).sum();
    }
}

} // namespace sp
