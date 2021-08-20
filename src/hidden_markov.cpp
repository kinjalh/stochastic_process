#include <iostream>

#include "hidden_markov.h"

namespace sp
{

HMarkov::HMarkov()
{
}

HMarkov::HMarkov(const Eigen::MatrixXf &trans_mat,
    const Eigen::MatrixXf &emiss_mat) :
    trans_mat_(trans_mat), emiss_mat_(emiss_mat)
{
}

HMarkov::HMarkov(const HMarkov &m) : 
    trans_mat_(m.trans_mat_), emiss_mat_(m.emiss_mat_)
{
}

HMarkov::~HMarkov()
{
}

float HMarkov::seq_prob(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    return forward(obs_seq, init_dist);
}

float HMarkov::forward(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    if (obs_seq.empty())
    {
        return -1.0;
    }

    Eigen::MatrixXf alpha =
        emiss_mat_.col(obs_seq[0]).transpose().cwiseProduct(init_dist);
    std::cout << alpha << std::endl;

    for (int t = 1; t < obs_seq.size(); t++)
    {
        Eigen::MatrixXf alpha_next(alpha.rows(), alpha.cols());
        for (int j = 0; j < alpha_next.cols(); j++)
        {
            Eigen::MatrixXf trans_probs = trans_mat_.col(j);
            float emiss_p = emiss_mat_(j, obs_seq[t]);
            float alpha_t_j = (alpha * trans_probs)(0, 0) * emiss_p;
            alpha_next(0, j) = alpha_t_j;
        }
        alpha = alpha_next;
        std::cout << "-----------------------" << std::endl;
        std::cout << alpha << std::endl;
    }
    std::cout << "----------------------" << std::endl;
    std::cout << "final result = " << alpha.sum() << std::endl;
    return alpha.sum();
}

std::vector<int> HMarkov::best_st_seq(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    viterbi(obs_seq, init_dist);
    return std::vector<int>({});
}

void HMarkov::viterbi(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    if (obs_seq.empty())
    {
        return;
    }

    Eigen::MatrixXf delta =
        emiss_mat_.col(obs_seq[0]).transpose().cwiseProduct(init_dist);
    Eigen::MatrixXf psi =
        Eigen::MatrixXf::Zero(trans_mat_.rows(), obs_seq.size() + 1);
    psi.col(0).setConstant(-1);

    for (int t = 1; t < obs_seq.size(); t++)
    {
        Eigen::MatrixXf delta_next(delta.rows(), delta.cols());
        for (int j = 0; j < delta_next.cols(); j++)
        {
            Eigen::MatrixXf trans_probs = trans_mat_.col(j).transpose();
            Eigen::MatrixXf arrival_probs = delta.cwiseProduct(trans_probs);
            float emiss_p = emiss_mat_(j, obs_seq[t]);   
            float delta_best = 0;
            int predecessor_best = 0;
            for (int i = 0; i < arrival_probs.cols(); i++)
            {
                if (arrival_probs(0, i) > delta_best)
                {
                    delta_best = arrival_probs(0, i);
                    predecessor_best = i;
                }
            }
            delta_next(0, j) = delta_best;
            psi(j, t) = predecessor_best;
        }
    }
    std::cout << delta << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << psi << std::endl;
}

} // namespace sp

