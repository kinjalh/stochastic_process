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

float HMarkov::forward_algo(const std::vector<int> obs_seq,
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
        Eigen::MatrixXf next(alpha.rows(), alpha.cols());
        for (int j = 0; j < next.cols(); j++)
        {
            Eigen::MatrixXf trans_probs = trans_mat_.col(j);
            float emiss_p = emiss_mat_(j, obs_seq[t]);
            float alpha_t_j = (alpha * trans_probs)(0, 0) * emiss_p;
            next(0, j) = alpha_t_j;
        }
        alpha = next;
        std::cout << "-----------------------" << std::endl;
        std::cout << alpha << std::endl;
    }
    std::cout << "----------------------" << std::endl;
    std::cout << "final result = " << alpha.sum() << std::endl;
    return alpha.sum();
}

} // namespace sp

