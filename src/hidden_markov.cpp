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
    Eigen::MatrixXf alpha = forward(obs_seq, init_dist);
    return alpha.row(alpha.rows() - 1).sum();
}

Eigen::MatrixXf HMarkov::forward(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    if (obs_seq.empty())
    {
        return Eigen::MatrixXf::Zero(0, 0);
    }

    Eigen::MatrixXf alpha =
        Eigen::MatrixXf::Zero(obs_seq.size(), trans_mat_.rows());
    alpha.row(0) =
        emiss_mat_.col(obs_seq[0]).transpose().cwiseProduct(init_dist);

    for (int t = 1; t < obs_seq.size(); t++)
    {
        Eigen::MatrixXf alpha_next(1, alpha.cols());
        for (int j = 0; j < alpha_next.cols(); j++)
        {
            Eigen::MatrixXf trans_probs = trans_mat_.col(j);
            float emiss_p = emiss_mat_(j, obs_seq[t]);
            float alpha_t_j = (alpha.row(t - 1) * trans_probs)(0, 0) * emiss_p;
            alpha_next(0, j) = alpha_t_j;
        }
        alpha.row(t) = alpha_next;
    }
    return alpha;
}

Eigen::MatrixXf HMarkov::st_probs(const std::vector<int> &obs_seq)
{
    Eigen::MatrixXf beta = backward(obs_seq);
    return beta.row(0);
}

Eigen::MatrixXf HMarkov::backward(const std::vector<int> &obs_seq)
{
    if (obs_seq.empty())
    {
        return Eigen::MatrixXf::Zero(1, trans_mat_.cols());
    }

    Eigen::MatrixXf beta =
        Eigen::MatrixXf::Ones(obs_seq.size(), trans_mat_.cols());

    for (int t = obs_seq.size() - 2; t >= 0; t--)
    {
        Eigen::MatrixXf beta_next(1, beta.cols());
        for (int i = 0; i < beta_next.cols(); i++)
        {
            Eigen::MatrixXf trans_probs = trans_mat_.row(i);
            Eigen::MatrixXf emiss_probs =
                emiss_mat_.col(obs_seq[t + 1]).transpose();
            Eigen::MatrixXf tmp = trans_probs.cwiseProduct(
                emiss_probs).cwiseProduct(beta.row(t + 1));
            beta_next(0, i) = tmp.sum();
        }
        beta.row(t) = beta_next;
    }

    return beta;
}

std::vector<int> HMarkov::best_st_seq(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    return viterbi(obs_seq, init_dist);
}

std::vector<int> HMarkov::viterbi(const std::vector<int> &obs_seq,
    const Eigen::MatrixXf &init_dist)
{
    if (obs_seq.empty())
    {
        return {-1};
    }

    Eigen::MatrixXf delta =
        emiss_mat_.col(obs_seq[0]).transpose().cwiseProduct(init_dist);
    Eigen::MatrixXi psi =
        Eigen::MatrixXi::Zero(trans_mat_.rows(), obs_seq.size() - 1);

    for (int t = 1; t < obs_seq.size(); t++)
    {
        Eigen::MatrixXf delta_next(delta.rows(), delta.cols());
        for (int j = 0; j < delta_next.cols(); j++)
        {
            Eigen::MatrixXf trans_probs = trans_mat_.col(j).transpose();
            Eigen::MatrixXf arrival_probs = delta.cwiseProduct(trans_probs);
            float emiss_p = emiss_mat_(j, obs_seq[t]);   
            float delta_best = -1;
            int predecessor_best = -1;
            for (int i = 0; i < arrival_probs.cols(); i++)
            {
                if (arrival_probs(0, i) > delta_best)
                {
                    delta_best = arrival_probs(0, i);
                    predecessor_best = i;
                }
            }
            delta_next(0, j) = delta_best * emiss_p;
            psi(j, t - 1) = predecessor_best;
        }
        delta = delta_next;
    }

    float delta_best = 0;
    int predecessor_best = 0;
    for (int i = 0; i < delta.cols(); i++) {
        if (delta(0, i) > delta_best) {
            delta_best = delta(0, i);
            predecessor_best = i;
        }
    }

    std::vector<int> st_path;
    st_path.push_back(predecessor_best);
    for (int psi_col = psi.cols() - 1; psi_col >= 0; psi_col--)
    {
        st_path.push_back(psi(predecessor_best, psi_col));
    }

    std::reverse(st_path.begin(), st_path.end());
    return st_path;
}


void HMarkov::fit_hmm(const std::vector<int> &obs_seq)
{
}

void HMarkov::baum_welch(const std::vector<int> &obs_seq)
{

}

} // namespace sp

