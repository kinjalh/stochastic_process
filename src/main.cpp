#include <iostream>
#include <fstream>

#include "sp_math.h"
#include "hidden_markov.h"

int main(int argc, char const *argv[])
{
    Eigen::MatrixXf trans_mat(2, 2);
    trans_mat << 0.7, 0.3,
                 0.4, 0.6;
    
    Eigen::MatrixXf emiss_mat(2, 3);
    emiss_mat << 0.5, 0.4, 0.1,
                 0.1, 0.3, 0.6;
    
    sp::HMarkov hmm(trans_mat, emiss_mat);

    Eigen::MatrixXf init_dist(1, 2);
    init_dist << 0.6, 0.4;

    std::vector<int> obs_seq = {0, 1, 2};

    // hmm.seq_prob(obs_seq, init_dist);

    std::vector<int> seq = hmm.best_st_seq(obs_seq, init_dist);
    for (int s : seq)
    {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    return 0;
}

