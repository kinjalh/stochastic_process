#include <iostream>
#include <fstream>

#include "sp_math.h"
#include "hidden_markov.h"

int main(int argc, char const *argv[])
{
    Eigen::MatrixXf trans_mat(2, 2);
    trans_mat << 0.5, 0.5,
                 0.3, 0.7;
    
    Eigen::MatrixXf emiss_mat(2, 2);
    emiss_mat << 0.8, 0.2,
                 0.4, 0.6;
    
    sp::HMarkov hmm(trans_mat, emiss_mat);

    Eigen::MatrixXf init_dist(1, 2);
    init_dist << 0.375, 0.625;

    std::vector<int> obs_seq = {0, 0, 1};

    hmm.forward_algo(obs_seq, init_dist);

    return 0;
}

