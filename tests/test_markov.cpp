#define CATCH_CONFIG_MAIN

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "sp_math.h"
#include "markov.h"

TEST_CASE("compute_probs")
{
    Eigen::MatrixXf trans_mat(3, 3);
    trans_mat << 0.3, 0.5, 0.2,
                 0.5, 0.1, 0.4,
                 0.2, 0.6, 0.2;

    Eigen::MatrixXf init_dist(1, 3);
    init_dist << 1, 0, 0;

    sp::Markov markov(trans_mat);

    Eigen::MatrixXf probs = markov.compute_probs(init_dist, 10000);
    Eigen::MatrixXf expected = markov.stationary_dist();

    REQUIRE(probs.isApprox(expected, 0.001));
}

TEST_CASE("gen_seq")
{
    Eigen::MatrixXf trans_mat(3, 3);
    trans_mat << 0.3, 0.5, 0.2,
                 0.5, 0.1, 0.4,
                 0.4, 0.5, 0.1;

    sp::Markov chain(trans_mat);

    std::vector<int> states = chain.gen_sequence(1, 1000);
    REQUIRE((
        std::find(states.begin(), states.end(), 0) != states.end() &&
        std::find(states.begin(), states.end(), 1) != states.end() &&
        std::find(states.begin(), states.end(), 2) != states.end()));
}

TEST_CASE("stationary_prob")
{
    Eigen::MatrixXf trans_mat(3, 3);
    trans_mat << 0.3, 0.5, 0.2,
                 0.5, 0.1, 0.4,
                 0.2, 0.6, 0.2;

    sp::Markov chain(trans_mat);

    Eigen::MatrixXf eig(1, 3);
    eig << 1.26315, 1.36842, 1;
    eig = eig / eig.sum();

    Eigen::MatrixXf res = chain.stationary_dist();
    REQUIRE(eig.isApprox(res, 0.01));
}

TEST_CASE("set_trans_probs")
{
    sp::Markov chain;

    chain.set_transition_probs({0, 1, 3, 2, 0, 1, 0, 2});

    Eigen::MatrixXf expected(4, 4);
    expected << 0, 0.67, 0.33, 0,
                0.5, 0, 0, 0.5,
                1, 0, 0, 0,
                0, 0, 1, 0;

    Eigen::MatrixXf init_dist(1, 4);
    init_dist << 0.25, 0.25, 0.25, 0.25;

    REQUIRE(chain.compute_probs(init_dist, 1).isApprox(init_dist * expected,
        0.01));
}
