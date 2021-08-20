#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "gbm.h"

TEST_CASE("get_vals")
{
    sp::GBM gbm(0.5, 0.3, 1.0 / 365, 300);

    gbm.n_step(100);
    std::vector<float> vals = gbm.get_gmb_path();
    for (float v : vals)
    {
        std::cout << v << std::endl;
    }

    REQUIRE(1 == 1);
}
