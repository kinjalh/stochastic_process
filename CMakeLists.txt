cmake_minimum_required(VERSION 3.10)
project(stock_pred)
set(CMAKE_CXX_STANDARD 14)

include_directories(include)
add_library(sp src/sp_math.cpp src/markov.cpp src/gbm.cpp src/hidden_markov.cpp)

find_package(Catch2 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test_markov tests/test_markov.cpp)
target_link_libraries(test_markov sp Catch2::Catch2WithMain Eigen3::Eigen)

add_executable(test_gbm tests/test_gbm.cpp)
target_link_libraries(test_gbm sp Catch2::Catch2WithMain Eigen3::Eigen)

add_executable(main src/main.cpp)
target_link_libraries(main sp)
