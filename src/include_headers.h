#ifndef __INCLUDE_STD_H__
#define __INCLUDE_STD_H__

//Writing
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
//Containers
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
//Generic
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <memory>
#include <exception>
#include <numeric>
#include <iterator> //for std::inserter
#include <utility>  //for std::forward
#include <tuple>
#include <type_traits>
#include <functional>
//Eigen
#include <Eigen/Dense>
#include <Eigen/Cholesky>
//#include <Eigen/Sparse>
//Parallel
#include <omp.h>
//Rcpp -> decomment for using pure c++ code. It is used in progress bar, samplers and Groups
#include <Rcpp.h>
#endif
