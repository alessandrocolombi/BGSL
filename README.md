# BGSL - Block Graph Structural Learning
**Author:** Alessandro Colombi

## Table of Contents
1. [Overview](#Overview)
2. [Prerequisites](#Prerequisites)
3. [Installation - Unix](#Installation---Unix)
4. [Installation - Window](#Installation---Window)

## Overview

The `R` package **BGSL** provides statistical tools for Bayesian structural learning of undirected Gaussian graphical
models (GGM).
The main contribution of this project is to allow the user to sample from models that enforce the graph to have a block
structure. Classical, non block models, are provided as well. Beside, **BGSL** does not limit itself to sample from GGM
but also apply them in the more complex scenario of functional data. Indeed it provides a sampler for both smoothing
functional data and estimating the underlying structure of thier regression coefficients.<br/>
In order to get efficiency the whole core of **BGSL**'s code is written in `C++` and makes use of **OpenMP** to
parallelize the most heavy computational tasks. `R` is used only to provide an user friendly and easy to use interface. 

## Prerequisites

This package makes use of the following libraries:
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page): it is an highly optimized C++ template library for
linear algebra. All matrices, vectors and solvers in **BGSL** makes use of Eigen. 
* [GSL](https://www.gnu.org/software/gsl/): the GNU Scientific Library (GSL) provides a wide range of fast mathematical 
routines. **BGSL** utilizies GSL for generating random numbers, drawing samples from common distributions and for
the computation of smoothing basis splines (Bsplines).

The user should not be worried of dealing with them because they work only under the hood. The only task of the user is
to be sure to have them installed. Follow instructions in [Installation - Unix](#Installation---Unix) and   
[Installation - Window](#Installation---Window) to make them available on your operating system.
#### Note for future developperers
* The first version of this package when developped using Eigen 3.3.7. As soon as Eigen 3.4 will be released and
integrated in RcppEigen new useful features will be available. For example slicing and indexing will become standard
operations, making life much easier when extracting submatrices. 
See [Slicing and Indexing](https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html).
RcppParallel 
* RcppParallel is used in orded to achive parallelization. The first version on **BGSL** has been build using
RcppParallel 5.0.2. It provides a complete toolkit for creating portable, high-performance parallel algorithms without  
requiring direct manipulation of operating system threads. In order to achive this task, it includes 
[Intel(R) TBB](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onetbb.html). Unfortunately
right now it includes only version 2016 4.3 which is too old to achive 
[parallel execution](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t) of STL algorithms. 
Intel(R) TBB 2018 would be required.

## Installation - Unix

Questa importante

## Installation - Window

Questa serve un mezzo miracolo