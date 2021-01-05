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

The user should not be worried of dealing with them because they only work under the hood. The only task of the user is
to be sure to have them installed. Follow instructions in [Installation - Unix](#Installation---Unix) and   
[Installation - Window](#Installation---Window) to make them available on your operating system.
#### Note for future developperers
* The first version of this package was developped using Eigen 3.3.7. As soon as Eigen 3.4 will be released and
integrated in RcppEigen new useful features will be available. For example slicing and indexing will become standard
operations, making life much easier when extracting submatrices. 
See [Slicing and Indexing](https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html). 
* `RcppParallel` is used in orded to achive parallelization. The first version on **BGSL** has been build using
`RcppParallel` 5.0.2. It provides a complete toolkit for creating portable, high-performance parallel algorithms without requiring direct manipulation of operating system threads. In order to achive this task, it includes 
[Intel(R) TBB](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onetbb.html). Unfortunately
right now it includes only version 2016 4.3 which is too old to achive 
[parallel execution](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t) of STL algorithms. 
Intel(R) TBB 2018 would be required.

## Installation - Unix
Installation on Unix systems is straightforward. Note that all the following commands are valid for Ubuntu/Debian users only. 
On other platforms, should be easy to use the corresponding package managing tool commands them. 
### R - Rstudio - devtools
First step is to have `R` installed on your computer. If you already have that, make sure that it is at least version 4.0.3 and that `devtool` package is available. In this case go directly to 
the next point.
Otherwise, follow instruction in [r-project](https://cloud.r-project.org/) to download the right version of `R` according to your distribution. Ubuntu users may get lost or get into trouble because of an issue with the key signing Ubuntu archives on CRAN.
This [guide](https://cran.r-project.org/bin/linux/ubuntu/) is very well done and should clarify all doubts. 
The longest and safest way to proceed is summarized below. For sake of brevity, only the case of Ubuntu Focal Fossa in given.
1. Search [here](http://keyserver.ubuntu.com:11371/) for key ID 0x51716619e084dab9. A page with keys by Michael Rutter marutter@gmail.com should display. Click on the first 51716619e084dab9 you see.
2. A PGP PUBLIC KEY opens, copy it to a plain text file, called key.txt.
3. Open a terminal and type 
```shell
$ sudo apt-key add key.txt
```
4. Go to etc/apt folder and open source.list file with your prefered text editor, let's say sublime
```shell
$ cd /
$ cd etc/apt
$ sudo subl sources.list
```
5. Add the two following lines to the bottom of the page
```PlainText
deb http://cz.archive.ubuntu.com/ubuntu eoan main universe
deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/
```
The anxious user may check that this operation was succesful by opening "Sowtare and updates" > "Other software". The second string should be displayed.
6. Install `R` by typing
```shell
$ sudo apt-get update
$ sudo apt-get install r-base-dev
$ R
```
The last instruction should launch the program and display the downloaded version that should be at least 4.0.3.
7. Install [Rstudio](https://rstudio.com/products/rstudio/download/#download) as IDE for `R`. It is not the only possibility but it is a standard choice and highly recommended. 
The very last step is to install the `devtools` package. This installation may take few minutes since has many dependencies that rely on external libraries. It is indeed suggested to install them at this stage simply typing 
```shell
$ sudo apt-get install libssl-dev libcurl4-openssl-dev libxml2-dev libgit2-dev libnode-dev
```
Once they have all been installed, you are ready to complete this first step by typing
```R
install.packages("devtools")
```
from the R console.

### GSL

The GNU Scientific Library is linked to the package as external library. It cannot be installed directly from `R`. Fortunately it is available through the package manager, just use the command
```shell
$ sudo apt-get install libgsl-dev
```
In order to test if everything went smoothly, try 
```shell
$ gsl-config --cflags
-I/usr/include #where my the header files are
$ gsl-config --libs
-L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm #where the linked libraries are
```

### R depencencies

**BGSL**'s core code is all written in `C++` language and exploits the `Rcpp` package to interfece with `R` which allow to map many `R` data types and objects back and forth to their `C++` equivalents.
`Rcpp` by itself does not map matrices and vectors in Eigen types, for that the `RcppEigen` package has to be used. It provides all the necessary header files of the 
[Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page). Finally, the last requirement is to install the [`RcppParallel`](http://rcppcore.github.io/RcppParallel/) package. It is used to
achive cheap and portable parallelization via [Intel(R) TBB](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onetbb.html). 
Other useful packages are `fields` and `plot.matrix`, used to plot matrices.
All those packages can be installed directly from the `R` console via
```R
install.packages(c("Rcpp", "RcppEigen", "RcppParallel", "fields", "plot.matrix"))
```

## Installation - Window

Questa serve un mezzo miracolo