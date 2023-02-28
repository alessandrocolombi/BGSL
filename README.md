# BGSL - Block Graph Structural Learning
**Author:** Alessandro Colombi

## Table of Contents
1. [Overview](#Overview)
2. [Prerequisites](#Prerequisites)
3. [Installation - Unix](#Installation---Unix)
4. [Installation - Windows](#Installation---Windows)

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
routines. **BGSL** utilizies **GSL** for generating random numbers, drawing samples from common distributions and for
the computation of smoothing basis splines (Bsplines).
* [HDF5](https://www.hdfgroup.org/): is a C++ library for writing binary files that are architecture independent. 
**BGSL** uses it to write all sampled values directly on file not to saturate the available memory, which prevents `R` session to abort.

The user should not be worried of dealing with them because they only work under the hood. Its only task is
to be sure to have them installed. Follow instructions in [Installation - Unix](#Installation---Unix) and   
[Installation - Windows](#Installation---Windows) to make them available on your operating system.


## Installation - Unix
Installation on Unix systems is straightforward. Note that all following commands are valid for Ubuntu/Debian users only. 
On other platforms, it should be easy to use the corresponding package managing tool commands. 
### R - Rstudio - devtools
First step is to have `R` installed on your computer. If you already have that, make sure that it is at least version 4.0.2 and that `devtool` package is available. In this case go directly to 
the [next point](#GSL).
Otherwise, follow instruction in [r-project](https://cloud.r-project.org/) to download the right version of `R` according to your distribution. Ubuntu users may get lost or get into trouble because of an issue with the key signing Ubuntu archives on CRAN.
This [guide](https://cran.r-project.org/bin/linux/ubuntu/) is very well done and should clarify all doubts. 
The longest and safest way to proceed is summarized below. For sake of brevity, only the case of Ubuntu Focal Fossa in given.
1. Search [here](http://keyserver.ubuntu.com:11371/) for key ID 0x51716619e084dab9. A page with keys by Michael Rutter marutter@gmail.com should display. Click on the first 51716619e084dab9 you see.
2. A PGP PUBLIC KEY opens, copy it to a plain text file, called key.txt.
3. Open a terminal and type 
```shell
$ sudo apt-key add key.txt
```
4. Go to etc/apt folder and open source.list file with your prefered text editor, let us say sublime
```shell
$ cd /etc/apt
$ sudo subl sources.list
```
5. Add the two following lines to the bottom of the page
```PlainText
deb http://cz.archive.ubuntu.com/ubuntu eoan main universe
deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/
```
The anxious user may check that this operation was succesful by opening "Software and updates" > "Other software". The second string should be displayed. <br/>
6. Install `R` by typing
```shell
$ sudo apt-get update
$ sudo apt-get install r-base-dev
$ R
```
The last instruction should launch the program and display the downloaded version that should be at least 4.0.2.<br/>
7. Install [Rstudio](https://rstudio.com/products/rstudio/download/#download) as IDE for `R`. It is not the only possibility but it is a standard choice and highly recommended. Once you downloaded it, write the following lines to unpack it.
```shell
$ cd Downloads/
$ sudo apt-get install gdebi 
$ sudo gdebi rstudio-1.3.1093-amd64.deb 
```
The last command need to be modified accordin to the downloaded version. <br/>
8. The very last step is to install the `devtools` package. This installation may take few minutes since has many dependencies that rely on external libraries. It is indeed suggested to install them at this stage simply typing 
```shell
$ sudo apt-get install libssl-dev libcurl4-openssl-dev libxml2-dev libgit2-dev libnode-dev
```
9. Once they have all been installed, you are ready to complete this first step. Open Rstudio and launch this command from `R` console.
```R
install.packages("devtools")
```

### GSL

The GNU Scientific Library is linked to the package as external library. It cannot be installed directly from `R`. Fortunately it is available through the package manager, just use the command
```shell
$ sudo apt-get install libgsl-dev
```
In order to test if everything went smoothly, try 
```shell
$ gsl-config --cflags
-I/usr/include #where the header files are
$ gsl-config --libs
-L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm #where the linked libraries are
```

### HDF5
Like **GSL**, also this library can be directly downloaded from the package manager with
```shell
$ sudo apt-get install libhdf5-dev
```
It is probably going to ask to install some auxiliary packeges. Say yes and continue. As before, if everything is fine, something like this should be displayed
```shell
$ pkg-config --cflags hdf5
-I/usr/include/hdf5/serial #where the header files are
$ pkg-config --libs hdf5
-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 #where the linked libraries are
```

### R depencencies

**BGSL**'s core code is all written in `C++` language and exploits the `Rcpp` package to interfece with `R` which allow to map many `R` data types and objects back and forth to their `C++` equivalents.
`Rcpp` by itself does not map matrices and vectors in **Eigen** types, for that the `RcppEigen` package has to be used. It provides all the necessary header files of the 
[Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page). Finally, the last requirement is to install the [`RcppParallel`](http://rcppcore.github.io/RcppParallel/) package. It is used to
achive cheap and portable parallelization via [Intel(R) TBB](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onetbb.html). 
The last mandatory package to be installed is `mathjaxr`. It is needed to improve the style mathematical forumals are written in the documentation.
Other useful packages are `fields` and `plot.matrix`, used to plot matrices. It is not mandatory to install them, it is just a suggestion to visualize the graphs.
All those packages can be installed directly from the `R` console via
```R
install.packages(c("Rcpp", "RcppEigen", "RcppParallel", "fields", "tidyverse", "plot.matrix", "mathjaxr"))
```

### BGSL
You are now ready to download, build and install **BGSL**. Open `R` and run 
```R
devtools::install_github("alessandrocolombi/BGSL")
library("BGSL")
```

## Installation - Windows

Windows users will have to leave their comfort zone when installing this package as things get a little more complicated. There are two difficulties here. 
First of all, note that installing a package is a non-trivial process, involving making different kinds of documentation, checking `R` code for sanity, compiling `C++`  and so on. All those steps require a whole series of external tools like make, sed, tar, gzip, a C/C++ compiler etc. collectively known as a toolchain. All those helpers are immediately available in Linux distribution but they are not on standard Windows boxes and would need to be provided. The second problem will be to make **GSL** and **HDF5** available at linking stage when **BGSL** would be compiled. The issue here is that Unix files and directories are organized in a standard way and libraries are usually automatically manged via package managing programs. What has to be done is exploiting the `R` toolchain provided with **Rtools** to manage libraries emulating a Unix procedure. 

### Rtools

First step is to have `R` installed on your computer and integrated with [Rstudio](https://rstudio.com/products/rstudio/download/#download) as IDE. If you do not have it already install, this should
be easy, just follow [these instructions](https://cran.r-project.org/bin/windows/base/). In case you already have it available, make sure it is at least version 4.0.2. <br/>
The second step is to make [Rtools40](https://cran.r-project.org/bin/windows/Rtools/) work. It is likely that you already have it available, just check by typing in `R` console
```R
Sys.which("make")
> "C:\\rtools40\\usr\\bin\\make.exe" 
```
If that output is displayed, you are good to go. If not, instructions given in the link above should guide in installation process. Once you have done, install `devtools` and all its dependencies with
```R
install.packages("devtools")
```
### GSL

At linking stage, **BGSL** would need to know where the GNU Scientific Library is. The main reference to make it available is [Rtools Packages](https://github.com/r-windows/rtools-packages).
It offers an automatized procedure for both 32 and 64 bits architectures.

1. Open a Rtool Bash terminal, you can find it in "C:\rtools40\mingw64.exe" or "C:\rtools40\msys2.exe" or using the shortcut in your Start Menu. Both executables should work fine. <br/>
**Note** it is in general not equivalent to the default terminal that is available in Rstudio (Alt+Shift+R). That usually opens the Command Prompt or Git Bash. It is suggested to open the correct
terminal as explained above.
2. `cd` into [Rtools Packages directory](https://github.com/r-windows/rtools-packages). The easiest way to that is by cloning it with Git. Make sure if it is already available with
```shell
$ where git
C:\Program Files\Git\cmd\git.exe #if so, you already have it
```
otherwise you can get [git for windows](https://gitforwindows.org/). It is also suggested to 
[enable version control with Rstudio](https://support.rstudio.com/hc/en-us/articles/200532077-Version-Control-with-Git-and-SVN). <br/>
It is now possible to clone the repository 
```shell
$ git clone https://github.com/r-windows/rtools-packages.git
$ cd rtools-packages
```
3. Install **GSL** in window, open the Rtools bash terminal and type
```shell
$ pacman -Sy 
$ pacman -S mingw-w64-{x86_64,i686}-gsl
```
The first command updates the packages and the second one installs the library. Installation may take a while.
4. Install **HDF5** in window, open the Rtools bash terminal and type
```shell
$ pacman -Sy 
$ pacman -S mingw-w64-{x86_64,i686}-hdf5
```
Installation may take a while also in this case.
<!-- 
3. Compile and install GSL, you just need to write
```shell
$ cd mingw-w64-gsl
$ makepkg-mingw --syncdeps --noconfirm
```
This operation may take a while, up to 20 minutes. If successful, it will automatically generate binary packages for 32 and 64 bit mingw-w64. You can install these typing:
```shell
$ pacman -U mingw-w64-i686-gsl-2.6-1-any.pkg.tar.xz
$ pacman -U mingw-w64-x86_64-gsl-2.6-1-any.pkg.tar.xz
$ rm -f -r pkg src *.xz *.gz #clean the folder
$ cd ../
```
If 32-bit version is needed, perform the same test but with "C:\rtools40\mingw32.exe" terminal.

### HDF5

All you have to do now is to repeat the same procedure as for GSL. Open a Rtools Bash terminal as previously explained and go into [Rtools Packages directory](https://github.com/r-windows/rtools-packages). It should now be available  without cloning again.
```shell
$ cd rtools-packages
$ cd mingw-w64-hdf5
$ makepkg-mingw --syncdeps --noconfirm
$ pacman -U mingw-w64-i686-hdf5-1.10.5-9002-any.pkg-tar.xz
$ pacman -U mingw-w64-x86_64-hdf5-1.10.5-9002-any.pkg-tar.xz
$ rm -f -r pkg src *.xz *.gz *.bz2 #clean the folder
$ cd ../
```
Building and installing **HDF5** may take a while as well.
 -->


### R depencencies

No difference with Unix system here, [see](#R-depencencies) for details or simply type on `R` console
```R
install.packages(c("Rcpp", "RcppEigen", "RcppParallel", "fields", "tidyverse" ,"plot.matrix", "mathjaxr"))
```

### BGSL
You are now ready to dowload, build and install **BGSL**. Open `R` and run 
```R
devtools::install_github("alessandrocolombi/BGSL")
library("BGSL")
```