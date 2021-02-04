#ifndef __BGSLEXPORT_HPP__
#define __BGSLEXPORT_HPP__

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#define STRICT_R_HEADERS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "BGSLheaders.h"

using MatRow        = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatCol        = Eigen::MatrixXd;
using MatUnsRow     = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatUnsCol     = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using VecCol        = Eigen::VectorXd;
using VecRow        = Eigen::RowVectorXd;


 

//' A direct sampler for GWishart distributed random variables.  
//'
//'\loadmathjax This function draws a random matrices, distributed according to the GWishart distribution with Shape parameter \code{b} and Inverse-Scale matrix \code{D}, 
//' whose structure is constrained by graph \code{G}. The GWishart distribution, taking into account a Shape-Inverse Scale parametrization, is the following:
//' \mjsdeqn{p(K~|~ G, b,D) = I_{G}\left(b, D\right)^{-1} |K|^{\frac{b - 2}{2}} \exp\left( - \frac{1}{2}tr\left(K D\right)\right)}
//' It works with both decomposable and non decomposable graphs. In particular it is possible to provide a graph in block form. 
//' @param G matrix representing the desired graph. It has to be a \mjseqn{p \times p} matrix if the graph is in block form, i.e if groups is non null, 
//' otherwise it has to be coherent with the number of groups. Only the upper triangular part is needed.
//' @param b GWishart Shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D It is a \mjseqn{p \times p} matrix. Different parametrizations are possible, they are handled by the \code{form} input parameter.
//' @param norm String to choose the matrix norm with respect to whom convergence takes place. The available choices are \code{"Mean"}, \code{"Inf"}, \code{"One"} and \code{"Squared"}. 
//' \code{"Mean"} is the default value and it is also used when a wrong input is provided.
//' @param form String, states what type of parameter is represented by \code{D}. Possible values are \code{"Scale"} for Scale parametrization, 
//' \code{"InvScale"} for Inverse-Scale parametrization or \code{"CholLower_InvScale"} and \code{"CholUpper_InvScale"} to pass directly the Cholesky factorization of the Inverse-Scale matrix.
//' Usually GWishart distributions are parametrized with respect to the Inverse Scale matrix. However the first step of the sampling requires the Scale matrix parameter or, even better, its Cholesky decomposition. 
//' This functions leaves a lot of freedom to the user so that the most efficient available parametrization can be used.
//' @param groups List representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave NULL if the graph is not in block form.
//' @param check_structure bool, if \code{TRUE} it is checked if the sampled matrix actually respects the structure of the graph.
//' @param max_iter unsigned int, the maximum number of iteration.
//' @param threshold_check the accurancy for checking if the sampled matrix respects the structure of the graph.
//' @param threshold_conv  the threshold value for the convergence of sampling algorithm from GWishart. Algorithm stops if the difference between two subsequent iterations is less than \code{threshold_conv}.
//' @param seed integer, seeding value. Set 0 for random seed.
//' @return A list is returned, it is composed of: \code{Matrix} that contains the random matrix just sampled, \code{Converged} that is a boolean that states if the algorithm reached convergence or not, 
//' the number performed iterations can be found in \code{iterations} and, if requested in the \code{check_structure} parameters, an additional boolean called \code{CheckStructure} is returned.
//' @export
// [[Rcpp::export]]
Rcpp::List rGwish(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const & G,
                  double const & b, Eigen::MatrixXd & D,Rcpp::String norm = "Mean", Rcpp::String form = "InvScale", 
                  Rcpp::Nullable<Rcpp::List> groups = R_NilValue, bool check_structure = false,
                  unsigned int const & max_iter = 500, long double const & threshold_check = 0.00001, long double const & threshold_conv = 0.00000001, int seed = 0)
{
  sample::GSL_RNG engine( static_cast<unsigned int>(seed) );
  if (groups.isNotNull()){ //Assume it is a BlockGraph   
    Rcpp::List L(groups); // casting to underlying type List
    std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
    BlockGraph<unsigned int> Graph(G, ptr_gr);
    auto rgwish_fun = utils::build_rgwish_function<CompleteView, unsigned int>(form, norm);
    auto [Mat, converged, iter] = rgwish_fun(Graph.completeview(), b, D, threshold_conv, engine, max_iter);
    if(check_structure){
      return Rcpp::List::create ( Rcpp::Named("Matrix")= Mat, 
                                  Rcpp::Named("Converged")=converged, 
                                  Rcpp::Named("iterations")=iter, 
                                  Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)  );
    }
    else{
      return Rcpp::List::create ( Rcpp::Named("Matrix")= Mat, 
                                  Rcpp::Named("Converged")=converged, 
                                  Rcpp::Named("iterations")=iter );
    }
    
  }
  else{ //Assume it is a Complete Graph
    GraphType<unsigned int> Graph(G);
    auto rgwish_fun = utils::build_rgwish_function<GraphType, unsigned int>(form, norm);
    auto [Mat, converged, iter] = rgwish_fun(Graph.completeview(), b, D, threshold_conv, engine, max_iter);
    if(check_structure){
      return Rcpp::List::create ( Rcpp::Named("Matrix")= Mat, 
                                  Rcpp::Named("Converged")=converged, 
                                  Rcpp::Named("iterations")=iter, 
                                  Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)  );
    }
    else{
      return Rcpp::List::create ( Rcpp::Named("Matrix")= Mat, 
                                  Rcpp::Named("Converged")=converged, 
                                  Rcpp::Named("iterations")=iter );
    }
  } 
}


//' Normalizing constant for GWishart distribution
//'
//' \loadmathjax This function computes the logarithm of the normalizing constant of GWishart distribution. Its distribution, taking into account a Shape-Inverse Scale parametrization, is the following:
//' \mjsdeqn{p\left(K~|~ G, b,D \right) = I_{G}\left(b, D\right)^{-1} |K|^{\frac{b - 2}{2}} \exp\left( - \frac{1}{2}tr\left(K D\right)\right)}
//' The Monte Carlo method, developed by Atay-Kayis and Massam (2005), is implemented. It works with both decomposable and non decomposable graphs. 
//' In particular it is possible to provide a graph in block form. 
//' @param G matrix representing the desired graph. It has to be a \mjseqn{p \times p} matrix if the graph is in block form, i.e if groups is non null, 
//' otherwise it has to be coherent with the number of groups. Only the upper triangular part is needed.
//' @param b GWishart Shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D GWishart Inverse-Scale matrix. It has to be of size \mjseqn{p \times p}, symmetric and positive definite. 
//' @param MCiteration the number of iterations for the Monte Carlo approximation. 
//' @param groups a Rcpp list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//' @param seed integer, seeding value. Set 0 for random seed.
//' @return log of the normalizing constant of GWishart distribution.
//' @export
// [[Rcpp::export]]
long double log_Gconstant(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const & G,
                          double const & b, Eigen::MatrixXd const & D,  unsigned int const & MCiteration = 500, 
                          Rcpp::Nullable<Rcpp::List> groups = R_NilValue, int seed = 0)
{
    sample::GSL_RNG engine( static_cast<unsigned int>(seed) );
    if (groups.isNotNull()){ //Assume it is in block form with respect to the groups given in groups
      
      Rcpp::List L(groups); // casting to underlying type List
      //Rcpp::Rcout << "List is not NULL." << std::endl;
      std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
      BlockGraph<unsigned int> Graph(G, ptr_gr);
      return utils::log_normalizing_constat(Graph.completeview(), b, D, MCiteration, engine);
    }
    else{ //Assume it is a BlockGraph
      GraphType<unsigned int> Graph(G);
      return utils::log_normalizing_constat(Graph.completeview(), b, D, MCiteration, engine);
    }
}




//' Generate a random graph
//'
//' \loadmathjax This function genrates random graphs both in \code{"Complete"} or \code{"Block"} form. 
//' @param p integer, the dimension of the underlying graph. It has to be provided even if \code{form} is \code{"Block"}.
//' @param n_groups iinteger, number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered in \code{groups} parameter.
//' @param form string that may take as values only \code{"Complete"} of \code{"Block"} . It states if the algorithm has to run with \code{"Block"} or \code{"Complete"} graphs.
//' @param groups List representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//' @param sparsity desired sparsity in the graph. It has to be in the range \mjseqn{(0,1)}. 
//' @param seed integer, seeding value. Set 0 for random seed.
//' @return A list is returned, it contains the randomly generated graph in \code{G} and its complete form in \code{G_Complete}. Those two coincide if \code{form} is \code{"Complete"}.
//' @export
// [[Rcpp::export]]
Rcpp::List Create_RandomGraph ( int const & p, int const & n_groups = 0, Rcpp::String form = "Complete", Rcpp::Nullable<Rcpp::List> groups = R_NilValue,
                                 double sparsity = 0.5, int seed = 0  )
{
  if(p <= 0)
    throw std::runtime_error("Wrong dimension inserted, the number of vertrices has to be strictly positive");
  if(form=="Complete"){
    GraphType Graph(p);
    Graph.fillRandom(sparsity, seed); //Corretness of sparsity is checked inside
    return Rcpp::List::create( Rcpp::Named("G")=Graph.get_graph(), Rcpp::Named("G_Complete")=Graph.get_graph() );
  } 
  else if (form == "Block"){

      if (groups.isNotNull()){ //Assume it is a BlockGraph
          Rcpp::List L(groups); // casting to underlying type List
          std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
          BlockGraph<unsigned int> Graph(ptr_gr);
          Graph.fillRandom(sparsity, seed);
          CompleteView<unsigned int> Complete(Graph);
          MatRow Complete_ret(MatRow::Zero(p,p));
          for(unsigned int i = 0; i < p; ++i)
            for(unsigned int j = i; j < p; ++j)
              Complete_ret(i,j) = Complete(i,j);
          return Rcpp::List::create( Rcpp::Named("G")=Graph.get_graph(), Rcpp::Named("G_Complete")=Complete_ret );  
      }
      else{
        if(n_groups <= 1 || n_groups > p)
          throw std::runtime_error("Incoherent number of groups inserted, it has to be at least 2 and smaller than p");
        else{
          std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(n_groups, p); //Create pointer to groups
          BlockGraph<unsigned int> Graph(ptr_gr);
          Graph.fillRandom(sparsity, seed);
          CompleteView<unsigned int> Complete(Graph);
          MatRow Complete_ret(MatRow::Zero(p,p));
          for(unsigned int i = 0; i < p; ++i)
            for(unsigned int j = i; j < p; ++j)
              Complete_ret(i,j) = Complete(i,j);
          return Rcpp::List::create( Rcpp::Named("G")=Graph.get_graph(), Rcpp::Named("G_Complete")=Complete_ret );  
        }
      }
  } 
  else
    throw std::runtime_error("The only possible forms are Complete and Block");
}

//' Sampler for Multivariate Normal random variables
//'
//' \loadmathjax This function draws random samples from Multivariate Gaussian distribution. It allows for both covariance and precision parametrization.
//' It is also possible to pass directly the Cholesky decomposition if it is available before the call.
//' @param mean vector of size \code{p} representig the mean of the Gaussian distribution.
//' @param Mat matrix of size \mjseqn{p \times p} representinf the covariance or the precision matrix or their Cholesky decompositions.
//' @param isPrec boolean, set \code{TRUE} if Mat parameter is a precision, \code{FALSE} if it is a covariance.
//' @param isChol boolean, set \code{TRUE} if Mat parameter is a triangular matrix representig the Cholesky decomposition of the precision or covariance matrix.
//' @param isUpper boolean, used only if \code{isChol} is \code{TRUE}. Set \code{TRUE} if Mat is upper triangular, \code{FALSE} if lower.
//' @param seed integer, seeding value. Set 0 for random seed.
//' @return It returns a \mjseqn{p}-dimensional vector with the sampled values.
//' @export
// [[Rcpp::export]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
rmvnormal(Eigen::VectorXd mean, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Mat, 
          bool isPrec = false, bool isChol = false, bool isUpper = false, int seed = 0) 
{
  sample::GSL_RNG engine( static_cast<unsigned int>(seed) );
  if(Mat.rows() != Mat.cols())
    throw std::runtime_error("Precision or Covariance matrix has to be symmetric.");
  if(!isPrec){ //Covariance parametrization
    if(!isChol){
      return sample::rmvnorm<sample::isChol::False>()(engine, mean, Mat);
    }
    else if(isUpper){
     return sample::rmvnorm<sample::isChol::Upper>()(engine, mean, Mat);
    }
    else{
     return sample::rmvnorm<sample::isChol::Lower>()(engine, mean, Mat);
    }
  }
  else{ //Precision parametrization
    if(!isChol){ //Not chol
      return sample::rmvnorm_prec<sample::isChol::False>()(engine, mean, Mat);
    }
    else if(isUpper){ //Chol, upper triangular
      return sample::rmvnorm_prec<sample::isChol::Upper>()(engine, mean, Mat);
    }
    else{ //Chol, lower triangular 
      return sample::rmvnorm_prec<sample::isChol::Lower>()(engine, mean, Mat);
    }
  }
}

//' Sampler for Wishart random variables
//'
//' \loadmathjax This function draws random samples from Wishart distribution. We use a Shape-Inverse Scale parametrization, the corresponding density is reported below
//' \mjsdeqn{ f(X) = \frac{|X|^{(b-2)/2}~~\exp\left( - \frac{1}{2}tr\left(X~D\right)\right)}{2^{p(b+p-1)/2}~|D^{-1}|^{(b+p-1)/2}~\Gamma_{p}((b+p-1)/2)}}
//' It is also possible to pass directly the Cholesky decomposition of the Inverse Scale matrix if it is available before the call.
//' @param b it is Shape parameter. 
//' @param D Inverse-Scale matrix of size \mjseqn{p \times p}. It may represent its Cholesky decomposition.
//' @param isChol boolean, set \code{TRUE} if D parameter is a triangular matrix representig the Cholesky decomposition of the precision or covariance
//' @param isUpper boolean, used only if isChol is \code{TRUE}. Set \code{TRUE} if D is upper triangular, \code{FALSE} if lower.
//' @param seed integer, seeding value. Set 0 for random seed.
//' @return It returns the \mjseqn{p \times p} sampled matrix.
//' @export
// [[Rcpp::export]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
rwishart(double const & b, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> D, 
         bool isChol = false, bool isUpper = false, int seed = 0)
{
  using MatCol = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  sample::GSL_RNG engine( static_cast<unsigned int>(seed) );
  if(D.rows() != D.cols())
    throw std::runtime_error("Inverse scale matrix has to be symmetric.");
  if(!isChol){ //Not chol
    return sample::rwish<MatCol, sample::isChol::False>()(engine, b, D);
  }
  else if(isUpper){ //Chol, upper triangular
    return sample::rwish<MatCol, sample::isChol::Upper>()(engine, b, D);
  }
  else{ //Chol, lower triangular 
    return sample::rwish<MatCol, sample::isChol::Lower>()(engine, b, D);
  }  
}

//' Sampler for Normal distribution
//' 
//' This function draw a sample from the Gaussian distribution.
//' @param mean the mean value.
//' @param sd the standard deviation.
//' @param seed integer, seeding value. Set 0 for random seed.
//'
//' @return A single drawn values from N(mean,sd*sd).
//' @export
// [[Rcpp::export]]
double rnormal(double const & mean = 0.0, double const & sd = 1.0, int seed = 0){
  sample::GSL_RNG engine( static_cast<unsigned int>(seed) );
  return sample::rnorm()(engine, mean, sd);
}


//' Generate Bspine basis 
//' 
//' \loadmathjax This function creates a truncated Bspline basis in the interval \mjseqn{\[range(1),range(2)\]} and evaluate them over a grid of points.
//' It assumes uniformly spaced breakpoints and constructs the corresponding knot vector using a number of breaks equal to \mjseqn{n\_basis + 2 - order}.
//' @param n_basis the number of basis functions.
//' @param range vector of two elements containing first the lower and then the upper bound of the interval.
//' @param points number of grid points where the basis has to be evaluated. It is not used if the points are directly passed in the \code{grid_points} parameter.
//' @param grid_points vector of points where the basis has to be evaluated. If defaulted, then \code{n_points} are uniformly generated in the interval.
//' @param order integer, order of the Bsplines. Set four for cubic splines.
//' @return a list containing a matrix of dimension \mjseqn{n\_points \times n\_basis} such that \code{r}-th rows contains all the spline computed in the \code{r}-th point of the grid and \code{j}-th column
//' contains the \code{j}-th spline evaluated in all grid points. It also returns a vector of size \mjseqn{n\_basis + 2 - order} containing the internal knots used to create the spline.
//' @export
// [[Rcpp::export]]
Rcpp::List Generate_Basis(int const & n_basis, Rcpp::NumericVector range, int n_points = 0, 
                          Rcpp::NumericVector grid_points = Rcpp::NumericVector::create(0), int order = 3 )
{
  if(n_basis <= 0)
    throw std::runtime_error("Invalid number of basis or grid points");
  if( !(range.size() == 2 && range[0] < range[1]))
    throw std::runtime_error("Irregular range inserted. It has to be a vector of two elements containing first the lower and then the upper bound of the interval.");
  if( grid_points.size() > 1 ){
     std::vector<double> points(grid_points.begin(), grid_points.end());
     auto[BaseMat, knots] = spline::generate_design_matrix(order, n_basis, range[0], range[1], points);
     return Rcpp::List::create(Rcpp::Named("BaseMat")=BaseMat,Rcpp::Named("InternalKnots")=knots);
  }
  else if (n_points > 2){
    auto[BaseMat, knots] = spline::generate_design_matrix(order, n_basis, range[0], range[1], n_points);
    return Rcpp::List::create(Rcpp::Named("BaseMat")=BaseMat,Rcpp::Named("InternalKnots")=knots);
  }
  else    
    throw std::runtime_error("Invalid number of grid points inserted. They have to be at least three");
}

//' Generate Bspine basis and its derivatives
//'
//' \loadmathjax This function creates a truncated Bspline basis in the interval \mjseqn{\[range(1),range(2)\]} and evaluate them over a grid of points up to derivative of order \code{nderiv}.
//' For convention, derivatives of order 0 are the splines themselves. This implimes that the first returned element is always equal to the output of the function \code{\link{Generate_Basis}}.
//' @param n_basis the number of basis functions.
//' @param nderiv number of derivates that have to be computed. It can also be 0.
//' @param range vector of two elements containing first the lower and then the upper bound of the interval.
//' @param points number of grid points where the basis has to be evaluated. It is not used if the points are directly passed in the \code{grid_points} parameter.
//' @param grid_points vector of points where the basis has to be evaluated. If defaulted, then \code{n_points} are uniformly generated in the interval.
//' @param order integer, order of the Bsplines. Set four for cubic splines.
//' @return a list of length nderiv+1 such that each element is a \mjseqn{n\_points \times n\_basis} matrix representing the evaluation of 
//' the \code{k}-th derivative of all the splines in all the grid points. 
//' @export
// [[Rcpp::export]]
Rcpp::List Generate_Basis_derivatives(int const & n_basis, int const & nderiv, Rcpp::NumericVector range, int n_points = 0,
                                       Rcpp::NumericVector grid_points = Rcpp::NumericVector::create(0), int order = 3  )
{
  using MatCol = Eigen::MatrixXd;
  if(n_basis <= 0)
    throw std::runtime_error("Invalid number of basis or grid points");
  if( !(range.size() == 2 && range[0] < range[1]))
    throw std::runtime_error("Irregular range inserted. It has to be a vector of two elements containing first the lower and then the upper bound of the interval.");
  if( grid_points.size() > 1 ){
     std::vector<double> points(grid_points.begin(), grid_points.end());
     std::vector<MatCol> res = spline::evaluate_spline_derivative(order, n_basis, range[0], range[1], points, nderiv);
     Rcpp::List ret_list(res.size());
     for(int i = 0; i < res.size(); ++i){
        ret_list[i] = res[i];
     }
     return ret_list;
  }
  else if (n_points > 2){
    std::vector<MatCol> res = spline::evaluate_spline_derivative(order, n_basis, range[0], range[1], n_points, nderiv);
    Rcpp::List ret_list(res.size());
    for(int i = 0; i < res.size(); ++i){
       ret_list[i] = res[i];
    }
    return ret_list;
  }
  else    
    throw std::runtime_error("Invalid number of grid points inserted. They have to be at least three");
}


//' Read information from file
//'
//' \loadmathjax Read from \code{file_name} some information that are needed to extract data from it. 
//' @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
//' @return It returns a list containing \code{p}, the dimension of the graph, \code{n} the number of observed data, \code{stored_iter} the number of saved iterations for the regression parameters,
//' \code{stored_iterG} the number of saved iterations for the graphical related quantities, i.e the graph and the precision matrix. Finally, \code{sampler} recalls what type of sampler was used. 
//' Possibilities are \code{"GGMsampler"}, \code{"FGMsampler"}, \code{"FLMsampler_diagonal"} or \code{"FLMsampler_fixed"}.
//' @export
// [[Rcpp::export]]
Rcpp::List Read_InfoFile( Rcpp::String const & file_name )
{
  //std::vector< unsigned int > info =  HDF5conversion::GetInfo(file_name);
  auto [info, sampler] =  HDF5conversion::GetInfo(file_name);
  return  Rcpp::List::create ( Rcpp::Named("p") = info[0], 
                               Rcpp::Named("n") = info[1], 
                               Rcpp::Named("stored_iter") = info[2],   
                               Rcpp::Named("stored_iterG") = info[3],
                               Rcpp::Named("sampler") = sampler
                             );
}


//' Compute quantiles of sampled values
//'
//' \loadmathjax This function reads the sampled values saved in a binary file and computes the quantiles of the desired level.
//' @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
//' @param Beta boolean, set \code{TRUE} to compute the quantiles for all \code{p*n} \mjseqn{\beta} coefficients. It may require long time.
//' @param Mu boolean, set \code{TRUE} to compute the quantiles for all \mjseqn{p} parameters. 
//' @param TauEps boolean, set \code{TRUE} to compute the quantiles of \mjseqn{\tau_{\epsilon}} parameter.
//' @param Precision boolean, set \code{TRUE} to compute the mean for all the elements of the precision matrix 
//' or the \mjseqn{\tau_{j}} coefficients if the file contains the output of a \code{\link{FLMsampling}}, diagonal version.
//' @param lower_qtl the level of the first desired quantile.
//' @param upper_qtl the level of the second desired quantile.
//'
//' @return It returns a list containig the upper and lower quantiles of the requested quantities.
//' @export
// [[Rcpp::export]]
Rcpp::List Compute_Quantiles( Rcpp::String const & file_name, bool Beta = false, bool Mu = false, bool TauEps = false, bool Precision = false,
                               double const & lower_qtl = 0.05, double const & upper_qtl = 0.95  )
{
  
  Rcpp::List Quantiles = Rcpp::List::create( Rcpp::Named("Beta"),      
                                             Rcpp::Named("Mu"),        
                                             Rcpp::Named("TauEps"),    
                                             Rcpp::Named("Precision") 
                                           );

  //Read file info
  auto [info, sampler] =  HDF5conversion::GetInfo(file_name);
  const unsigned int& p = info[0];
  const unsigned int& n = info[1];
  const unsigned int& stored_iter  = info[2];
  const unsigned int& stored_iterG = info[3]; 
  if(!(sampler == "FLMsampler_diagonal" || sampler == "FLMsampler_fixed" || sampler == "FGMsampler" || sampler == "GGMsampler"))
    throw std::runtime_error("Unrecognized sampler type, it can only be: FLMsampler_diagonal, FLMsampler_fixed, FGMsampler or GGMsampler");

  if(!(Precision || Beta || Mu || TauEps))
    Rcpp::Rcout<<"All possible parameters were FALSE, no mean has been computed"<<std::endl;
  
  if(Precision){
    if(stored_iterG <= 0)
          throw std::runtime_error("stored_iterG parameter has to be positive");
    unsigned int  prec_elem = 0;
    if(sampler == "FLMsampler_diagonal"){
      prec_elem = p;
      Rcpp::Rcout<<"Compute TauK quantiles..."<<std::endl;
      auto [Lower, Upper] =  analysis::Vector_ComputeQuantiles( file_name, stored_iterG, prec_elem, "Precision", lower_qtl, upper_qtl );
      Quantiles["Precision"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
    }
    else{
      prec_elem = 0.5*p*(p+1);
      Rcpp::Rcout<<"Compute Precision quantiles..."<<std::endl;
      auto [Lower_vett, Upper_vett] =  analysis::Vector_ComputeQuantiles( file_name, stored_iterG, prec_elem, "Precision", lower_qtl, upper_qtl );
      MatRow Lower(MatRow::Zero(p,p));  
      MatRow Upper(MatRow::Zero(p,p));  
      unsigned int pos{0};
      for(unsigned int i = 0; i < p; ++i){
        for(unsigned int j = i; j < p; ++j){
          Lower(i,j) = Lower_vett(pos);
          Upper(i,j) = Upper_vett(pos);
          pos++;
        }
      }
      Quantiles["Precision"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
    }
    
  }
  if(Beta){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no Beta coefficient for this sampler.");
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    Rcpp::Rcout<<"Compute Beta quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Matrix_ComputeQuantiles( file_name, stored_iter, p, n, lower_qtl, upper_qtl );
    Quantiles["Beta"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  if(Mu){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no Mu coefficient for this sampler.");
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    Rcpp::Rcout<<"Compute Mu quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Vector_ComputeQuantiles( file_name, stored_iter, p, "Mu", lower_qtl, upper_qtl );
    Quantiles["Mu"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  if(TauEps){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no TauEps coefficient for this sampler.");
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    Rcpp::Rcout<<"Compute TauEps quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Scalar_ComputeQuantiles( file_name, stored_iter, lower_qtl, upper_qtl );
    Quantiles["TauEps"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  return Quantiles;
}


//' Compute Posterior means of sampled values
//'
//' \loadmathjax This function reads the sampled values saved in a binary file and computes the mean of the requested quantities.
//' @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
//' @param Beta boolean, set \code{TRUE} to compute the mean for all \code{p*n} \mjseqn{\beta} coefficients. It may require long time.
//' @param Mu boolean, set \code{TRUE} to compute the mean for all \mjseqn{p} parameters. 
//' @param TauEps boolean, set \code{TRUE} to compute the mean of \mjseqn{\tau_{\epsilon}} parameter.
//' @param Precision boolean, set \code{TRUE} to compute the mean for all the elements of the precision matrix 
//' or the \mjseqn{\tau_{j}} coefficients if the file contains the output of a \code{\link{FLMsampling}}, diagonal version.
//' @return It returns a list containig the mean of the requested quantities.
//' @export
// [[Rcpp::export]]
Rcpp::List Compute_PosteriorMeans( Rcpp::String const & file_name, bool Beta = false, bool Mu = false, bool TauEps = false, bool Precision = false)
{

  Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta"), 
                                                   Rcpp::Named("MeanMu"), 
                                                   Rcpp::Named("MeanK"),   
                                                   Rcpp::Named("MeanTaueps") );  
  //Read file info
  auto [info, sampler] =  HDF5conversion::GetInfo(file_name);
  const unsigned int& p = info[0];
  const unsigned int& n = info[1];
  const unsigned int& stored_iter = info[2];
  const unsigned int& stored_iterG = info[3];
  if(!(sampler == "FLMsampler_diagonal" || sampler == "FLMsampler_fixed" || sampler == "FGMsampler" || sampler == "GGMsampler"))
    throw std::runtime_error("Unrecognized sampler type, it can only be: FLMsampler_diagonal, FLMsampler_fixed, FGMsampler or GGMsampler");

  if(!(Precision || Beta || Mu || TauEps))
    Rcpp::Rcout<<"All possible parameters were FALSE, no mean has been computed"<<std::endl;
  if(Precision){
    if(stored_iterG <= 0)
      throw std::runtime_error("stored_iterG parameter has to be positive");
    unsigned int  prec_elem = 0;
    if(sampler == "FLMsampler_diagonal"){
      prec_elem = p;
      VecCol MeanK_vett = analysis::Vector_PointwiseEstimate(file_name, stored_iterG, prec_elem, "Precision" );
      PosteriorMeans["MeanK"] = MeanK_vett;
    }
    else{
      prec_elem = 0.5*p*(p+1);
      MatRow MeanK(MatRow::Zero(p,p));  
      VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate(file_name, stored_iterG, prec_elem, "Precision" );
      unsigned int pos{0};
      for(unsigned int i = 0; i < p; ++i){
        for(unsigned int j = i; j < p; ++j){
          MeanK(i,j) = MeanK_vett(pos++);
        }
      }
      PosteriorMeans["MeanK"] = MeanK;
    }
  } //End Precision
  if(Beta){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no Beta coefficient for this sampler.");
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    MatCol MeanBeta = analysis::Matrix_PointwiseEstimate( file_name, stored_iter, p, n );
    PosteriorMeans["MeanBeta"] = MeanBeta;
  }
  if(Mu){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no Mu coefficient for this sampler."); 
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    VecCol MeanMu = analysis::Vector_PointwiseEstimate( file_name, stored_iter, p, "Mu" );
    PosteriorMeans["MeanMu"] = MeanMu;
  }
  if(TauEps){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no TauEps coefficient for this sampler.");
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    double MeanTaueps =  analysis::Scalar_PointwiseEstimate( file_name, stored_iter );
    PosteriorMeans["MeanTaueps"] = MeanTaueps;
  }
  return PosteriorMeans;
}


//' Read chain from file
//'
//' \loadmathjax This function read from a binary file the sampled chain for the \code{index1}-th component of variable specified in \code{variable}. The chain is then saved in memory to make it available for further analysis.
//' Both \code{index1} and \code{index2} start counting from 1, the first element is obtained by settin \code{index1} equal to 1, not 0.
//' Only \code{"Beta"} coefficients require the usage of \code{index2}. This function allows to extract only one chain at the time. The idea of writing the sampled values on a file is indeed to
//' avoid to fill the memory. This function has to carefully used.
//' @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
//' @param variable string, the name of the dataset to be read from the file. Only possibilities are \code{"Beta"}, \code{"Mu"}, \code{"Precision"} and \code{"TauEps"}.
//' @param index1 integer, the index of the element whose chain has to read from the file. The first elements corresponds to \code{index1} equal to 1. 
//' If \code{variable} is equal to \code{"Precision"} some care is required. If the file represents the output of \code{GGM}, \code{FGM} of \code{FLM} with fixed graph, the sampled precision matrices 
//' are of size \mjseqn{p \times p} stored row by row. This means that the second diagonal elements corresponds to \code{index1} equal to \mjseqn{p+1}. 
//' Moreover, in this case set \code{prec_elem} parameter equal to \mjseqn{\frac{p(p+1)}{2}}. 
//' Instead, for outputs coming from \code{FLM} sampler with diagonal graph, only the diagonal of the precision matrix is saved. If so, \code{index1} ranges from 1 up to \mjseqn{p}. Moreover, set \code{prec_ele} eqaul to \mjseqn{p}.
//' If \code{variable} is equal to \code{"Beta"}, this index ranges for 1 up to \mjseqn{p}, it represents the spine coefficinet.
//' @param index2 integer, to be used only if \code{variable} is equal to \code{"Beta"}. It ranges from 1 up to \mjseqn{n}. In this case, the chain for the spline_index-th coefficients of the curve_index-th curve is read.
//'
//' @return It returns a numeric vector all the sampled values of the required element.
//' @export
// [[Rcpp::export]]
Eigen::VectorXd Extract_Chain( Rcpp::String const & file_name, Rcpp::String const & variable, unsigned int  index1 = 1, unsigned int index2 = 1 )
{ 

  if(variable != "Beta" && variable != "Mu" && variable != "Precision" && variable != "TauEps")
    throw std::runtime_error("Error, only possible values for variable are Beta, Precision, Mu, TauEps");

  if(index1 <= 0 || index2 <= 0)
    throw std::runtime_error("index1 and index2 parameters start counting from 1, not from 0. The first element corresponds to index 1, not 0. The inserted values has to be strictly positive");

  index1 = index1 - 1;
  index2 = index2 - 1;

  HDF5conversion::FileType file;
  HDF5conversion::DatasetType dataset_rd;
  std::string file_name_stl{file_name};

  //Read file info
  auto [info, sampler] =  HDF5conversion::GetInfo(file_name);
  const unsigned int& p = info[0];
  const unsigned int& n = info[1];
  const unsigned int& stored_iter = info[2];
  const unsigned int& stored_iterG = info[3];
  if(!(sampler == "FLMsampler_diagonal" || sampler == "FLMsampler_fixed" || sampler == "FGMsampler" || sampler == "GGMsampler"))
    throw std::runtime_error("Unrecognized sampler type, it can only be: FLMsampler_diagonal, FLMsampler_fixed, FGMsampler or GGMsampler");

  //Open file
  file=H5Fopen(file_name_stl.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
  if(file < 0)
    throw std::runtime_error("Error, can not open the file. Probably it was not closed correctly");

  std::vector<double> chain;
  if(variable == "Beta"){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no Beta coefficient for this sampler.");
    if(n <= 0)
      throw std::runtime_error("The number of observed data n has to be provided to extract the chain for Beta");
    //Open datasets
    dataset_rd  = H5Dopen(file, "/Beta", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open dataset for Beta ");
    chain = HDF5conversion::GetChain_from_Matrix(dataset_rd, index1, index2, stored_iter, n);
    H5Dclose(dataset_rd);
  }
  else if(variable == "Mu"){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no Mu coefficient for this sampler.");
    //Open datasets
    dataset_rd  = H5Dopen(file, "/Mu", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open dataset for MU");
    chain = HDF5conversion::GetChain_from_Vector(dataset_rd, index1, stored_iter, p);
    H5Dclose(dataset_rd);
  }
  else if(variable == "Precision"){
    unsigned int prec_elem{0};
    if(sampler == "FLMsampler_diagonal"){
      prec_elem = p;
    }
    else{
      prec_elem = 0.5*p*(p+1);
    }    
    //Open datasets
    dataset_rd  = H5Dopen(file, "/Precision", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open dataset for Precision");
    chain = HDF5conversion::GetChain_from_Vector(dataset_rd, index1, stored_iter, prec_elem);
    H5Dclose(dataset_rd);
  }
  else if(variable == "TauEps"){
    if(sampler == "GGMsampler")
      throw std::runtime_error("The file was recognized as output of a GGMsampler. There is no TauEps coefficient for this sampler.");
    //Open datasets
    dataset_rd  = H5Dopen(file, "/TauEps", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open the dataset for TauEps");
    chain.resize(stored_iter);
    double * buffer = chain.data();
    HDF5conversion::StatusType status = H5Dread(dataset_rd, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    if(status < 0)
      throw std::runtime_error("Error, can not read the chain for TauEps");
    H5Dclose(dataset_rd);
  }
  else{
    throw std::runtime_error("Error, only possible values for variable are Beta, Precision, Mu, TauEps");
  }

  H5Fclose(file);

  Eigen::Map< Eigen::VectorXd > result(&(chain[0]), chain.size());
  return result;
}

//' Read the sampled Graph saved on file
//'
//' \loadmathjax This function reads the sampled graphs that are saved on a binary file and performs a summary of all visited graphs.
//' @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
//' @param groups List representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//'
//' @return It returns a list composed of: \code{plinks} that contains the posterior probability of inclusion of each possible link. \code{AcceptedMoves} contains the number of
//' Metropolis-Hastings moves that were accepted in the sampling, \code{VisitedGraphs} the number of graph that were visited at least once, \code{TracePlot_Gsize} is a vector 
//' such that each element is equal to the size of the visited graph in that particular iteration and finally \code{SampledGraphs} is a list containing all the visited graphs and their absolute frequence of visit.
//' To save memory, the graphs are represented only by the upper triangular part, stored row-wise. 
//' @export
// [[Rcpp::export]]
Rcpp::List Summary_Graph(Rcpp::String const & file_name, Rcpp::Nullable<Rcpp::List> groups = R_NilValue)
{
  //Read file info
  auto [info, sampler] = HDF5conversion::GetInfo(file_name);
  const unsigned int& p = info[0];
  const unsigned int& n = info[1];
  const unsigned int& stored_iter = info[2];
  const unsigned int& stored_iterG = info[3];
  if(!(sampler == "FGMsampler" || sampler == "GGMsampler"))
    throw std::runtime_error("Sampler type was not recognized as a graphical sampler. Only possibilities are FGMsampler or GGMsampler");

  std::shared_ptr<const Groups> ptr_gruppi = nullptr;
  if (groups.isNotNull()){ //Assume it is a BlockGraph
    Rcpp::List gr(groups);
    ptr_gruppi = std::make_shared<const Groups>(gr); 
    const Groups& gruppi_test = *ptr_gruppi;
  }
  auto [plinks, SampledG, TracePlot, visited] = analysis::Summary_Graph(file_name, stored_iterG, p, ptr_gruppi);
  //Create Rcpp::List of sampled Graphs
  std::vector< Rcpp::List > L(SampledG.size());
  int counter = 0;
  for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
    L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
  }
  return Rcpp::List::create ( Rcpp::Named("plinks")= plinks,  
                              Rcpp::Named("VisitedGraphs")= visited, 
                              Rcpp::Named("TracePlot_Gsize")= TracePlot, 
                              Rcpp::Named("SampledGraphs")= L   );
}

// [[Rcpp::export]]
Rcpp::List SimulateData_GGM_c(unsigned int const & p, unsigned int const & n, unsigned int const & n_groups, Rcpp::String const & form, 
                            Rcpp::String const & graph, 
                            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const & adj_mat,   
                            unsigned int seed, bool mean_null, double const & sparsity, Rcpp::Nullable<Rcpp::List> groups = R_NilValue )
{
  using MatRow = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::MatrixXd U;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Prec_true;
  std::vector<bool> G_true;
  if(form == "Complete"){
    if(graph == "random"){
      std::tie(U, Prec_true, G_true) = utils::SimulateDataGGM_Complete(p,n,seed,mean_null,sparsity);
    }
    else if(graph == "fixed"){
      GraphType<bool> G(adj_mat);
      std::tie(U, Prec_true, G_true) = utils::SimulateDataGGM_Complete(p,n,G,seed,mean_null);
    }
    MatRow G_ret(MatRow::Zero(p,p));
    unsigned int counter = 0;
     for(unsigned int i = 0; i < p; ++i)
      for(unsigned int j = i; j < p; ++j){
        if(i == j)
          G_ret(i,i) = 1;
        else
          G_ret(i,j) = (unsigned int)G_true[counter++];
      }
    
    return Rcpp::List::create( Rcpp::Named("U")=U, Rcpp::Named("Prec_true")=Prec_true, Rcpp::Named("G_true")=G_ret );  
  }
  else{ // Block form required
    std::shared_ptr<const Groups> ptr_gr = nullptr;
    if(groups.isNotNull()){ //groups are provided
      Rcpp::List L(groups);
      ptr_gr = std::make_shared<const Groups>(L);
    }
    else{ //groups are not provided
      ptr_gr = std::make_shared<const Groups>(n_groups, p);
    }
    if(graph == "random"){ //need to simulate the graph
      std::tie(U, Prec_true, G_true) = utils::SimulateDataGGM_Block(p,n,ptr_gr,seed,mean_null,sparsity);
    }
    else{ //graph is provided
      BlockGraph<bool> G(adj_mat, ptr_gr);
      std::tie(U, Prec_true, G_true) = utils::SimulateDataGGM_Block(p,n,G,seed,mean_null);
    }
    BlockGraph<bool> G_ret(G_true, ptr_gr);
    CompleteView<bool> Complete(G_ret);
    MatRow Complete_ret(MatRow::Zero(p,p));
    for(unsigned int i = 0; i < p; ++i)
      for(unsigned int j = i; j < p; ++j)
        Complete_ret(i,j) = Complete(i,j);

    return Rcpp::List::create( Rcpp::Named("U")=U, 
                               Rcpp::Named("Prec_true")=Prec_true, 
                               Rcpp::Named("G_true")=G_ret.get_graph().cast<unsigned int>(),
                               Rcpp::Named("G_complete")=Complete_ret ); 
  }
}

//' Create Groups
//' 
//' \loadmathjax This function creates a list with the groups. If possible, groups of equal size are created. The goal of this function is to fix a precise notation that will be used in all the code.
//' It is indeed recommended to use this function to create them as they need to follow a precise notation.
//' @param p integer, the dimension of the underlying graph.
//' @param n_groups number of desired groups. Has to be greater than \code{p}.
//' @return list representing the groups of the block form.
//' @export
// [[Rcpp::export]]
Rcpp::List CreateGroups(unsigned int const & p, unsigned int const & n_groups)
{
  Groups groups(n_groups, p);
  Rcpp::List L(groups.get_n_groups());
  for(unsigned int i = 0; i < n_groups; ++i){
    L[i] = groups.get_group(i);
  }
  return L;
}



// [[Rcpp::export]]
Rcpp::List GGM_sampling_c(  Eigen::MatrixXd const & data, 
                            int const & p, int const & n, int const & niter, int const & burnin, double const & thin, Rcpp::String file_name,
                            Eigen::MatrixXd D, double const & b, 
                            Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const & G0, Eigen::MatrixXd const & K0,
                            int const & MCprior = 100, int const & MCpost = 100, double const & threshold = 0.00000001,
                            Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                            Rcpp::Nullable<Rcpp::List> groups = R_NilValue, int seed = 0, double const & Gprior = 0.5, 
                            double const & sigmaG = 0.1, double const & paddrm = 0.5, bool print_info = true  )
{ 
  using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;  
  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost, threshold);
  Rcpp::String file_name_extension(file_name);
  file_name_extension += ".h5";

  if (form == "Complete")
  {
    Init<GraphType, unsigned int> init(n,p);
    init.set_init(MatRow (K0), GraphType<unsigned int> (G0)  );
    //Select the method to be used
    auto method = SelectMethod_Generic<GraphType, unsigned>(prior, algo, hy, param);
    //Crete sampler obj
    GGMsampler  Sampler(data, n, param, hy, init, method, file_name, seed, print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"GGM Sampler starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    int accepted = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    if(accepted < 0){
      std::string name(file_name);
      name += ".h5";
      Rcpp::Rcout<<"Removing file "<<name<<std::endl;
      Rcpp::Function R_file_remove("file.remove");
      R_file_remove(name);
      return Rcpp::List::create();
    }
    else{
      //Posterior Analysis
      Rcpp::Rcout<<"Created file: "<<std::string (file_name)<<".h5"<<std::endl;
      Rcpp::Rcout<<"Computing PosterionMeans ... "<<std::endl;
      VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_storeG, 0.5*p*(p+1), "Precision" );
      MatRow MeanK(MatRow::Zero(p,p));
      unsigned int pos{0};
      for(unsigned int i = 0; i < p; ++i){
        for(unsigned int j = i; j < p; ++j){
          MeanK(i,j) = MeanK_vett(pos++);
        }
      }
      auto [plinks, SampledG, TracePlot, visited] = analysis::Summary_Graph(file_name_extension, param.iter_to_storeG, p);
      //Create Rcpp::List of sampled Graphs
      std::vector< Rcpp::List > L(SampledG.size());
      int counter = 0;
      for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
        L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
      }
      return Rcpp::List::create ( Rcpp::Named("MeanK")= MeanK, 
                                  Rcpp::Named("plinks")= plinks,  
                                  Rcpp::Named("AcceptedMoves")= accepted, 
                                  Rcpp::Named("VisitedGraphs")= visited, 
                                  Rcpp::Named("TracePlot_Gsize")= TracePlot, 
                                  Rcpp::Named("SampledGraphs")= L   );
    }
    
  }
  else if(form == "Block")
  {
    if(!groups.isNotNull()){
      throw std::runtime_error("Error, group list has to be provided if Block form is selected");  
    }
    Rcpp::List gr(groups);
    std::shared_ptr<const Groups> ptr_gruppi = std::make_shared<const Groups>(gr); 
    param.ptr_groups = ptr_gruppi;
    Init<BlockGraph,  unsigned int> init(n,p, ptr_gruppi);
    init.set_init(MatRow (K0),BlockGraph<unsigned int>(G0,ptr_gruppi));
    //Select the method to be used
    auto method = SelectMethod_Generic<BlockGraph, unsigned int>(prior, algo, hy, param);
    //Crete sampler obj
    GGMsampler<BlockGraph> Sampler(data, n, param, hy, init, method, file_name, seed, print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"Block GGM Sampler starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    int accepted = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    if(accepted < 0){
      std::string name(file_name);
      name += ".h5";
      Rcpp::Rcout<<"Removing file "<<name<<std::endl;
      Rcpp::Function R_file_remove("file.remove");
      R_file_remove(name);
      return Rcpp::List::create();
    }
    else{
      //Posterior Analysis
      Rcpp::Rcout<<"Created file: "<<std::string (file_name)<<".h5"<<std::endl;
      Rcpp::Rcout<<"Computing PosterionMeans ... "<<std::endl;
      VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_storeG, 0.5*p*(p+1), "Precision" );
      MatRow MeanK(MatRow::Zero(p,p));
      unsigned int pos{0};
      for(unsigned int i = 0; i < p; ++i){
        for(unsigned int j = i; j < p; ++j){
          MeanK(i,j) = MeanK_vett(pos++);
        }
      }
      auto [plinks, SampledG, TracePlot, visited] = analysis::Summary_Graph(file_name_extension, param.iter_to_storeG, p, ptr_gruppi);
      //Create Rcpp::List of sampled Graphs
      std::vector< Rcpp::List > L(SampledG.size());
      int counter = 0;
      for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
        L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
      }
      return Rcpp::List::create ( Rcpp::Named("MeanK")= MeanK, 
                                  Rcpp::Named("plinks")= plinks,  
                                  Rcpp::Named("AcceptedMoves")= accepted, 
                                  Rcpp::Named("VisitedGraphs")= visited, 
                                  Rcpp::Named("TracePlot_Gsize")= TracePlot, 
                                  Rcpp::Named("SampledGraphs")= L   );
    }

    
  }
  else
    throw std::runtime_error("Error, the only possible form are: Complete and Block.");
}



// [[Rcpp::export]]
Rcpp::List FLM_sampling_c(Eigen::MatrixXd const & data, int const & niter, int const & burnin, double const & thin, Eigen::MatrixXd const & BaseMat,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                          Eigen::MatrixXd const & Beta0, Eigen::VectorXd const & mu0, double const & tau_eps0, Eigen::VectorXd const & tauK0, Eigen::MatrixXd const & K0,
                          double const & a_tau_eps, double const & b_tau_eps, double const & sigmamu, double const & aTauK, double const & bTauK, double const & bK, Eigen::MatrixXd const & DK,
                          Rcpp::String file_name, bool diagonal_graph = true, 
                          double const & threshold_GWish = 0.00000001, int seed = 0, bool print_info = true)
{

  const unsigned int p = BaseMat.cols();
  const unsigned int n = data.cols();
  const unsigned int r = BaseMat.rows();
  Rcpp::String file_name_extension(file_name);
  file_name_extension += ".h5";
  if(data.rows() != r)
    throw std::runtime_error("Dimension of data and BaseMat are incoherent. data has to be (n_grid_points x n), BaseMat is (n_grid_points x p)");
  
  if(diagonal_graph){
    //FLMHyperparameters hy(p);
    FLMHyperparameters hy(a_tau_eps, b_tau_eps, sigmamu, aTauK, bTauK );
    FLMParameters param(niter, burnin, thin, BaseMat);
    InitFLM init(n,p);
    init.set_init(Beta0, mu0, tau_eps0, tauK0);
    FLMsampler<GraphForm::Diagonal> Sampler(data, param, hy, init, file_name, seed, print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"FLM Sampler diagonal starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    int status = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    if(status < 0 ){
      std::string name(file_name);
      name += ".h5";
      Rcpp::Rcout<<"Removing file "<<name<<std::endl;
      Rcpp::Function R_file_remove("file.remove");
      R_file_remove(name);
      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta"), 
                                                       Rcpp::Named("MeanMu"), 
                                                       Rcpp::Named("MeanTauK"),   
                                                       Rcpp::Named("MeanTaueps") );   
      return PosteriorMeans;
    }
    else{
      //Posterior analysis
      Rcpp::Rcout<<"Created file: "<<std::string (file_name)<<".h5"<<std::endl;
      Rcpp::Rcout<<"Computing PosterionMeans ... "<<std::endl;
      MatCol MeanBeta   =  analysis::Matrix_PointwiseEstimate( file_name_extension, param.iter_to_store, p, n );
      VecCol MeanMu     =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_store, p, "Mu" );
      VecCol MeanTauK   =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_store, p, "Precision" );
      double MeanTaueps =  analysis::Scalar_PointwiseEstimate( file_name_extension, param.iter_to_store );
      
      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta")=MeanBeta, 
                                                       Rcpp::Named("MeanMu")=MeanMu, 
                                                       Rcpp::Named("MeanTauK")=MeanTauK ,   
                                                       Rcpp::Named("MeanTaueps")=MeanTaueps );   
      return PosteriorMeans;
    }
    

  }
  else{
    
    //FLMHyperparameters hy(p);
    FLMHyperparameters hy(a_tau_eps, b_tau_eps, sigmamu, bK, DK );
    FLMParameters param(niter, burnin, thin, BaseMat, threshold_GWish);
    if(G.rows() != G.cols())
      throw std::runtime_error("Inserted graph is not squared");
    if(G.rows() != p)
      throw std::runtime_error("Dimension of graph and BaseMat are incoherent. graph has to be (p x p), BaseMat is (n_grid_points x p)");
    G.cast<unsigned int>();
    GraphType<unsigned int> Graph(G);
    InitFLM init(n,p, Graph);
    init.set_init(Beta0, mu0, tau_eps0, K0 );
    FLMsampler<GraphForm::Fix> Sampler(data, param, hy, init, file_name, seed, print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"FLM Sampler fixed starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    int  status = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }

    if(status < 0){
      std::string name(file_name);
      name += ".h5";
      Rcpp::Rcout<<"Removing file "<<name<<std::endl;
      Rcpp::Function R_file_remove("file.remove");
      R_file_remove(name);
      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta"), 
                                                       Rcpp::Named("MeanMu"), 
                                                       Rcpp::Named("MeanTauK"),   
                                                       Rcpp::Named("MeanTaueps") );   
      return PosteriorMeans;
    }
    else{
      //Posterior analysis
      Rcpp::Rcout<<"Created file: "<<std::string (file_name)<<".h5"<<std::endl;
      Rcpp::Rcout<<"Computing PosterionMeans ... "<<std::endl;
      MatCol MeanBeta   =  analysis::Matrix_PointwiseEstimate( file_name_extension, param.iter_to_store, p, n );
      VecCol MeanMu     =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_store, p, "Mu" );
      VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_store, 0.5*p*(p+1), "Precision" );
      double MeanTaueps =  analysis::Scalar_PointwiseEstimate( file_name_extension, param.iter_to_store );
      
      MatRow MeanK(MatRow::Zero(p,p));
      unsigned int pos{0};
          for(unsigned int i = 0; i < p; ++i){
            for(unsigned int j = i; j < p; ++j){
              MeanK(i,j) = MeanK_vett(pos++);
            }
          }
      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta")=MeanBeta, 
                                                       Rcpp::Named("MeanMu")=MeanMu, 
                                                       Rcpp::Named("MeanK")=MeanK ,   
                                                       Rcpp::Named("MeanTaueps")=MeanTaueps );   
      
      return PosteriorMeans;
    }
    
  }
}



// [[Rcpp::export]]
Rcpp::List FGM_sampling_c(Eigen::MatrixXd const & data, int const & niter, int const & burnin, double const & thin, double const & thinG,  //data and iterations
                          Eigen::MatrixXd const & BaseMat, Rcpp::String const & file_name,  //Basemat and name of file 

                          Eigen::MatrixXd const & Beta0, Eigen::VectorXd const & mu0, double const & tau_eps0,  //initial values
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const & G0, Eigen::MatrixXd const & K0, 

                          double const & a_tau_eps, double const & b_tau_eps, double const & sigmamu,  double const & bK, //hyperparam
                          Eigen::MatrixXd const & DK, double const & sigmaG, double const & paddrm , double const & Gprior,
                          
                          int const & MCprior, int const & MCpost, double const & threshold,  //GGM_parameters
                          Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                          Rcpp::Nullable<Rcpp::List> groups = R_NilValue, 

                          int seed = 0, bool print_info = true )
{
  using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;  

  const unsigned int p = BaseMat.cols();
  const unsigned int n = data.cols();
  const unsigned int r = BaseMat.rows();
  Rcpp::String file_name_extension(file_name);
  file_name_extension += ".h5";
  if(data.rows() != r)
    throw std::runtime_error("Dimension of data and BaseMat are incoherent. data has to be (n_grid_points x n), BaseMat is (n_grid_points x p)");

 Hyperparameters hy(a_tau_eps, b_tau_eps, sigmamu, bK, DK, paddrm, sigmaG, Gprior);
 Parameters param(niter, burnin, thin, thinG, MCprior, MCpost, BaseMat, threshold);
 if (form == "Complete")
 {

   Init<GraphType, unsigned int> init(Beta0, mu0, tau_eps0, MatRow (K0), GraphType<unsigned int> (G0) );
   //Select the method to be used
   auto method = SelectMethod_Generic<GraphType, unsigned>(prior, algo, hy, param);
   //Crete sampler obj
   FGMsampler  Sampler(data, param, hy, init, method, file_name, seed, print_info);
   //Run
   if(print_info){
     Rcpp::Rcout<<"FGM Sampler starts:"<<std::endl; 
   }
   auto start = std::chrono::high_resolution_clock::now();
   int accepted = Sampler.run();
   auto stop = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> timer = stop - start;
   if(print_info){
     Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
   }
   if(accepted < 0){
      std::string name(file_name);
      name += ".h5";
      Rcpp::Rcout<<"Removing file "<<name<<std::endl;
      Rcpp::Function R_file_remove("file.remove");
      R_file_remove(name);
      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta"), 
                                                       Rcpp::Named("MeanMu"), 
                                                       Rcpp::Named("MeanK"),   
                                                       Rcpp::Named("MeanTaueps") );   

      Rcpp::List GraphAnalysis  = Rcpp::List::create ( Rcpp::Named("plinks"),  
                                                       Rcpp::Named("AcceptedMoves"), 
                                                       Rcpp::Named("VisitedGraphs"), 
                                                       Rcpp::Named("TracePlot_Gsize"), 
                                                       Rcpp::Named("SampledGraphs")  );
      return Rcpp::List::create ( Rcpp::Named("PosteriorMeans")=PosteriorMeans, Rcpp::Named("GraphAnalysis")=GraphAnalysis );
   }
   else{
      //Posterior Analysis
      Rcpp::Rcout<<"Created file: "<<std::string (file_name)<<".h5"<<std::endl;
      Rcpp::Rcout<<"Computing PosterionMeans ... "<<std::endl;
      MatCol MeanBeta   =  analysis::Matrix_PointwiseEstimate( file_name_extension, param.iter_to_store, p, n );
      VecCol MeanMu     =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_store, p, "Mu" );
      VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_storeG, 0.5*p*(p+1), "Precision" );
      double MeanTaueps =  analysis::Scalar_PointwiseEstimate( file_name_extension, param.iter_to_store );
      MatRow MeanK(MatRow::Zero(p,p));
      unsigned int pos{0};
          for(unsigned int i = 0; i < p; ++i){
            for(unsigned int j = i; j < p; ++j){
              MeanK(i,j) = MeanK_vett(pos++);
            }
          }
      //Graph analysis
      auto [plinks, SampledG, TracePlot, visited] = analysis::Summary_Graph(file_name_extension, param.iter_to_storeG, p);
      //Create Rcpp::List of sampled Graphs
      std::vector< Rcpp::List > L(SampledG.size());
      int counter = 0;
      for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
        L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
      }

      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta")=MeanBeta, 
                                                       Rcpp::Named("MeanMu")=MeanMu, 
                                                       Rcpp::Named("MeanK")=MeanK ,   
                                                       Rcpp::Named("MeanTaueps")=MeanTaueps );   

      Rcpp::List GraphAnalysis  = Rcpp::List::create ( Rcpp::Named("plinks")= plinks,  
                                                       Rcpp::Named("AcceptedMoves")= accepted, 
                                                       Rcpp::Named("VisitedGraphs")= visited, 
                                                       Rcpp::Named("TracePlot_Gsize")= TracePlot, 
                                                       Rcpp::Named("SampledGraphs")= L   );

      return Rcpp::List::create ( Rcpp::Named("PosteriorMeans")=PosteriorMeans, Rcpp::Named("GraphAnalysis")=GraphAnalysis ); 
   }
    
 }
 else if(form == "Block")
 {
   if(!groups.isNotNull()){
     throw std::runtime_error("Error, group list has to be provided if Block form is selected");  
   }
   Rcpp::List gr(groups);
   std::shared_ptr<const Groups> ptr_gruppi = std::make_shared<const Groups>(gr); 
   param.ptr_groups = ptr_gruppi;
   Init<BlockGraph, unsigned int> init(Beta0, mu0, tau_eps0, MatRow (K0), BlockGraph<unsigned int> (G0, ptr_gruppi) );
   //Select the method to be used
   auto method = SelectMethod_Generic<BlockGraph, unsigned int>(prior, algo, hy, param);
   //Crete sampler obj
   FGMsampler<BlockGraph, unsigned int> Sampler(data, param, hy, init, method, file_name, seed, print_info);
   //Run
   if(print_info){
     Rcpp::Rcout<<"FGM Sampler starts:"<<std::endl; 
   }
   auto start = std::chrono::high_resolution_clock::now();
   int accepted = Sampler.run();
   auto stop = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> timer = stop - start;
   if(print_info){
     Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
   }
   if(accepted < 0){
      std::string name(file_name);
      name += ".h5";
      Rcpp::Rcout<<"Removing file "<<name<<std::endl;
      Rcpp::Function R_file_remove("file.remove");
      R_file_remove(name);
      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta"), 
                                                       Rcpp::Named("MeanMu"), 
                                                       Rcpp::Named("MeanK"),   
                                                       Rcpp::Named("MeanTaueps") );   

      Rcpp::List GraphAnalysis  = Rcpp::List::create ( Rcpp::Named("plinks"),  
                                                       Rcpp::Named("AcceptedMoves"), 
                                                       Rcpp::Named("VisitedGraphs"), 
                                                       Rcpp::Named("TracePlot_Gsize"), 
                                                       Rcpp::Named("SampledGraphs")  );
      return Rcpp::List::create ( Rcpp::Named("PosteriorMeans")=PosteriorMeans, Rcpp::Named("GraphAnalysis")=GraphAnalysis );
   }
   else{
      //Posterior Analysis
      Rcpp::Rcout<<"Created file: "<<std::string (file_name)<<".h5"<<std::endl;
      Rcpp::Rcout<<"Computing PosterionMeans ... "<<std::endl;
      MatCol MeanBeta   =  analysis::Matrix_PointwiseEstimate( file_name_extension, param.iter_to_store, p, n );
      VecCol MeanMu     =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_store, p, "Mu" );
      VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate( file_name_extension, param.iter_to_storeG, 0.5*p*(p+1), "Precision" );
      double MeanTaueps =  analysis::Scalar_PointwiseEstimate( file_name_extension, param.iter_to_store );
      MatRow MeanK(MatRow::Zero(p,p));
      unsigned int pos{0};
          for(unsigned int i = 0; i < p; ++i){
            for(unsigned int j = i; j < p; ++j){
              MeanK(i,j) = MeanK_vett(pos++);
            }
          }
      //Graph analysis
      auto [plinks, SampledG, TracePlot, visited] = analysis::Summary_Graph(file_name_extension, param.iter_to_storeG, p, ptr_gruppi);
      //Create Rcpp::List of sampled Graphs
      std::vector< Rcpp::List > L(SampledG.size());
      int counter = 0;
      for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
        L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
      }

      Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta")=MeanBeta, 
                                                       Rcpp::Named("MeanMu")=MeanMu, 
                                                       Rcpp::Named("MeanK")=MeanK ,   
                                                       Rcpp::Named("MeanTaueps")=MeanTaueps );   

      Rcpp::List GraphAnalysis  = Rcpp::List::create ( Rcpp::Named("plinks")= plinks,  
                                                       Rcpp::Named("AcceptedMoves")= accepted, 
                                                       Rcpp::Named("VisitedGraphs")= visited, 
                                                       Rcpp::Named("TracePlot_Gsize")= TracePlot, 
                                                       Rcpp::Named("SampledGraphs")= L   );

      return Rcpp::List::create ( Rcpp::Named("PosteriorMeans")=PosteriorMeans, Rcpp::Named("GraphAnalysis")=GraphAnalysis ); 
   }
   
 }
 else
   throw std::runtime_error("Error, the only possible forms are: Complete and Block.");
}


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*
* Ho cambiato il modo di aprire e leggere i file. Niente di che, ma ora non devo piu passare manualmente i parametri ma li legge direttamente la file. Più facile controllare che non vengano inseriti
* valori sbagliati. Il problema è che queste nuove funzioni non possono aprire i vecchi file. Quindi tengo salvate anche le vecchie versioni. Non sono da usare se non con i file in cui non salvo 
* quale sampler è stato usato.
*/

// Read information from file
//
// \loadmathjax Read from \code{file_name} some information that are needed to extract data from it. 
// @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
// @return It returns a list containing \code{p}, the dimension of the graph, \code{n} the number of observed data, \code{stored_iter} the number of saved iterations for the regression parameters,
// \code{stored_iterG} the number of saved iterations for the graphical related quantities, i.e the graph and the precision matrix.
// [[Rcpp::export]]
Rcpp::List Read_InfoFile_old( Rcpp::String const & file_name )
{
  std::vector< unsigned int > info =  HDF5conversion::GetInfo_old(file_name);
  return  Rcpp::List::create ( Rcpp::Named("p") = info[0], 
                               Rcpp::Named("n") = info[1], 
                               Rcpp::Named("stored_iter") = info[2],   
                               Rcpp::Named("stored_iterG") = info[3]
                             );
}



// Compute quantiles of sampled values
//
// \loadmathjax This function reads the sampled values saved in a binary file and computes the quantiles of the desired level.
// @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
// @param p integer, the dimension of the graph or the number of basis functions used. It depends on the type of output that is contained in \code{file_name}. 
// It has the same meaning of \code{p} parameter of \code{\link{GGM_sampling}}, \code{\link{FLM_sampling}} or \code{\link{FGM_sampling}}.
// @param n integer, the number of observed data.
// @param stored_iterG integer, the number of saved iterations for the graphical related quantities, i.e the graph and the precision matrix. Required only if \code{Precision} parameter is \code{TRUE}.
// @param stored_iter integer, the number of saved iterations for the regression parameters, i.e \mjseqn{\beta}s, \mjseqn{\mu} and \mjseqn{\tau_{\epsilon}}. 
// Required if at least one of \code{Beta}, \code{Mu},  \code{TauEps} parameter is \code{TRUE}.
// @param Beta boolean, set \code{TRUE} to compute the quantiles for all \code{p*n} \mjseqn{\beta} coefficients. It may require long time.
// @param Mu boolean, set \code{TRUE} to compute the quantiles for all \mjseqn{p} parameters. 
// @param TauEps boolean, set \code{TRUE} to compute the quantiles of \mjseqn{\tau_{\epsilon}} parameter.
// @param Precision boolean, set \code{TRUE} to compute the quantiles for all the elements of the precision matrix. Some care is requested. 
// If the file represents the output of \code{GGM}, \code{FGM} of \code{FLM} with fixed graph, the sampled precision matrices are of size \mjseqn{p \times p}. In that case set \code{prec_elem} parameter
// equal to \mjseqn{\frac{p(p+1)}{2}}. Instead, for outputs coming from \code{FLM} sampler with diagonal graph, only the diagonal of the precision matrix is saved. If so, set \code{prec_ele} eqaul to \mjseqn{p}.
// @param prec_elem integer, set equal to \mjseqn{p} if \code{file_name} represents the output of \code{FLM} sampler with diagonal graph. Set \mjseqn{\frac{p(p+1)}{2}} otherwise.
// @param lower_qtl the level of the first desired quantile.
// @param upper_qtl the level of the second desired quantile.
//
// @return It returns a list containig the upper and lower quantiles of the requested quantities.
// [[Rcpp::export]]
Rcpp::List Compute_Quantiles_old( Rcpp::String const & file_name, unsigned int const & p, unsigned int const & n, unsigned int const & stored_iterG = 0, unsigned int const & stored_iter = 0, 
                              bool Beta = false, bool Mu = false, bool TauEps = false, bool Precision = false, unsigned int const & prec_elem = 0,
                              double const & lower_qtl = 0.05, double const & upper_qtl = 0.95  )
{
  
  Rcpp::List Quantiles = Rcpp::List::create( Rcpp::Named("Beta"),      
                                             Rcpp::Named("Mu"),        
                                             Rcpp::Named("TauEps"),    
                                             Rcpp::Named("Precision") 
                                           );
  if(Precision){
    if(stored_iterG <= 0)
      throw std::runtime_error("stored_iterG parameter has to be positive");
    if(prec_elem <= 0)
      throw std::runtime_error("Need to specify the number of elements of the precision matrix in prec_elem parameter");
    if(prec_elem != p && prec_elem != 0.5*p*(p+1))
      throw std::runtime_error("prec_elem can only be p or 0.5*p*(p+1)");
    Rcpp::Rcout<<"Compute Precision quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Vector_ComputeQuantiles( file_name, stored_iterG, prec_elem, "Precision", lower_qtl, upper_qtl );
    Quantiles["Precision"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  if(Beta){
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    Rcpp::Rcout<<"Compute Beta quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Matrix_ComputeQuantiles( file_name, stored_iter, p, n, lower_qtl, upper_qtl );
    Quantiles["Beta"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  if(Mu){
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    Rcpp::Rcout<<"Compute Mu quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Vector_ComputeQuantiles( file_name, stored_iter, p, "Mu", lower_qtl, upper_qtl );
    Quantiles["Mu"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  if(TauEps){
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    Rcpp::Rcout<<"Compute TauEps quantiles..."<<std::endl;
    auto [Lower, Upper] =  analysis::Scalar_ComputeQuantiles( file_name, stored_iter, lower_qtl, upper_qtl );
    Quantiles["TauEps"] = Rcpp::List::create(Rcpp::Named("Lower")=Lower, Rcpp::Named("Upper")=Upper);
  }
  return Quantiles;
}


// Read chain from file
//
// \loadmathjax This function read from a binary file the sampled chain for the \code{index1}-th component of variable specified in \code{variable}. The chain is then saved in memory to make it available for further analysis.
// Both \code{index1} and \code{index2} start counting from 1, the first element is obtained by settin \code{index1} equal to 1, not 0.
// Only \code{"Beta"} coefficients require the usage of \code{index2}. This function allows to extract only one chain at the time. The idea of writing the sampled values on a file is indeed to
// avoid to fill the memory. This function has to carefully used.
// @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
// @param variable string, the name of the dataset to be read from the file. Only possibilities are \code{"Beta"}, \code{"Mu"}, \code{"Precision"} and \code{"TauEps"}.
// @param stored_iter integer, the number of saved iterations. It can be read from \code{\link{Read_InfoFile}}.
// @param p integer, the dimension of the graph or the number of basis functions used. It depends on the type of output that is contained in \code{file_name}. 
// It has the same meaning of \code{p} parameter of \code{\link{GGM_sampling}}, \code{\link{FLM_sampling}} or \code{\link{FGM_sampling}}.
// @param n integer, the number of observed data.
// @param index1 integer, the index of the element whose chain has to read from the file. The first elements corresponds to \code{index1} equal to 1. 
// If \code{variable} is equal to \code{"Precision"} some care is required. If the file represents the output of \code{GGM}, \code{FGM} of \code{FLM} with fixed graph, the sampled precision matrices 
// are of size \mjseqn{p \times p} stored row by row. This means that the second diagonal elements corresponds to \code{index1} equal to \mjseqn{p+1}. 
// Moreover, in this case set \code{prec_elem} parameter equal to \mjseqn{\frac{p(p+1)}{2}}. 
// Instead, for outputs coming from \code{FLM} sampler with diagonal graph, only the diagonal of the precision matrix is saved. If so, \code{index1} ranges from 1 up to \mjseqn{p}. Moreover, set \code{prec_ele} eqaul to \mjseqn{p}.
// If \code{variable} is equal to \code{"Beta"}, this index ranges for 1 up to \mjseqn{p}, it represents the spine coefficinet.
// @param index2 integer, to be used only if \code{variable} is equal to \code{"Beta"}. It ranges from 1 up to \mjseqn{n}. In this case, the chain for the spline_index-th coefficients of the curve_index-th curve is read.
// @param prec_elem integer, set equal to \mjseqn{p} if \code{file_name} represents the output of \code{FLM} sampler with diagonal graph. Set \mjseqn{\frac{p(p+1)}{2}} otherwise.
//
// @return It returns a numeric vector all the sampled values of the required element.
// [[Rcpp::export]]
Eigen::VectorXd Extract_Chain_old( Rcpp::String const & file_name, Rcpp::String const & variable, unsigned int const & stored_iter, unsigned int const & p, 
                               unsigned int const & n = 0, unsigned int  index1 = 1, unsigned int index2 = 1, unsigned int const & prec_elem = 0  )
{ 

  if(variable != "Beta" && variable != "Mu" && variable != "Precision" && variable != "TauEps")
    throw std::runtime_error("Error, only possible values for variable are Beta, Precision, Mu, TauEps");

  if(index1 <= 0 || index2 <= 0)
    throw std::runtime_error("index1 and index2 parameters start counting from 1, not from 0. The first element corresponds to index 1, not 0. The inserted values has to be strictly positive");

  index1 = index1 - 1;
  index2 = index2 - 1;

  HDF5conversion::FileType file;
  HDF5conversion::DatasetType dataset_rd;
  std::string file_name_stl{file_name};
  //Open file
  file=H5Fopen(file_name_stl.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
  if(file < 0)
    throw std::runtime_error("Error, can not open the file. Probably it was not closed correctly");

  std::vector<double> chain;
  if(variable == "Beta"){
    if(n <= 0)
      throw std::runtime_error("The number of observed data n has to be provided to extract the chain for Beta");
    //Open datasets
    dataset_rd  = H5Dopen(file, "/Beta", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open dataset for Beta ");
    chain = HDF5conversion::GetChain_from_Matrix(dataset_rd, index1, index2, stored_iter, n);
    H5Dclose(dataset_rd);
  }
  else if(variable == "Mu"){
    //Open datasets
    dataset_rd  = H5Dopen(file, "/Mu", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open dataset for MU");
    chain = HDF5conversion::GetChain_from_Vector(dataset_rd, index1, stored_iter, p);
    H5Dclose(dataset_rd);
  }
  else if(variable == "Precision"){
    if(prec_elem <= 0)
      throw std::runtime_error("Need to specify the number of elements of the precision matrix in prec_elem parameter");
    //Open datasets
    dataset_rd  = H5Dopen(file, "/Precision", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open dataset for Precision");
    chain = HDF5conversion::GetChain_from_Vector(dataset_rd, index1, stored_iter, prec_elem);
    H5Dclose(dataset_rd);
  }
  else if(variable == "TauEps"){
    //Open datasets
    dataset_rd  = H5Dopen(file, "/TauEps", H5P_DEFAULT);
    if(dataset_rd < 0)
      throw std::runtime_error("Error, can not open the dataset for TauEps");
    chain.resize(stored_iter);
    double * buffer = chain.data();
    HDF5conversion::StatusType status = H5Dread(dataset_rd, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
    if(status < 0)
      throw std::runtime_error("Error, can not read the chain for TauEps");
    H5Dclose(dataset_rd);
  }
  else{
    throw std::runtime_error("Error, only possible values for variable are Beta, Precision, Mu, TauEps");
  }

  H5Fclose(file);

  Eigen::Map< Eigen::VectorXd > result(&(chain[0]), chain.size());
  return result;
}

// Compute Posterior means of sampled values
//
// \loadmathjax This function reads the sampled values saved in a binary file and computes the mean of the requested quantities.
// @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
// @param p integer, the dimension of the graph or the number of basis functions used. It depends on the type of output that is contained in \code{file_name}. 
// It has the same meaning of \code{p} parameter of \code{\link{GGM_sampling}}, \code{\link{FLM_sampling}} or \code{\link{FGM_sampling}}.
// @param n integer, the number of observed data.
// @param stored_iterG integer, the number of saved iterations for the graphical related quantities, i.e the graph and the precision matrix. Required only if \code{Precision} parameter is \code{TRUE}.
// @param stored_iter integer, the number of saved iterations for the regression parameters, i.e \mjseqn{\beta}s, \mjseqn{\mu} and \mjseqn{\tau_{\epsilon}}.
// Required if at least one of \code{Beta}, \code{Mu},  \code{TauEps} parameter is \code{TRUE}.
// @param Beta boolean, set \code{TRUE} to compute the mean for all \code{p*n} \mjseqn{\beta} coefficients. It may require long time.
// @param Mu boolean, set \code{TRUE} to compute the mean for all \mjseqn{p} parameters. 
// @param TauEps boolean, set \code{TRUE} to compute the mean of \mjseqn{\tau_{\epsilon}} parameter.
// @param Precision boolean, set \code{TRUE} to compute the mean for all the elements of the precision matrix. Some care is requested. 
// If the file represents the output of \code{GGM}, \code{FGM} of \code{FLM} with fixed graph, the sampled precision matrices are of size \mjseqn{p \times p}. In that case set \code{prec_elem} parameter
// equal to \mjseqn{\frac{p(p+1)}{2}}. Instead, for outputs coming from \code{FLM} sampler with diagonal graph, only the diagonal of the precision matrix is saved. If so, set \code{prec_ele} eqaul to \mjseqn{p}.
// @param prec_elem integer, set equal to \mjseqn{p} if \code{file_name} represents the output of \code{FLM} sampler with diagonal graph. Set \mjseqn{\frac{p(p+1)}{2}} otherwise.
//
// @return It returns a list containig the mean of the requested quantities.
// [[Rcpp::export]]
Rcpp::List Compute_PosteriorMeans_old( Rcpp::String const & file_name, unsigned int const & p, unsigned int const & n, unsigned int const & stored_iterG = 0, unsigned int const & stored_iter = 0, 
                                       bool Beta = false, bool Mu = false, bool TauEps = false, bool Precision = false, unsigned int const & prec_elem = 0)
{

  Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta"), 
                                                   Rcpp::Named("MeanMu"), 
                                                   Rcpp::Named("MeanK"),   
                                                   Rcpp::Named("MeanTaueps") );  
  if(Precision){
    if(stored_iterG <= 0)
      throw std::runtime_error("stored_iterG parameter has to be positive");
    if(prec_elem <= 0)
      throw std::runtime_error("Need to specify the number of elements of the precision matrix in prec_elem parameter");
    MatRow MeanK(MatRow::Zero(p,p));  
    VecCol MeanK_vett =  analysis::Vector_PointwiseEstimate( file_name, stored_iterG, prec_elem, "Precision" );
    unsigned int pos{0};
    for(unsigned int i = 0; i < p; ++i){
      for(unsigned int j = i; j < p; ++j){
        MeanK(i,j) = MeanK_vett(pos++);
      }
    }
    PosteriorMeans["MeanK"] = MeanK;
  }
  if(Beta){
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    MatCol MeanBeta = analysis::Matrix_PointwiseEstimate( file_name, stored_iter, p, n );
    PosteriorMeans["MeanBeta"] = MeanBeta;
  }
  if(Mu){
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    VecCol MeanMu = analysis::Vector_PointwiseEstimate( file_name, stored_iter, p, "Mu" );
    PosteriorMeans["MeanMu"] = MeanMu;
  }
  if(TauEps){
    if(stored_iter <= 0)
      throw std::runtime_error("stored_iter parameter has to be positive");
    double MeanTaueps =  analysis::Scalar_PointwiseEstimate( file_name, stored_iter );
    PosteriorMeans["MeanTaueps"] = MeanTaueps;
  }
  return PosteriorMeans;
}


// Read the sampled Graph saved on file
//
// \loadmathjax This function reads the sampled graphs that are saved on a binary file and performs a summary of all visited graphs.
// @param file_name, string with the name of the file to be open. It has to include the extension, usually \code{.h5}.
// @param stored_iter integer, the number of saved iterations. It can be read from \code{\link{Read_InfoFile}}.
// @param p integer, the dimension of the graph or the number of basis functions used. It depends on the type of output that is contained in \code{file_name}. 
// It has the same meaning of \code{p} parameter of \code{\link{GGM_sampling}}, \code{\link{FLM_sampling}} or \code{\link{FGM_sampling}}.
// @param groups List representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
// i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//
// @return It returns a list composed of: \code{plinks} that contains the posterior probability of inclusion of each possible link. \code{AcceptedMoves} contains the number of
// Metropolis-Hastings moves that were accepted in the sampling, \code{VisitedGraphs} the number of graph that were visited at least once, \code{TracePlot_Gsize} is a vector 
// such that each element is equal to the size of the visited graph in that particular iteration and finally \code{SampledGraphs} is a list containing all the visited graphs and their absolute frequence of visit.
// To save memory, the graphs are represented only by the upper triangular part, stored row-wise. 
// [[Rcpp::export]]
Rcpp::List Summary_Graph_old(Rcpp::String const & file_name, unsigned int const & stored_iterG, unsigned int const & p, Rcpp::Nullable<Rcpp::List> groups = R_NilValue)
{

  std::shared_ptr<const Groups> ptr_gruppi = nullptr;
  if (groups.isNotNull()){ //Assume it is a BlockGraph
    Rcpp::List gr(groups);
    ptr_gruppi = std::make_shared<const Groups>(gr); 
    const Groups& gruppi_test = *ptr_gruppi;
  }
  auto [plinks, SampledG, TracePlot, visited] = analysis::Summary_Graph(file_name, stored_iterG, p, ptr_gruppi);
  //Create Rcpp::List of sampled Graphs
  std::vector< Rcpp::List > L(SampledG.size());
  int counter = 0;
  for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
    L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
  }
  return Rcpp::List::create ( Rcpp::Named("plinks")= plinks,  
                              Rcpp::Named("VisitedGraphs")= visited, 
                              Rcpp::Named("TracePlot_Gsize")= TracePlot, 
                              Rcpp::Named("SampledGraphs")= L   );
}

#endif





