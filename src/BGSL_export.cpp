#ifndef __BGSLEXPORT_HPP__
#define __BGSLEXPORT_HPP__

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#define STRICT_R_HEADERS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "include_graphs.h"
#include "include_headers.h"
#include "include_helpers.h"
#include "GraphPrior.h"
#include "GibbsSamplerBase.h"
#include "GGMFactory.h"
#include "PosteriorAnalysis.h"
#include "GibbsSamplerDebug.h"
#include "GGMsampler.h"
#include "FLMsampler.h"



using MatRow        = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatCol        = Eigen::MatrixXd;
using MatUnsRow     = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatUnsCol     = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using VecCol        = Eigen::VectorXd;
using VecRow        = Eigen::RowVectorXd;

//' Primo esempio di export
//'
//' @param Eigen row matrix.
//' @export
// [[Rcpp::export]]
void test_null(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
               Rcpp::Nullable<Rcpp::List> l = R_NilValue )
{
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> G2(G);
  Rcpp::Rcout<<G2<<std::endl;
  if (l.isNotNull()){
    Rcpp::List L(l);                          // casting to underlying type List
    Rcpp::Rcout << "List is not NULL." << std::endl;
    Groups Gr(L);
    std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L);
    Rcpp::Rcout<<"Gruppi:"<<std::endl<<Gr<<std::endl;
    std::vector<int> v1 = L[0];
    Rcpp::Rcout<<L.size()<<std::endl;
  }
  else{
    Rcpp::Rcout << "List is NULL." << std::endl;
  }
  std::vector< std::pair<int, bool> > v(10);
  v[0] = std::make_pair(10,true);
  v[1] = std::make_pair(9,false);
  v[2] = std::make_pair(8,true);
  std::vector<bool> vett{true, true, false};
  std::vector< Rcpp::List > L(10);
  for(int i = 0; i < 2; ++i){
    L[i] = Rcpp::List::create(Rcpp::Named("key")=vett, Rcpp::Named("value")=v[i].second);
  }
}


//' A direct sampler for GWishart distributed random variables.  
//'
//' This function draws a random matrices, distributed according to the GWishart distribution with shape parameter \code{b} and inverse scale \code{D}, 
//' with respect to the graph structure \code{G}. METTERE LA FORMULA DELLA DISTRIBUZIONE 
//' It implements the algorithm described by METTERE CITAZIONI LENKOSKI. It works with both decomposable and non decomposable graphs. 
//' In particular it is possible to provide a graph in block form. 
//' @param G Eigen Matrix of unsigned int stored columnwise. If a standard R matrix is provided, it is automaticaly converted. The lower part 
//' is not used, the elements is taken in consideration only if the graph is in block form, i.e if groups is non null.
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param norm String to choose the matrix norm with respect to whom convergence takes place. The available choices are \code{"Mean"}, \code{"Inf"}, \code{"One"} and \code{"Squared"}. 
//' \code{"Mean"} is the default value and it is also used when a wrong input is provided.
//' @param groups a Rcpp list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//' @param max_iter unsigned int, the maximum number of iteration.
//' @param threshold_check double, the accurancy for checking if the sampled matrix respects the structure of the graph.
//' @param threshold_conv double, stop algorithm if the difference between two subsequent iterations is less than \code{threshold_conv}.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return List containing the sampled matrix as an Eigen RowMajor matrix, a bool that states if the convergence was reached or not and finally an int with the number of performed iterations.
//' If the graph is empty or complete, no iterations are performed.
//' @export
// [[Rcpp::export]]
Rcpp::List rGwish_old(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                  double const & b, Eigen::MatrixXd const & D,
                  Rcpp::String norm = "Mean", Rcpp::Nullable<Rcpp::List> groups = R_NilValue, 
                  unsigned int const & max_iter = 500, long double const & threshold_check = 1e-5, long double const & threshold_conv = 1e-8, int seed = 0)
{

  if (groups.isNotNull()){ //Assume it is a BlockGraph
    
    Rcpp::List L(groups); // casting to underlying type List
    //Rcpp::Rcout << "List is not NULL." << std::endl;
    std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
    BlockGraph<unsigned int> Graph(G, ptr_gr);

    if(norm == "Mean"){
      //Rcpp::Rcout<<"Mean norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::MeanNorm>(Graph.completeview(), b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)  );
    }
    if(norm == "Inf"){
      //Rcpp::Rcout<<"Inf norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::NormInf>(Graph.completeview(), b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)  );
    }
    if(norm == "One"){
      //Rcpp::Rcout<<"One norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::Norm1>(Graph.completeview(), b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)   );
    }
    if(norm == "Squared"){
      //Rcpp::Rcout<<"Squared norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::NormSq>(Graph.completeview(), b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)   );
    }
    else{
      Rcpp::Rcout<<"The only available norms are Mean, Inf, One and Squared. Run with default type that is Mean"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::MeanNorm>(Graph.completeview(), b, D, max_iter, threshold_conv,(unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold_check)    );
    } 
  }
  else{ //Assume it is a Complete Graph
    //Rcpp::Rcout << "List is NULL." << std::endl;
    GraphType<unsigned int> Graph(G);
   if(norm == "Mean"){
      //Rcpp::Rcout<<"Mean norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::MeanNorm>(Graph, b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create ( Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold_check)  );
    }
     if(norm == "Inf"){
      //Rcpp::Rcout<<"Inf norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::NormInf>(Graph, b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold_check)  );
    }
    if(norm == "One"){
      //Rcpp::Rcout<<"Norm L1 selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::Norm1>(Graph, b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold_check)  );
    }
    if(norm == "Squared"){
      //Rcpp::Rcout<<"Squared norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::NormSq>(Graph, b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold_check)  );
    }
    else{
      Rcpp::Rcout<<"The only available norms are Mean, Inf, One and Squared. Run with default type that is Mean"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose(Graph, b, D, max_iter, threshold_conv, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold_check)  );
    }
  }
}

//' A direct sampler for GWishart distributed random variables.  
//'
//' This function draws a random matrices, distributed according to the GWishart distribution with shape parameter \code{b} and inverse scale \code{D}, 
//' with respect to the graph structure \code{G}. METTERE LA FORMULA DELLA DISTRIBUZIONE 
//' It implements the algorithm described by METTERE CITAZIONI LENKOSKI. It works with both decomposable and non decomposable graphs. 
//' In particular it is possible to provide a graph in block form. 
//' @param G Matrix of int stored columnwise. If a standard R matrix is provided, it is automaticaly converted. The lower part 
//' is not used, the elements is taken in consideration only if the graph is in block form, i.e if groups is non null.
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Matrix of double stored columnwise. It represents the Scale or Inverse Scale parameter for a GWishart distribution. It is also possibile to provide a lower or upper triangular matrix representing the Cholesky decomposition of Inverse Scale matrix.
//' @param norm String to choose the matrix norm with respect to whom convergence takes place. The available choices are \code{"Mean"}, \code{"Inf"}, \code{"One"} and \code{"Squared"}. 
//' \code{"Mean"} is the default value and it is also used when a wrong input is provided.
//' @param form String, states what type of parameter is represented by \code{D}. Possible values are \code{"Scale"}, \code{"InvScale"}, \code{"CholLower_InvScale"}, \code{"CholUpper_InvScale"}.
//' Usually GWishart distributions are parametrized with respect to Inverse Scale matrix. However the first step of the sampling requires the Scale matrix parameter or, even better, its Cholesky decomposition. 
//' @param groups List representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave NULL if the graph is not in block form.
//' @param check_structure bool, if \code{TRUE} it is checked if the sampled matrix actually respects the structure of the graph.
//' @param max_iter unsigned int, the maximum number of iteration.
//' @param threshold_check double, the accurancy for checking if the sampled matrix respects the structure of the graph.
//' @param threshold_conv double,  the threshold value for the convergence of sampling algorithm from GWishart. Algorithm stops if the difference between two subsequent iterations is less than \code{threshold_conv}.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return Rcpp::List containing the sampled matrix as an Eigen RowMajor matrix, a bool that states if the convergence was reached or not and finally an int with the number of performed iterations.
//' If the graph is empty or complete, no iterations are performed. If check_structure is \code{TRUE}, then the result of that check is also returned.
//' @export
// [[Rcpp::export]]
Rcpp::List rGwish(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                  double const & b, Eigen::MatrixXd & D,Rcpp::String norm = "Mean", Rcpp::String form = "InvScale", 
                  Rcpp::Nullable<Rcpp::List> groups = R_NilValue, bool check_structure = false,
                  unsigned int const & max_iter = 500, long double const & threshold_check = 1e-5, long double const & threshold_conv = 1e-8, int seed = 0)
{

  if (groups.isNotNull()){ //Assume it is a BlockGraph   
    Rcpp::List L(groups); // casting to underlying type List
    //Rcpp::Rcout << "List is not NULL." << std::endl;
    std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
    BlockGraph<unsigned int> Graph(G, ptr_gr);
    auto rgwish_fun = utils::build_rgwish_function<CompleteView, unsigned int>(form, norm);
    auto [Mat, converged, iter] = rgwish_fun(Graph.completeview(), b, D, threshold_conv, (unsigned int)seed, max_iter);
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
    //Rcpp::Rcout << "List is NULL." << std::endl;
    GraphType<unsigned int> Graph(G);
    auto rgwish_fun = utils::build_rgwish_function<GraphType, unsigned int>(form, norm);
    auto [Mat, converged, iter] = rgwish_fun(Graph.completeview(), b, D, threshold_conv, (unsigned int)seed, max_iter);
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


//' log of GWishart normalizing constant
//'
//' This function computes the logarithm of the normalizing constant of GWishart distribution. It implements the Monte Carlo method, developed by Atay-Kayis and Massam (2005).
//' METTERE LA FORMULA
//' It also works for non decomposable graphs, actually the exact formula for the decomposable case is not yet implemented. 
//' In particular it is possible to provide a graph in block form. 
//' @param G Eigen Matrix of unsigned int stored columnwise. If a standard R matrix is provided, it is automaticaly converted. The lower part 
//' is not used, the elements is taken in consideration only if the graph is in block form, i.e if \code{groups} is non null.
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param MCiteration unsigned int, the number of iteration for the MonteCarlo approximation. 
//' @param groups a Rcpp list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return long double, the logarithm of the normalizing constant of GWishart distribution.
//' @export
// [[Rcpp::export]]
long double log_Gconstant(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                          double const & b, Eigen::MatrixXd const & D,  unsigned int const & MCiteration = 100, 
                          Rcpp::Nullable<Rcpp::List> groups = R_NilValue, int seed = 0)
{
    if (groups.isNotNull()){ //Assume it is in block form with respect to the groups given in groups
      
      Rcpp::List L(groups); // casting to underlying type List
      //Rcpp::Rcout << "List is not NULL." << std::endl;
      std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
      BlockGraph<unsigned int> Graph(G, ptr_gr);
      return utils::log_normalizing_constat(Graph.completeview(), b, D, MCiteration, seed);
    }
    else{ //Assume it is a BlockGraph
      //Rcpp::Rcout << "List is NULL." << std::endl;
      GraphType<unsigned int> Graph(G);
      return utils::log_normalizing_constat(Graph.completeview(), b, D, MCiteration, seed);
    }
}

//' Second version of log of GWishart normalizing constant
//'
//' This function computes the logarithm of the normalizing constant of GWishart distribution. It implements the Monte Carlo method, developed by Atay-Kayis and Massam (2005).
//' METTERE LA FORMULA
//' It also works for non decomposable graphs, actually the exact formula for the decomposable case is not yet implemented. 
//' In particular it is possible to provide a graph in block form. 
//' @param G Eigen Matrix of unsigned int stored columnwise. If a standard R matrix is provided, it is automaticaly converted. The lower part 
//' is not used, the elements is taken in consideration only if the graph is in block form, i.e if groups is non null.
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param MCiteration unsigned int, the number of iteration for the MonteCarlo approximation. 
//' @param groups a Rcpp list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave NULL if the graph is not in block form.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return long double, the logarithm of the normalizing constant of GWishart distribution.
//' @export
// [[Rcpp::export]]
long double log_Gconstant2(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                          double const & b, Eigen::MatrixXd const & D,  unsigned int const & MCiteration = 100, 
                          Rcpp::Nullable<Rcpp::List> groups = R_NilValue, int seed = 0)
{
    if (groups.isNotNull()){ //Assume it is in block form with respect to the groups given in groups
      
      Rcpp::List L(groups); // casting to underlying type List
      //Rcpp::Rcout << "List is not NULL." << std::endl;
      std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
      BlockGraph<unsigned int> Graph(G, ptr_gr);
      return utils::log_normalizing_constat2(Graph.completeview(), b, D, MCiteration, seed);
    }
    else{ //Assume it is a BlockGraph
      //Rcpp::Rcout << "List is NULL." << std::endl;
      GraphType<unsigned int> Graph(G);
      return utils::log_normalizing_constat2(Graph, b, D, MCiteration, seed);
    }
}


//' Testing all the Gaussian Graphical Models samplers with simulated data
//'
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param MCiteration unsigned int, the number of iteration for the MonteCarlo approximation. 
//' @export
// [[Rcpp::export]]
Rcpp::List GGM_sim_sampling( int const & p, int const & n, int const & niter, int const & burnin, double const & thin, Eigen::MatrixXd const & D, 
                            double const & b = 3.0, int const & MCprior = 100, int const & MCpost = 100, double const & threshold = 1e-8,
                            Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                            int const & n_groups = 0, int seed = 0, double sparsity = 0.5, double const & Gprior = 0.5, double const & sigmaG = 0.1, 
                            double const & paddrm = 0.5, bool print_info = true)
{
  using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost, threshold);
  if (form == "Complete")
  {
    Init<GraphType, unsigned int> init(n,p);
    //Select the method to be used
    auto method = SelectMethod_Generic<GraphType, unsigned int>(prior, algo, hy, param);
    //Simulate data
    auto [data, Prec_true, G_true] = utils::SimulateDataGGM_Complete(p,n,seed);
    GraphType<bool> G_true_mat(G_true);
    //Crete sampler obj
    GGMsampler  Sampler(data, n, param, hy, init, method,0,print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"GGM Sampler starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    //Posterior Analysis
    auto[MeanK_vec, plinks] = analysis::PointwiseEstimate<decltype(SampledG)>(std::tie(SampledK, SampledG),param.iter_to_storeG);
    MatRow MeanK(MatRow::Zero(p,p));
    unsigned int pos{0};
    for(unsigned int i = 0; i < p; ++i){
      for(unsigned int j = i; j < p; ++j){
        MeanK(i,j) = MeanK_vec(pos++);
      }
    }
    //Create Rcpp::List of sampled Graphs
    std::vector< Rcpp::List > L(SampledG.size());
    int counter = 0;
    for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
      L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
    }
    return Rcpp::List::create ( Rcpp::Named("data") = data,
                                Rcpp::Named("TrueGraph") = G_true_mat.get_graph(),
                                Rcpp::Named("TruePrecision") = Prec_true,
                                Rcpp::Named("SampledG") = L,
                                Rcpp::Named("MeanK")= MeanK, 
                                Rcpp::Named("plinks")= plinks, 
                                Rcpp::Named("AcceptedMoves")=accepted, 
                                Rcpp::Named("VisitedGraphs")=visited  );
  }
  else if(form == "Block")
  {
    if(n_groups <= 1 || n_groups > p)
      throw std::runtime_error("Error, invalid number of groups inserted");  
    std::shared_ptr<const Groups> ptr_gruppi(std::make_shared<const Groups>(n_groups,p));
    param.ptr_groups = ptr_gruppi;
    Init<BlockGraph,  unsigned int> init(n,p, ptr_gruppi);
    //Select the method to be used
    auto method = SelectMethod_Generic<BlockGraph, unsigned int>(prior, algo, hy, param);
    //Simulate data
    auto [data, Prec_true, G_true] = utils::SimulateDataGGM_Block(p,n,ptr_gruppi,seed);
    BlockGraph<bool> G_true_mat(G_true, ptr_gruppi);
    //Crete sampler obj
    GGMsampler<BlockGraph> Sampler(data, n, param, hy, init, method,0,print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"GGM Sampler starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    //Posterior Analysis
    auto[MeanK_vec, plinks] = analysis::PointwiseEstimate<decltype(SampledG)>(std::tie(SampledK, SampledG),param.iter_to_storeG, ptr_gruppi);
    MatRow MeanK(MatRow::Zero(p,p));
    unsigned int pos{0};
    for(unsigned int i = 0; i < p; ++i){
      for(unsigned int j = i; j < p; ++j){
        MeanK(i,j) = MeanK_vec(pos++);
      }
    }
    //Create Rcpp::List of sampled Graphs
    std::vector< Rcpp::List > L(SampledG.size());
    int counter = 0;
    for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
      L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
    }
    return Rcpp::List::create ( Rcpp::Named("data") = data,
                                Rcpp::Named("TrueGraph") = G_true_mat.get_graph(),
                                Rcpp::Named("TruePrecision") = Prec_true,
                                Rcpp::Named("SampledG") = L,
                                Rcpp::Named("MeanK")= MeanK, 
                                Rcpp::Named("plinks")= plinks,  
                                Rcpp::Named("AcceptedMoves")=accepted, 
                                Rcpp::Named("VisitedGraphs")=visited   );
  }
  else
    throw std::runtime_error("Error, the only possible form are: Complete and Block.");
}

//' Sampler for Guassian Graphical Models
//'
//' This function draws samples a posteriori from a Gaussian Graphical Models. NON MI SERVE ESPORTARLA
//' @param data matrix of size \eqn{p \times p} containing \eqn{\sum(Y_i^{T}Y_i)}. Data are required to be zero mean.
//' @param p non necessario
//' @param n number of observed data.
//' @param niter number of total iterations to be performed in the sampling. The number of saved iteration will be \eqn{(niter - burnin)/thin}.
//' @param burnin number of discarded iterations.
//' @param thin thining value, it means that only one out of thin itarations is saved.
//' @param b double, it is prior GWishart shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D matrix of double stored columnwise. It is prior GWishart inverse scale parameter. It has to be symmetric and positive definite. 
//' @param MCprior positive integer, the number of iteration for the MonteCarlo approximation of prior normalizing constant of GWishart distribution. Not needed if algo is set to \code{"DRJ"}. 
//' @param MCprior positive integer, the number of iteration for the MonteCarlo approximation of posterior normalizing constant of GWishart distribution. Needed only if algo is set to \code{"MH"}.
//' @param threshold double, threshold for convergence in GWishart sampler.
//' @param form string that may take as values only \code{"Complete"} of \code{"Block"}. It states if the algorithm has to run with \code{"Block"} or \code{"Complete"} graphs.
//' @param prior string with the desidered prior for the graph. Possibilities are \code{"Uniform"}, \code{"Bernoulli"} and for \code{"Block"} graphs only \code{"TruncatedBernoulli"} and \code{"TruncatedUniform"} are also available.
//' @param algo string with the desidered algorithm for sampling from a GGM. Possibilities are \code{"MH"}, \code{"RJ"} and \code{"DRJ"}.
//' @param n_groups int, number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as group parameter. (CHE PER ORA NON ESISTE)
//' @param seed set 0 for random seed.
//' @param Gprior double representing the prior probability of inclusion of each link in case \code{"Bernoulli"} prior is selected for the graph. Set 0.5 for uniform prior.
//' @param sigmaG double, the standard deviation used to perturb elements of precision matrix when constructing the new proposed matrix.
//' @param paddrm double, probability of proposing a new graph by adding one link.
//' @param print_info boolean, if \code{TRUE} progress bar and execution time are displayed.
//' @return This function returns a list with the posterior precision mean, a matrix with the probability of inclusion of each link, the number of accepted moves, the number of visited graphs and the list of all visited graphs.
//' @export 
// [[Rcpp::export]]
Rcpp::List GGM_sampling_c(  Eigen::MatrixXd const & data,
                            int const & p, int const & n, int const & niter, int const & burnin, double const & thin, 
                            Eigen::MatrixXd D, 
                            double const & b = 3.0, int const & MCprior = 100, int const & MCpost = 100, double const & threshold = 1e-8,
                            Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                            Rcpp::Nullable<Rcpp::List> groups = R_NilValue, int seed = 0, double const & Gprior = 0.5, 
                            double const & sigmaG = 0.1, double const & paddrm = 0.5, bool print_info = true  )
{ 
  using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;  
  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost, threshold);
  if (form == "Complete")
  {
    Init<GraphType, unsigned int> init(n,p);
    /*
    GraphType<unsigned int> G0(p);
    G0.fillRandom();
    MatRow K0(utils::rgwish(G0,b,D,1e-16));
    init.set_init(K0,G0);
    Rcpp::Rcout<<"init.G0:"<<std::endl<<init.G0<<std::endl;
    Rcpp::Rcout<<"init.K0:"<<std::endl<<init.K0<<std::endl;
    */
    //Select the method to be used
    auto method = SelectMethod_Generic<GraphType, unsigned>(prior, algo, hy, param);
    //Crete sampler obj
    GGMsampler  Sampler(data, n, param, hy, init, method,0,print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"GGM Sampler starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    //Posterior Analysis
    auto[MeanK_vec, plinks] = analysis::PointwiseEstimate<decltype(SampledG)>(std::tie(SampledK, SampledG),param.iter_to_storeG);
    MatRow MeanK(MatRow::Zero(p,p));
    unsigned int pos{0};
    for(unsigned int i = 0; i < p; ++i){
      for(unsigned int j = i; j < p; ++j){
        MeanK(i,j) = MeanK_vec(pos++);
      }
    } 
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
                                Rcpp::Named("SampledGraphs")= L   );
  }
  else if(form == "Block")
  {
    if(!groups.isNotNull()){
      throw std::runtime_error("Error, group list has to be provided if Block form is selected");  
    }
    Rcpp::List gr(groups);
    //std::shared_ptr<const Groups> ptr_gruppi(std::make_shared<const Groups>(n_groups,p));
          //Groups prova_stampa_gruppi(gr);
          //Rcpp::Rcout<<"prova_stampa_gruppi:"<<std::endl<<prova_stampa_gruppi<<std::endl;
    std::shared_ptr<const Groups> ptr_gruppi = std::make_shared<const Groups>(gr); 
    param.ptr_groups = ptr_gruppi;
    Init<BlockGraph,  unsigned int> init(n,p, ptr_gruppi);
    BlockGraph<unsigned int> G0(ptr_gruppi);
    G0.fillRandom();
    MatRow K0(utils::rgwish(G0.completeview(),b,D,1e-16));
    init.set_init(K0,G0);
    //Select the method to be used
    auto method = SelectMethod_Generic<BlockGraph, unsigned int>(prior, algo, hy, param);
    //Crete sampler obj
    GGMsampler<BlockGraph> Sampler(data, n, param, hy, init, method,0,print_info);
    //Run
    if(print_info){
      Rcpp::Rcout<<"Block GGM Sampler starts:"<<std::endl; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    //Posterior Analysis
    auto[MeanK_vec, plinks] = analysis::PointwiseEstimate<decltype(SampledG)>(std::tie(SampledK, SampledG),param.iter_to_storeG, ptr_gruppi);
    MatRow MeanK(MatRow::Zero(p,p));
    unsigned int pos{0};
    for(unsigned int i = 0; i < p; ++i){
      for(unsigned int j = i; j < p; ++j){
        MeanK(i,j) = MeanK_vec(pos++);
      }
    }
    //Create Rcpp::List of sampled Graphs
    std::vector< Rcpp::List > L(SampledG.size());
    int counter = 0;
    for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
      L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
    }
    return Rcpp::List::create ( Rcpp::Named("MeanK")= MeanK, 
                                Rcpp::Named("plinks")= plinks, 
                                Rcpp::Named("AcceptedMoves")=accepted, 
                                Rcpp::Named("VisitedGraphs")=visited, 
                                Rcpp::Named("SampledGraphs")=L );
  }
  else
    throw std::runtime_error("Error, the only possible form are: Complete and Block.");
}


//' Functional Linear model for smoothing
//'
//' This function performs a linear regression for functional data, according to model (INSERIRE FORMULA MODELLO). 
//' It is not a graphical model, indeed the precision matrix for the regression coefficients is chosen diagonal.
//' @param data matrix of dimension r x n containing the evaluation of n functional data over a grid of r nodes.
//' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be (niter - burnin)/thin.
//' @param burnin the number of discarded iterations. 
//' @param thin the thining value, it means that only one out of thin itarations is saved.
//' @param BaseMat matrix of dimension r x p containing the evalutation of p Bspline basis over a grid of r nodes
//' 
// [[Rcpp::export]]
Rcpp::List FLM_sampling_c(Eigen::MatrixXd const & data, int const & niter, int const & burnin, double const & thin, Eigen::MatrixXd const & BaseMat,
                        Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                        bool diagonal_graph = true, 
                        double const & threshold_GWish = 1e-8, bool print_info = true)
{
  const unsigned int p = BaseMat.cols();
  const unsigned int n = data.cols();
  const unsigned int r = BaseMat.rows();
  if(data.rows() != r)
    throw std::runtime_error("Dimension of data and BaseMat are incoherent. data has to be (n_grid_points x n), BaseMat is (n_grid_points x p)");
  
  if(diagonal_graph){
    FLMHyperparameters hy(p);
    FLMParameters param(niter, burnin, thin, BaseMat);
    InitFLM init(n,p);
    FLMsampler<GraphForm::Diagonal> Sampler(data, param, hy, init, 0, print_info);
    //Run
    auto start = std::chrono::high_resolution_clock::now();
    auto [SaveBeta, SaveMu, SaveTauK, SaveTaueps] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    //Summary
    auto [MeanBeta, MeanMu, MeanTauK,MeanTaueps] = 
       analysis::PointwiseEstimate( std::tie(SaveBeta, SaveMu, SaveTauK, SaveTaueps),param.iter_to_store );
    //Return an Rcpp::List
    Rcpp::List SampledValues = Rcpp::List::create ( Rcpp::Named("SaveBeta")=SaveBeta, 
                                                    Rcpp::Named("SaveMu")=SaveMu, 
                                                    Rcpp::Named("SaveTauK")=SaveTauK ,   
                                                    Rcpp::Named("SaveTaueps")=SaveTaueps ); 

    Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta")=MeanBeta, 
                                                     Rcpp::Named("MeanMu")=MeanMu, 
                                                     Rcpp::Named("MeanTauK")=MeanTauK ,   
                                                     Rcpp::Named("MeanTaueps")=MeanTaueps );   
    /*
    if(Quantiles){
      auto[LowerBound, UpperBound] = analysis::QuantileBeta(SaveBeta, lower_qtl, upper_qtl);
      Rcpp::List Quantiles = Rcpp::List::create( Rcpp::Named("BetaLower")=LowerBound, Rcpp::Named("BetaUpper")=UpperBound );
      return Rcpp::List::create(Rcpp::Named("SampledValues")=SampledValues,Rcpp::Named("PostMeans")=PosteriorMeans, 
                                Rcpp::Named("Quantiles")=Quantiles );
    }
    else{
      return Rcpp::List::create(Rcpp::Named("SampledValues")=SampledValues,Rcpp::Named("PostMeans")=PosteriorMeans, 
                                Rcpp::Named("Quantiles")=R_NilValue );
    }
    */
    return Rcpp::List::create(Rcpp::Named("SampledValues")=SampledValues,Rcpp::Named("PostMeans")=PosteriorMeans);
  }
  else{
    
    FLMHyperparameters hy(p);
    FLMParameters param(niter, burnin, thin, BaseMat, threshold_GWish);
    if(G.rows() != G.cols())
      throw std::runtime_error("Inserted graph is not squared");
    if(G.rows() != p)
      throw std::runtime_error("Dimension of graph and BaseMat are incoherent. graph has to be (p x p), BaseMat is (n_grid_points x p)");
    G.cast<unsigned int>();
    GraphType<unsigned int> Graph(G);
    InitFLM init(n,p, Graph);
    FLMsampler<GraphForm::Fix> Sampler(data, param, hy, init, 0, print_info);
    //Run
    auto start = std::chrono::high_resolution_clock::now();
    auto [SaveBeta, SaveMu, SaveK, SaveTaueps] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    if(print_info){
      Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    }
    //Summary
    auto [MeanBeta, MeanMu, MeanK,MeanTaueps] = 
       analysis::PointwiseEstimate( std::tie(SaveBeta, SaveMu, SaveK, SaveTaueps),param.iter_to_store );
    //Return an Rcpp::List
    Rcpp::List SampledValues = Rcpp::List::create ( Rcpp::Named("SaveBeta")=SaveBeta, 
                                                    Rcpp::Named("SaveMu")=SaveMu, 
                                                    Rcpp::Named("SaveK")=SaveK ,   
                                                    Rcpp::Named("SaveTaueps")=SaveTaueps ); 

    Rcpp::List PosteriorMeans = Rcpp::List::create ( Rcpp::Named("MeanBeta")=MeanBeta, 
                                                     Rcpp::Named("MeanMu")=MeanMu, 
                                                     Rcpp::Named("MeanK")=MeanK ,   
                                                     Rcpp::Named("MeanTaueps")=MeanTaueps );   
    return Rcpp::List::create(Rcpp::Named("SampledValues")=SampledValues,Rcpp::Named("PostMeans")=PosteriorMeans);
    
  }
}



//' Generates random Graphs
//'
//' This function genrates random graph both in \code{"Complete"} or \code{"Block"} form
//' @param p int, the dimension of the graph in its complete form.
//' @param n_groups int, the number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as group parameter.
//' @param form String, the only possibilities are \code{"Complete"} and \code{"Block"}.
//' @param groups List representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
//' @param sparsity double, the desired sparsity of the randomly generated graph. It has to be striclty positive and striclty less than one. It is set to 0.5 otherwise.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return the adjacency matrix of the randomly generated graph.
//' @export
// [[Rcpp::export]]
Rcpp::List                          //Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
 Create_RandomGraph ( int const & p, int const & n_groups = 0, Rcpp::String form = "Complete", Rcpp::Nullable<Rcpp::List> groups = R_NilValue,
                      double sparsity = 0.5, int seed = 0  )
{
  if(p <= 0)
    throw std::runtime_error("Wrong dimension inserted, the number of vertrices has to be strictly positive");
  if(form=="Complete"){
    GraphType Graph(p);
    Graph.fillRandom(sparsity, seed); //Corretness of sparsity is checked inside
    return Rcpp::List::create( Rcpp::Named("G")=Graph.get_graph() );
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
//' This function draws random samples from Multivariate Gaussian distribution. It implements both covariance and precision parametrization.
//' It is also possible to pass directly the Cholesky decomposition if it is available before the call.
//' @param mean vector of size \code{p} representig the mean of the Gaussian distribution.
//' @param Mat matrix of size \eqn{p \times p} reprenting the covariance or the precision matrix or their Cholesky decompositions.
//' @param isPrec boolean, set \code{TRUE} if Mat parameter is a precision, \code{FALSE} if it is a covariance.
//' @param isChol boolean, set \code{TRUE} if Mat parameter is a triangular matrix representig the Cholesky decomposition of the precision or covariance matrix.
//' @param isUpper boolean, used only if \code{isChol} is \code{TRUE}. Set \code{TRUE} if Mat is upper triangular, \code{FALSE} if lower.
//' @return It returns a \code{p} dimensional vector with the sampled values.
//' @export
// [[Rcpp::export]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
rmvnormal(Eigen::VectorXd mean, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Mat, 
          bool isPrec = false, bool isChol = false, bool isUpper = false)
{
  if(Mat.rows() != Mat.cols())
    throw std::runtime_error("Precision or Covariance matrix has to be symmetric.");
  if(!isPrec){ //Covariance parametrization
    if(!isChol){
      return sample::rmvnorm<sample::isChol::False>()(mean, Mat);
    }
    else if(isUpper){
     return sample::rmvnorm<sample::isChol::Upper>()(mean, Mat);
    }
    else{
     return sample::rmvnorm<sample::isChol::Lower>()(mean, Mat);
    }
  }
  else{ //Precision parametrization
    if(!isChol){ //Not chol
      return sample::rmvnorm_prec<sample::isChol::False>()(mean, Mat);
    }
    else if(isUpper){ //Chol, upper triangular
      return sample::rmvnorm_prec<sample::isChol::Upper>()(mean, Mat);
    }
    else{ //Chol, lower triangular 
      return sample::rmvnorm_prec<sample::isChol::Lower>()(mean, Mat);
    }
  }
}

//' Sampler for Wishart random variables
//'
//' This function draws random samples from Wishart distribution. 
//' It is also possible to pass directly the Cholesky decomposition of the Inverse Scale matrix if it is available before the call.
//' @param b double, it is shape parameter. 
//' @param D matrix of size \eqn{p \times p} representig the Inverse Scale parameter. It has to be symmetric and positive definite. 
//' @param isChol boolean, set \code{TRUE} if Mat parameter is a triangular matrix representig the Cholesky decomposition of the precision or covariance
//' @param isUpper boolean, used only if isChol is \code{TRUE}. Set \code{TRUE} if Mat is upper triangular, \code{FALSE} if lower.
//' @return It returns a p x p matrix.
//' @export
// [[Rcpp::export]]
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
rwishart(double const & b, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> D, 
         bool isChol = false, bool isUpper = false)
{
  using MatCol = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  if(D.rows() != D.cols())
    throw std::runtime_error("Inverse scale matrix has to be symmetric.");
  if(!isChol){ //Not chol
    return sample::rwish<MatCol, sample::isChol::False>()(b, D);
  }
  else if(isUpper){ //Chol, upper triangular
    return sample::rwish<MatCol, sample::isChol::Upper>()(b, D);
  }
  else{ //Chol, lower triangular 
    return sample::rwish<MatCol, sample::isChol::Lower>()(b, D);
  }  
}

//' Sampler for Normal distribution
//'
//' @export
// [[Rcpp::export]]
double rnormal(double const & mean = 0.0, double const & sd = 1.0){
  return sample::rnorm()(mean, sd);
}


//' Generate Bspine basis 
//'
//' This function creates a truncated Bspline basis in the interval \eqn{[range(1),range(2)]} and evaluate them over a grid of points.
//' It assumes uniformly spaced breakpoints and constructs the corresponding knot vector using a number of breaks equal to \eqn{n_basis + 2 - order}.
//' @param n_basis number of basis functions.
//' @param range vector of two elements containing first the lower and then the upper bound of the interval.
//' @param points number of grid points where the basis has to be evaluated. It is not used if the points are directly passed in the \code{grid_points} parameter.
//' @param grid_points vector of points where the basis has to be evaluated. If defaulted, then \code{n_points} are uniformly generated in the interval.
//' @param order integer, order of the Bsplines. Set four for cubic splines.
//' @return a List with a matrix of dimension grid_points.size() x n_basis such that r-th rows contains all the spline computed in grid_points[r] and j-th column
//' contains the j-th spline evaluated in all points. It also returns a vector of size n_basis + 2 - order containing the internal knots used to create the spline
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
//' This function creates a truncated Bspline basis in the interval \eqn{[range(1),range(2)]} and evaluate them over a grid of points up to derivative of order \code{nderiv}.
//' For convention, derivatives of order 0 are the splines themselves. This implimes that the first returned element is always equal to the output of the function \code{\link{Generate_Basis}}.
//' @param n_basis number of basis functions.
//' @param nderiv number of derivates that have to be computed.
//' @param range vector of two elements containing first the lower and then the upper bound of the interval.
//' @param points number of grid points where the basis has to be evaluated. It is not used if the points are directly passed in the \code{grid_points} parameter.
//' @param grid_points vector of points where the basis has to be evaluated. If defaulted, then \code{n_points} are uniformly generated in the interval.
//' @param order integer, order of the Bsplines. Set four for cubic splines.
//' @return a List of length nderiv+1 such that each element is a grid_points.size() x n_basis matrix representing the evaluation of 
//' all the k-th derivative of all the splines in all the grid points. 
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



//' Compute quantiles of sampled values
//'
//' @export
// [[Rcpp::export]]
Rcpp::List Compute_QuantileBeta(std::vector<Eigen::MatrixXd> const & SaveBeta, double const & lower_qtl = 0.05, double const & upper_qtl = 0.95)
{
  auto[LowerBound, UpperBound] = analysis::QuantileBeta(SaveBeta, lower_qtl, upper_qtl);
  Rcpp::List Quantiles = Rcpp::List::create( Rcpp::Named("BetaLower")=LowerBound, Rcpp::Named("BetaUpper")=UpperBound );
  return Quantiles;
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
//' This function creates a list with the groups. If possible, groups of equal size are created. The goal of this function is to fix a precise notation that will be used in all the code.
//' It is indeed recommended to use this function to create them as they need to follow a precise notation.
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














#endif





