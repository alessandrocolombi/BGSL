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
//' This function draws a random matrices, distributed according to the G-Wishart distribution with shape parameter b and inverse scale D, 
//' with respect to the graph structure G. METTERE LA FORMULA DELLA DISTRIBUZIONE 
//' It implements the algorithm described by METTERE CITAZIONI LENKOSKI. It works with both decomposable and non decomposable graphs. 
//' In particular it is possible to provide a graph in block form. 
//' @param G Eigen Matrix of unsigned int stored columnwise. If a standard R matrix is provided, it is automaticaly converted. The lower part 
//' is not used, the elements is taken in consideration only if the graph is in block form, i.e if groups is non null.
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param norm Rcpp::String to choose the matrix norm with respect to whom convergence takes place. The available choices are Mean, Inf, One and Squared. Mean is the default value
//' @param groups a Rcpp list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave NULL if the graph is not in block form.
//' @param max_iter unsigned int, the maximum number of iteration.
//' @param threshold_check double, the accurancy for checking if the sampled matrix respects the structure of the graph.
//' @param threshold_conv double, stop algorithm if the difference between two subsequent iterations is less than threshold_conv.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return Rcpp::List containing the sampled matrix as an Eigen RowMajor matrix, a bool that states if the convergence was reached or not and finally an int with the number of performed iterations.
//' If the graph is empty or complete, no iterations are performed. It is automatically converted in standard R list.
//' @export
// [[Rcpp::export]]
Rcpp::List rGwish(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
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


//' log of GWishart normalizing constant
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
                            double const & paddrm = 0.5)
{

  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost, threshold);
  if (form == "Complete")
  {
    Init<GraphType, unsigned int> init(n,p);
    //Select the method to be used
    auto method = SelectMethod_GraphType(prior, algo, hy, param);
    //Simulate data
    auto [data, Prec_true, G_true] = utils::SimulateDataGGM_Complete(p,n,seed);
    GraphType<bool> G_true_mat(G_true);
    //Crete sampler obj
    GGMsampler  Sampler(data, n, param, hy, init, method);
    //Run
    Rcpp::Rcout<<"GGM Sampler is about to start:"<<std::endl; 
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    //Compute plinks
    return Rcpp::List::create ( Rcpp::Named("data") = data,
                                Rcpp::Named("TrueGraph") = G_true_mat.get_graph(),
                                Rcpp::Named("TruePrecision") = Prec_true,
                                Rcpp::Named("plinks")= analysis::Compute_plinks(SampledG, param.iter_to_storeG), 
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
    auto method = SelectMethod_BlockGraph(prior, algo, hy, param);
    //Simulate data
    auto [data, Prec_true, G_true] = utils::SimulateDataGGM_Block(p,n,ptr_gruppi,seed);
    BlockGraph<bool> G_true_mat(G_true, ptr_gruppi);
    //Crete sampler obj
    GGMsampler<BlockGraph> Sampler(data, n, param, hy, init, method);
    //Run
    Rcpp::Rcout<<"GGM Sampler is about to start:"<<std::endl; 
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    //Compute plinks
    return Rcpp::List::create ( Rcpp::Named("data") = data,
                                Rcpp::Named("TrueGraph") = G_true_mat.get_graph(),
                                Rcpp::Named("TruePrecision") = Prec_true,
                                Rcpp::Named("plinks")= analysis::Compute_plinks(SampledG, param.iter_to_storeG, ptr_gruppi), 
                                Rcpp::Named("AcceptedMoves")=accepted, 
                                Rcpp::Named("VisitedGraphs")=visited   );
  }
  else
    throw std::runtime_error("Error, the only possible form are: Complete and Block.");
}

//' Testing all the Gaussian Graphical Models samplers
//'
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param MCiteration unsigned int, the number of iteration for the MonteCarlo approximation. 
//' @export
// [[Rcpp::export]]
Rcpp::List GGM_sampling(    Eigen::MatrixXd const & data,
                            int const & p, int const & n, int const & niter, int const & burnin, double const & thin, Eigen::MatrixXd const & D, 
                            double const & b = 3.0, int const & MCprior = 100, int const & MCpost = 100, double const & threshold = 1e-8,
                            Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                            int const & n_groups = 0, int seed = 0, double const & Gprior = 0.5, double const & sigmaG = 0.1, 
                            double const & paddrm = 0.5   )
{

  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost, threshold);
  if (form == "Complete")
  {
    Init<GraphType, unsigned int> init(n,p);
    //Select the method to be used
    auto method = SelectMethod_GraphType(prior, algo, hy, param);
    //Crete sampler obj
    GGMsampler  Sampler(data, n, param, hy, init, method);
    //Run
    Rcpp::Rcout<<"GGM Sampler is about to start:"<<std::endl; 
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    //Return an Rcpp::List
    std::vector< Rcpp::List > L(SampledG.size());
    int counter = 0;
    for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
      L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
    }
    return Rcpp::List::create ( Rcpp::Named("plinks")=analysis::Compute_plinks(SampledG, param.iter_to_storeG), 
                                Rcpp::Named("AcceptedMoves")=accepted, 
                                Rcpp::Named("VisitedGraphs")=visited, Rcpp::Named("SampledGraphs")=L   );
  }
  else if(form == "Block")
  {
    if(n_groups <= 1 || n_groups > p)
      throw std::runtime_error("Error, invalid number of groups inserted");  
    std::shared_ptr<const Groups> ptr_gruppi(std::make_shared<const Groups>(n_groups,p));
    param.ptr_groups = ptr_gruppi;
    Init<BlockGraph,  unsigned int> init(n,p, ptr_gruppi);
    //Select the method to be used
    auto method = SelectMethod_BlockGraph(prior, algo, hy, param);
    //Crete sampler obj
    GGMsampler<BlockGraph> Sampler(data, n, param, hy, init, method);
    //Run
    Rcpp::Rcout<<"GGM Sampler is about to start:"<<std::endl; 
    auto start = std::chrono::high_resolution_clock::now();
    auto [SampledK, SampledG, accepted, visited] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    Rcpp::Rcout<<std::endl<<"Time: "<<timer.count()<<" s "<<std::endl; 
    std::vector< Rcpp::List > L(SampledG.size());
    int counter = 0;
    for(auto it = SampledG.cbegin(); it != SampledG.cend(); ++it){
      L[counter++] = Rcpp::List::create(Rcpp::Named("Graph")=it->first, Rcpp::Named("Frequency")=it->second);
    }
    return Rcpp::List::create ( Rcpp::Named("plinks")= analysis::Compute_plinks(SampledG, param.iter_to_storeG, ptr_gruppi), 
                                Rcpp::Named("AcceptedMoves")=accepted, 
                                Rcpp::Named("VisitedGraphs")=visited, Rcpp::Named("SampledGraphs")=L );
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
                        double const & threshold_GWish = 1e-8)
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
    FLMsampler<GraphForm::Diagonal> Sampler(data, param, hy, init);
    //Run
    auto start = std::chrono::high_resolution_clock::now();
    auto [SaveBeta, SaveMu, SaveTauK, SaveTaueps] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    Rcpp::Rcout<<"Time: "<<timer.count()<<" s "<<std::endl;
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
    FLMsampler<GraphForm::Fix> Sampler(data, param, hy, init);
    //Run
    auto start = std::chrono::high_resolution_clock::now();
    auto [SaveBeta, SaveMu, SaveK, SaveTaueps] = Sampler.run();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = stop - start;
    Rcpp::Rcout<<"Time: "<<timer.count()<<" s "<<std::endl;
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
//' This function genrates random graph both in Complete or Block form
//' @param p int, the dimension of the graph in its complete form.
//' @param n_groups int, the number of desired groups. Not used if form is Complete or if the groups are directly insered as group parameter.
//' @param form Rcpp::String, the only possibilities are Complete and Block.
//' @param groups a Rcpp list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group, 
//' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave NULL if the graph is not in block form.
//' @param sparsity double, the desired sparsity of the randomly generated graph. It has to be striclty positive and striclty less than one. It is set to 0.5 otherwise.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return the adjacency matrix of the randomly generated graph.
//' @export
// [[Rcpp::export]]
Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
 Create_RandomGraph ( int const & p, int const & n_groups = 0, Rcpp::String form = "Complete", Rcpp::Nullable<Rcpp::List> groups = R_NilValue,
                      double sparsity = 0.5, int seed = 0  )
{
  if(p <= 0)
    throw std::runtime_error("Wrong dimension inserted, the number of vertrices has to be strictly positive");
  if(form=="Complete"){
    GraphType Graph(p);
    Graph.fillRandom(sparsity, seed); //Corretness of sparsity is checked inside
    return Graph.get_graph();
  } 
  else if (form == "Block"){

      if (groups.isNotNull()){ //Assume it is a BlockGraph
          Rcpp::List L(groups); // casting to underlying type List
          std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
          BlockGraph<unsigned int> Graph(ptr_gr);
          Graph.fillRandom(sparsity, seed);
          return Graph.get_graph();
      }
      else{
        if(n_groups <= 1 || n_groups > p)
          throw std::runtime_error("Incoherent number of groups inserted, it has to be at least 2 and smaller than p");
        else{
          std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(n_groups, p); //Create pointer to groups
          BlockGraph<unsigned int> Graph(ptr_gr);
          Graph.fillRandom(sparsity, seed);
          return Graph.get_graph();
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
//' @param mean Eigen column vector of size p representig the mean of the Gaussian distribution.
//' @param Mat Eigen p x p column major matrix reprenting the covariance or the precision or their Cholesky decompositions.
//' @param isPrec bool, set true if Mat parameter is a precision, false if it is a covariance.
//' @param isChol bool, set true if Mat parameter is a triangular matrix representig the Cholesky decomposition of the precision or covariance
//' @param isUpper bool, used only if isChol is true. Set true if Mat is upper triangular, false if lower.
//' @return It returns a p dimensional vector with the sampled values.
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
//' It is also possible to pass directly the Cholesky decomposition of the inverse scale matrix if it is available before the call.
//' @param b double, it is shape parameter. 
//' @param D Eigen p x p Matrix of double stored columnwise representig the inverse scale parameter. It has to be symmetric and positive definite. 
//' @param isChol bool, set true if Mat parameter is a triangular matrix representig the Cholesky decomposition of the precision or covariance
//' @param isUpper bool, used only if isChol is true. Set true if Mat is upper triangular, false if lower.
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
//' This function creates a truncated Bspline basis in the interval [range(1),range(2)] and evaluate them over a grid of points.
//' It assumes uniformly spaced breakpoints and constructs the corresponding knot vector using a number of breaks equal to n_basis + 2 - order.
//' @param n_basis number of basis functions.
//' @param range vector of two elements containing first the lower and then the upper bound of the interval.
//' @param points number of grid points where the basis has to be evaluated. It is not used if the points are directly passed in the grid_points parameter.
//' @param grid_points vector of points where the basis has to be evaluated. If defaulted, then n_points are uniformly generated in the interval.
//' @param order order of the Bsplines. Set four for cubic splines.
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
//' This function creates a truncated Bspline basis in the interval [range(1),range(2)] and evaluate them over a grid of points up to derivative of order nderiv.
//' For convention, derivatives of order 0 are the splines themselves. This implimes that the first returned element is always equal to the output of the function Generate_Basis.
//' @param n_basis number of basis functions.
//' @param nderiv number of derivates that have to be computed.
//' @param range vector of two elements containing first the lower and then the upper bound of the interval.
//' @param points number of grid points where the basis has to be evaluated. It is not used if the points are directly passed in the grid_points parameter.
//' @param grid_points vector of points where the basis has to be evaluated. If defaulted, then n_points are uniformly generated in the interval.
//' @param order order of the Bsplines. Set four for cubic splines.
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



//' Generate Bspine basis and its derivatives
//'
//' @export
// [[Rcpp::export]]
Rcpp::List Compute_QuantileBeta(std::vector<Eigen::MatrixXd> const & SaveBeta, double const & lower_qtl = 0.05, double const & upper_qtl = 0.95){
  auto[LowerBound, UpperBound] = analysis::QuantileBeta(SaveBeta, lower_qtl, upper_qtl);
  Rcpp::List Quantiles = Rcpp::List::create( Rcpp::Named("BetaLower")=LowerBound, Rcpp::Named("BetaUpper")=UpperBound );
  return Quantiles;
}






#endif