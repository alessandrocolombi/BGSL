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
//' @param unsigned int, the maximum number of iteration.
//' @param threshold double, the accurancy for checking if the sampled matrix respects the structure of the graph.
//' @param seed int, the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return Rcpp::List containing the sampled matrix as an Eigen RowMajor matrix, a bool that states if the convergence was reached or not and finally an int with the number of performed iterations.
//' If the graph is empty or complete, no iterations are performed. It is automatically converted in standard R list.
//' @export
// [[Rcpp::export]]
Rcpp::List rGwish(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                  double const & b, Eigen::MatrixXd const & D,
                  Rcpp::String norm = "Mean", Rcpp::Nullable<Rcpp::List> groups = R_NilValue, 
                  unsigned int const & max_iter = 500, long double const & threshold = 1e-5, int seed = 0)
{

  if (groups.isNotNull()){ //Assume it is a BlockGraph
    
    Rcpp::List L(groups); // casting to underlying type List
    //Rcpp::Rcout << "List is not NULL." << std::endl;
    std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
    BlockGraph<unsigned int> Graph(G, ptr_gr);

    if(norm == "Mean"){
      //Rcpp::Rcout<<"Mean norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::MeanNorm>(Graph.completeview(), b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold)  );
    }
    if(norm == "Inf"){
      Rcpp::Rcout<<"Inf norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::NormInf>(Graph.completeview(), b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold)  );
    }
    if(norm == "One"){
      Rcpp::Rcout<<"One norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::Norm1>(Graph.completeview(), b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold)   );
    }
    if(norm == "Squared"){
      Rcpp::Rcout<<"Squared norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::NormSq>(Graph.completeview(), b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold)   );
    }
    else{
      Rcpp::Rcout<<"The only available norms are Mean, Inf, One and Squared. Run with default type that is Mean"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<CompleteView, unsigned int, utils::MeanNorm>(Graph.completeview(), b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph.completeview(), Mat, threshold)    );
    } 
  }
  else{ //Assume it is a Complete Graph
    //Rcpp::Rcout << "List is NULL." << std::endl;
    GraphType<unsigned int> Graph(G);
   if(norm == "Mean"){
      //Rcpp::Rcout<<"Mean norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::MeanNorm>(Graph, b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create ( Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold)  );
    }
     if(norm == "Inf"){
      Rcpp::Rcout<<"Inf norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::NormInf>(Graph, b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold)  );
    }
    if(norm == "One"){
      Rcpp::Rcout<<"Norm L1 selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::Norm1>(Graph, b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold)  );
    }
    if(norm == "Squared"){
      Rcpp::Rcout<<"Squared norm selected"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose<GraphType, unsigned int, utils::NormSq>(Graph, b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold)  );
    }
    else{
      Rcpp::Rcout<<"The only available norms are Mean, Inf, One and Squared. Run with default type that is Mean"<<std::endl;
      auto[Mat, converged, iter] = utils::rgwish_verbose(Graph, b, D, max_iter, (unsigned int)seed);
      return Rcpp::List::create (Rcpp::Named("Matrix")= Mat, Rcpp::Named("Converged")=converged, 
             Rcpp::Named("iterations")=iter, Rcpp::Named("CheckStructure")=utils::check_structure(Graph, Mat, threshold)  );
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


//' Testing all the Gaussian Graphical Models samplers with simulated data
//'
//' @param b double, it is shape parameter. It has to be larger than 2 in order to have a well defined distribution.
//' @param D Eigen Matrix of double stored columnwise. It has to be symmetric and positive definite. 
//' @param MCiteration unsigned int, the number of iteration for the MonteCarlo approximation. 
//' @export
// [[Rcpp::export]]
Rcpp::List GGM_sim_sampling( int const & p, int const & n, int const & niter, int const & burnin, double const & thin, Eigen::MatrixXd const & D, 
                            double const & b = 3.0, int const & MCprior = 100, int const & MCpost = 100, 
                            Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                            int const & n_groups = 0, int seed = 0, double sparsity = 0.5, double const & Gprior = 0.5, double const & sigmaG = 0.1, 
                            double const & paddrm = 0.5)
{

  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost);
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
                            double const & b = 3.0, int const & MCprior = 100, int const & MCpost = 100, 
                            Rcpp::String form = "Complete", Rcpp::String prior = "Uniform", Rcpp::String algo = "MH",  
                            int const & n_groups = 0, int seed = 0, double const & Gprior = 0.5, double const & sigmaG = 0.1, 
                            double const & paddrm = 0.5   )
{

  Hyperparameters hy(b, D, paddrm, sigmaG, Gprior);
  Parameters param(niter, burnin, thin, MCprior, MCpost);
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


#endif