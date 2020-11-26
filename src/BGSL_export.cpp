// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#define STRICT_R_HEADERS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "include_graphs.h"
#include "include_headers.h"
#include "include_helpers.h"

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
//' @param seed int with the value of the seed. Default value is 0 that implies that a random seed is drawn.
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
    Rcpp::Rcout << "List is not NULL." << std::endl;
    std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
    BlockGraph<unsigned int> Graph(G, ptr_gr);

    if(norm == "Mean"){
      Rcpp::Rcout<<"Mean norm selected"<<std::endl;
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
    Rcpp::Rcout << "List is NULL." << std::endl;
    GraphType<unsigned int> Graph(G);
   if(norm == "Mean"){
      Rcpp::Rcout<<"Mean norm selected"<<std::endl;
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
//' @param seed int with the value of the seed. Default value is 0 that implies that a random seed is drawn.
//' @return long double, the logarithm of the normalizing constant of GWishart distribution.
//' @export
// [[Rcpp::export]]
long double log_Gconstant(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> G,
                          double const & b, Eigen::MatrixXd const & D,  unsigned int const & MCiteration = 100, 
                          Rcpp::Nullable<Rcpp::List> groups = R_NilValue, int seed = 0)
{
    if (groups.isNotNull()){ //Assume it is in block form with respect to the groups given in groups
      
      Rcpp::List L(groups); // casting to underlying type List
      Rcpp::Rcout << "List is not NULL." << std::endl;
      std::shared_ptr<const Groups> ptr_gr = std::make_shared<const Groups>(L); //Create pointer to groups
      BlockGraph<unsigned int> Graph(G, ptr_gr);
      return utils::log_normalizing_constat(Graph.completeview(), b, D, MCiteration, seed);
    }
    else{ //Assume it is a BlockGraph
      Rcpp::Rcout << "List is NULL." << std::endl;
      GraphType<unsigned int> Graph(G);
      return utils::log_normalizing_constat(Graph.completeview(), b, D, MCiteration, seed);
    }
}




