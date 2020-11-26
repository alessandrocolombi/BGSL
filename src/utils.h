#ifndef __UTILS_H__
#define __UTILS_H__

#include "include_headers.h"
#include "include_graphs.h"
#include "GSLwrappers.h"

//StatsLib
//#ifndef STATS_ENABLE_EIGEN_WRAPPERS
//#define STATS_ENABLE_EIGEN_WRAPPERS
//#define STATS_USE_OPENMP --> mi sa che è buggato il codice della libreria su sta cosa
//#include "stats.h"
//#endif

namespace utils{

	using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using IdxType   = std::size_t;
	using Container = std::vector<unsigned int>;
	//Some constants
	constexpr double pi 			 = M_PI;
	constexpr double pi_2 			 = M_PI_2;
	constexpr double sqrt_2 		 = M_SQRT2;
	constexpr double two_over_sqrtpi = M_2_SQRTPI;
	constexpr long double log_2	 	 = M_LN2;
	constexpr double sqrt_pi 		 = two_over_sqrtpi*pi_2;
	constexpr long double sqrt_2pi 	 = sqrt_pi*sqrt_2;
	constexpr long double log_2pi	 = std::log(2*pi);
	constexpr long double log_pi	 = std::log(pi);


  //------------------------------------------------------------------------------------------------------------------------------------------------------

	namespace Symmetric{
		struct True; //Assuming it is upper symmetric
		struct False;
	}

	template<typename Sym = Symmetric::False> 
	MatRow SubMatrix(Container const & nbd, MatRow const & M){
		//Check
		if(M.rows() != M.cols())
			throw std::runtime_error("Passing different number of rows and cols in a symmetric matrix. Maybe you need to use Symmetric::False");
		if(*nbd.crbegin() >= M.rows())
			throw std::runtime_error("Indeces exceed matrix dimension");
		//Create return obj
		MatRow res(nbd.size(), nbd.size());
		//Fill the sub-matrix
		//#pragma omp parallel
		//{
			//std::cout<<"Hello SubMatrix! "<<std::endl;
		//}
		if constexpr(std::is_same<Symmetric::True, Sym>::value){
			#pragma omp parallel for shared(res, M)
			for(IdxType i = 0; i < res.rows(); ++i)
				for(IdxType j = i; j < res.cols(); ++j)
					res(i,j) = M(nbd[i], nbd[j]);

			return res.selfadjointView<Eigen::Upper>();
		}
		else{
			#pragma omp parallel for shared(res, M)
			for(IdxType i = 0; i < res.rows(); ++i)
				for(IdxType j = 0; j < res.cols(); ++j)
					res(i,j) = M(nbd[i], nbd[j]);

			return res;
		}
	} //Gets vector whose rows and cols has to be estracted
	template<typename Sym = Symmetric::False>
	MatRow SubMatrix(unsigned int const & exclude, MatRow const & M){
		//Check
		if(M.cols() != M.rows())
			throw std::runtime_error("Non square matrix inserted.");
		if(exclude >= M.rows())
			throw std::runtime_error("Index exceed matrix dimension");
		//Create return obj
		MatRow res(MatRow::Zero(M.rows()-1, M.rows()-1));
		//Fill the sub-matrix
		if(exclude == 0){
			res = M.block(1,1,M.rows()-1, M.rows()-1);
		}
		else if(exclude == M.rows()-1){
			res = M.block(0,0,M.rows()-1, M.rows()-1);
		}
		else{
			res.block(0,0,exclude,exclude) = M.block(0,0,exclude,exclude);
			res.block(exclude,exclude,res.rows()-exclude,res.rows()-exclude) = M.block(exclude+1,exclude+1,res.rows()-exclude,res.rows()-exclude);
			res.block(0,exclude,exclude,res.rows()-exclude) = M.block(0,exclude+1,exclude,res.rows()-exclude);
				if constexpr(!std::is_same<Symmetric::True, Sym>::value)
					res.block(exclude,0,res.rows()-exclude,exclude) = M.block(exclude+1,0,res.rows()-exclude,exclude);
		}
		if constexpr(std::is_same<Symmetric::True, Sym>::value)
			return res.selfadjointView<Eigen::Upper>();
		else
			return res;
	} //gets the index whose row and column has to be excluded. Symmetry here is not for efficiency but to be sure to get a sym matrix
	MatRow SubMatrix(Container const & nbd_rows, Container const & nbd_cols, MatRow const & M){
		if(*nbd_rows.crbegin() >= M.rows() || *nbd_cols.crbegin() >= M.cols())
			throw std::runtime_error("Indeces exceed matrix dimension");
		//Create return obj
		MatRow res(nbd_rows.size(), nbd_cols.size());
			for(IdxType i = 0; i < res.rows(); ++i)
				for(IdxType j = 0; j < res.cols(); ++j)
					res(i,j) = M(nbd_rows[i], nbd_cols[j]);
		return res;
	} //Passing rows to be exctracted and cols to be extracted. May be different
	MatRow SubMatrix(Container const & nbd_rows, unsigned int const & idx, MatRow const & M){
		if(*nbd_rows.crbegin() >= M.rows() || idx >= M.cols())
			throw std::runtime_error("Indeces exceed matrix dimension");
		//Create return obj
		Eigen::VectorXd res(nbd_rows.size());
			for(IdxType i = 0; i < res.size(); ++i)
				res(i) = M(nbd_rows[i], idx);
		return res;
	} //Passing rows to be exctracted and the colums index to be extracted.
	MatRow SubMatrix(unsigned int const & idx, Container const & nbd_cols, MatRow const & M){
		if(*nbd_cols.crbegin() >= M.cols() || idx >= M.rows())
			throw std::runtime_error("Indeces exceed matrix dimension");
		Eigen::RowVectorXd res(nbd_cols.size());
			for(IdxType j = 0; j < res.size(); ++j)
				res(j) = M(idx, nbd_cols[j]);
		return res;
	} //Passing cols to be exctracted and row index to be extracted.

	//------------------------------------------------------------------------------------------------------------------------------------------------------
		//GraphStructure may be GraphType / CompleteViewAdj / CompleteView
	template<template <typename> class Graph, typename T = unsigned int >
	bool check_structure(Graph<T> const & G, MatRow const & data, double threshold = 1e-5){
		for(IdxType i = 1; i < data.rows()-1; ++i){
			for(IdxType j = i+1; j < data.cols(); ++j){
				if( ( G(i,j) == 0 && std::abs(data(i,j))>threshold ) || (G(i,j) == 1 && std::abs(data(i,j))<threshold)){
					//std::cout<<"Grafo("<<i<<", "<<j<<") = "<<G(i,j)<<"; Mat("<<i<<", "<<j<<") = "<<data(i,j)<<std::endl;
					return false;
				}
			}
		}
		return true;
	}
	//------------------------------------------------------------------------------------------------------------------------------------------------------
	double power(double const & x, int const & esp){
		double res{1.0};
		if(esp == 0)
			return res;
		#pragma parallel for reduction(*:res)
		for(unsigned int i = 0; i < std::abs(esp); ++i)
			res *= x;
		if(esp > 0)
			return res;
		else
			return 1.0/res;
	} //Rarely used
	//------------------------------------------------------------------------------------------------------------------------------------------------------
	struct NormInf{
		using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		static double norm(MatRow const & A, MatRow const & B){
			return (A - B).lpNorm<Eigen::Infinity>();
		}
	};
	struct Norm1{
		using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		static double norm(MatRow const & A, MatRow const & B){
			return (A - B).lpNorm<1>();
		}
	};
	struct NormSq{
		using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		static double norm(MatRow const & A, MatRow const & B){
			return (A - B).squaredNorm();
		}
	};//NB, it returns the squared L2 norm ||A - B||^2
	struct MeanNorm{
		using MatRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		static double norm(MatRow const & A, MatRow const & B){
			Eigen::VectorXd Ones(Eigen::VectorXd::Constant(A.cols(), 1));
			return  ( static_cast<double>(Ones.transpose() * (A-B).cwiseAbs() * Ones) ) / static_cast<double>(A.cols()*A.cols());
		}
	}; //returns sum_ij(a_ij - b_ij)/N*N


	//GraphStructure may be GraphType / CompleteViewAdj / CompleteView
 	template<template <typename> class GraphStructure = GraphType, typename T = unsigned int, typename NormType = MeanNorm >
 	MatRow rgwish(GraphStructure<T> const & G, double const & b, Eigen::MatrixXd const & D, unsigned int seed = 0){
		//Typedefs
		using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		using MatCol   	= Eigen::MatrixXd;
		using IdxType  	= std::size_t;
		using iterator  = std::vector<unsigned int>::iterator;
		using citerator = std::vector<unsigned int>::const_iterator;
		//Checks -> meglio farli in R
		if(D.rows()!=D.cols())
			throw std::runtime_error("Non squared matrix inserted");
		if(G.get_size() != D.rows())
			throw std::runtime_error("Dimension of D is not equal to the number of nodes");
		//Set parameters
		unsigned int const N = G.get_size();
		unsigned int const n_links = G.get_n_links();
		unsigned int const max_iter = 500;
		double const threshold = 1e-8;
		bool converged = false;
		unsigned int it{0};
		double norm_res{1.0};
		if(seed == 0)
			seed = std::chrono::system_clock::now().time_since_epoch().count();	
		sample::GSL_RNG engine(seed);

		if(n_links == 0){
					//std::cout<<"Empty Graph"<<std::endl;
			MatRow K_return(MatRow::Identity(N,N));
			for(unsigned int i = 0; i < N; ++i) //Non cache friendly
				K_return(i,i) = std::sqrt(  sample::rchisq()(engine, (double)(b + N - i - 1))  );
			return K_return;
		}

		//Step 1: Draw K from Wish(b,D) = wish(D^-1, b+N-1)
		MatCol Inv_D(D.llt().solve(MatCol::Identity(N,N))); 
		MatCol K( sample::rwish<MatCol, sample::isChol::False>()(engine, b, Inv_D) ); 
				//std::cout<<"K: "<<std::endl<<K<<std::endl;
		if(n_links == G.get_possible_links()){
					//std::cout<<"Complete Graph"<<std::endl;
			MatRow K_return(K);
			return K_return; 
		}
		//Step 2: Set Sigma=K^-1 and initialize Omega=Sigma
			MatRow Sigma(K.llt().solve(MatRow::Identity(N, N)));
			MatRow Omega_old(Sigma);
			MatRow Omega(Sigma);

			//std::cout<<"Sigma: "<<std::endl<<Sigma<<std::endl;
			//std::cout<<"Omega: "<<std::endl<<Omega<<std::endl;
		//Start looping
			while(!converged && it < max_iter){
			it++;
			//For every node. Perché devo farlo in ordine? Posso paralellizzare? Avrei sicuro data race ma perché devo fare questi aggiornamenti in ordine
			for(IdxType i = 0; i < N; ++i){
					//std::cout<<"i = "<<i<<std::endl;
					//std::cout<<"Stampo il nbd:"<<std::endl;
				std::vector<unsigned int> nbd_i = G.get_nbd(i);
					//for(auto el : nbd_i)
						//std::cout<<el<<" ";
					//std::cout<<" "<<std::endl;

				Eigen::VectorXd beta_i = Eigen::VectorXd::Zero(N-1);
				if(nbd_i.size() == 0){
					//do nothing
				}
				else if( nbd_i.size() == 1){
					//I need Omega to be symmetric here. This is not the best possible way because a need to complete Omega.
					//However i guess that this case would be pretty rare.
					Omega = Omega.selfadjointView<Eigen::Upper>();
							//std::cout<<"Omega quando nbd_i è single: "<<std::endl<<Omega<<std::endl;
					unsigned int k = nbd_i[0];
					double beta_star_i = Sigma(k,i) / Omega(k,k); //In this case it is not a linear system but a simple, scalar, equation
					if(i == 0){
						beta_i = Omega.block(1,k, N-1,1) * beta_star_i; //get k-th column except first row
					}
					else if(i == N-1){
						beta_i = Omega.block(0,k, N-1,1) * beta_star_i; //get k-th column except last row
					}
					else{
						//get k-th column except i-th row
						beta_i.head(i) = Omega.block(0,k, i, 1) * beta_star_i;
						beta_i.tail(beta_i.size()-i) = Omega.block(i+1,k, N-i-1,1) * beta_star_i;
					}
								//std::cout<<"beta_i: "<<std::endl<<beta_i<<std::endl;
				}
				else{
					//Step 3: Compute beta_star_i = (Omega_Ni_Ni)^-1*Sigma_Ni_i. beta_star_i in R^|Ni|
						MatRow Omega_Ni_Ni( SubMatrix<Symmetric::True>(nbd_i, Omega) );
							//std::cout<<"Omega_Ni_Ni: "<<std::endl<<Omega_Ni_Ni<<std::endl;
						Eigen::VectorXd Sigma_Ni_i( SubMatrix(nbd_i, i, Sigma) ); //-->questa SubMatrix() meglio farla con block() di eigen -> ma è sbatti
							//std::cout<<"Sigma_Ni_i: "<<std::endl<<Sigma_Ni_i<<std::endl;
						Eigen::VectorXd beta_star_i = Omega_Ni_Ni.llt().solve(Sigma_Ni_i);
							//std::cout<<"beta_star_i: "<<std::endl<<beta_star_i<<std::endl;
					//Step 4: Define beta_hat_i in R^N-1 such that:
					//- Entries not associated to elements in Ni are 0
					//- Entries associated to elements in Ni are given by beta_star_i
					//- Has no element associated to i
						Eigen::VectorXd beta_hat_i(Eigen::VectorXd::Zero(N-1));
						for(citerator j_it = nbd_i.cbegin(); j_it < nbd_i.cend(); ++j_it){
								//std::cout<<"*j_it = "<<*j_it<<std::endl;
							if(*j_it < i)
								beta_hat_i(*j_it) = beta_star_i( j_it - nbd_i.cbegin() );
							else if(*j_it > i)
								beta_hat_i(*j_it - 1) = beta_star_i( j_it - nbd_i.cbegin() );
						}
								//std::cout<<"beta_hat_i: "<<std::endl<<beta_hat_i<<std::endl;
					//Step 5: Set i-th row and col of Omega equal to Omega_noti_noti*beta_hat_i
						MatRow Omega_noti_noti( SubMatrix<Symmetric::True>(i , Omega) );
								//std::cout<<"Omega_noti_noti: "<<std::endl<<Omega_noti_noti<<std::endl;
						beta_i = Omega_noti_noti * beta_hat_i;
								//std::cout<<"beta_i: "<<std::endl<<beta_i<<std::endl;
				}
				//Plug beta_i into the i-th row / col except from diagonal element
				if(i == 0)//plug in first row (no first element)
					Omega.block(0,1,1,N-1) = beta_i.transpose();
				else if(i == N-1)//plug in last column (no last element)
					Omega.block(0,N-1,N-1,1) = beta_i;
				else{
							//std::cout<<"Omega block 1:"<<std::endl<<Omega.block(0,i,i,1)<<std::endl;
							//std::cout<<"Omega block 2:"<<std::endl<<Omega.block(i,i+1,1,beta_i.size()-i)<<std::endl;
							//std::cout<<"beta_i: "<<std::endl<<beta_i<<std::endl;
					Omega.block(0,i,i,1) 	   			 = beta_i.head(i);
					Omega.block(i,i+1,1,beta_i.size()-i) = beta_i.transpose().tail(beta_i.size()-i);
				}
							//std::cout<<"Omega: "<<std::endl<<Omega<<std::endl;
			}
		//Step 6: Compute the norm of differences
			norm_res = NormType::norm(Omega.selfadjointView<Eigen::Upper>(), Omega_old.selfadjointView<Eigen::Upper>());
			Omega_old = Omega;
		//Step 7: Check stop criteria
				//std::cout<<"Norm res = "<<norm_res<<std::endl;
			if(norm_res < threshold){
				converged = true;
			}
		}
		//Step 8: check if everything is fine
		//std::cout<<"Converged? "<<converged<<std::endl;
		//std::cout<<"#iterazioni = "<<it<<std::endl;
		//Step 9: return K = Omega^-1
		return Omega.selfadjointView<Eigen::Upper>().llt().solve(MatRow::Identity(N, N));
		//Omega = Omega.selfadjointView<Eigen::Upper>();
		//return Omega.inverse();
	}
	//------------------------------------------------------------------------------------------------------------------------------------------------------

	//GraphStructure may be GraphType / CompleteViewAdj / CompleteView
	//This functions returns also if convergence was reached and the number of iterations
	template<template <typename> class GraphStructure = GraphType, typename T = unsigned int, typename NormType = MeanNorm >
	std::tuple< MatRow, bool, int> 
	rgwish_verbose(GraphStructure<T> const & G, double const & b, Eigen::MatrixXd const & D, unsigned int const & max_iter = 500, unsigned int seed = 0)
	{
		//Typedefs
			using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
			using MatCol   	= Eigen::MatrixXd;
			using IdxType  	= std::size_t;
			using iterator  = std::vector<unsigned int>::iterator;
			using citerator = std::vector<unsigned int>::const_iterator;
		//Checks 
			if(b <= 2)
				throw std::runtime_error("The Gwishart distribution is well defined only if shape parameter b is larger than 2");
			if(D.rows()!=D.cols())
				throw std::runtime_error("Non squared matrix inserted");
			if(G.get_size() != D.rows())
				throw std::runtime_error("Dimension of D is not equal to the number of nodes");
		//Set parameters
			unsigned int const N = G.get_size();
			unsigned int const n_links = G.get_n_links();
			double const threshold = 1e-8;
			bool converged = false;
			unsigned int it{0};
			double norm_res{1.0};
			if(seed == 0)
				seed = std::chrono::system_clock::now().time_since_epoch().count();	
			sample::GSL_RNG engine(seed);

			if(n_links == 0){
						//std::cout<<"Empty Graph"<<std::endl;
				MatRow K_return(MatRow::Identity(N,N));
				for(unsigned int i = 0; i < N; ++i) //Non cache friendly
					K_return(i,i) = std::sqrt(  sample::rchisq()(engine, (double)(b + N - i - 1))  );
				return std::tuple(K_return,true, 0);
			}

		//Step 1: Draw K from Wish(b,D) = wish(D^-1, b+N-1)
			MatCol Inv_D(D.llt().solve(MatCol::Identity(N,N))); 
			MatCol K( sample::rwish<MatCol, sample::isChol::False>()(engine, b, Inv_D) ); 
					//std::cout<<"K: "<<std::endl<<K<<std::endl;
			if(n_links == G.get_possible_links()){
						//std::cout<<"Complete Graph"<<std::endl;
				MatRow K_return(K);
				return std::tuple(K_return,true, 0); 
			}
		//Step 2: Set Sigma=K^-1 and initialize Omega=Sigma
			MatRow Sigma(K.llt().solve(MatRow::Identity(N, N)));
			MatRow Omega_old(Sigma);
			MatRow Omega(Sigma);

				//std::cout<<"Sigma: "<<std::endl<<Sigma<<std::endl;
				//std::cout<<"Omega: "<<std::endl<<Omega<<std::endl;
		//Start looping
			while(!converged && it < max_iter){
				it++;
				//For every node. Perché devo farlo in ordine? Posso paralellizzare? Avrei sicuro data race ma perché devo fare questi aggiornamenti in ordine
				for(IdxType i = 0; i < N; ++i){
						//std::cout<<"i = "<<i<<std::endl;
						//std::cout<<"Stampo il nbd:"<<std::endl;
					std::vector<unsigned int> nbd_i = G.get_nbd(i);
						//for(auto el : nbd_i)
							//std::cout<<el<<" ";
						//std::cout<<" "<<std::endl;

					Eigen::VectorXd beta_i = Eigen::VectorXd::Zero(N-1);
					if(nbd_i.size() == 0){
						//do nothing
					}
					else if( nbd_i.size() == 1){
						//I need Omega to be symmetric here. This is not the best possible way because a need to complete Omega.
						//However i guess that this case would be pretty rare.
						Omega = Omega.selfadjointView<Eigen::Upper>();
								//std::cout<<"Omega quando nbd_i è single: "<<std::endl<<Omega<<std::endl;
						unsigned int k = nbd_i[0];
						double beta_star_i = Sigma(k,i) / Omega(k,k); //In this case it is not a linear system but a simple, scalar, equation
						if(i == 0){
							beta_i = Omega.block(1,k, N-1,1) * beta_star_i; //get k-th column except first row
						}
						else if(i == N-1){
							beta_i = Omega.block(0,k, N-1,1) * beta_star_i; //get k-th column except last row
						}
						else{
							//get k-th column except i-th row
							beta_i.head(i) = Omega.block(0,k, i, 1) * beta_star_i;
							beta_i.tail(beta_i.size()-i) = Omega.block(i+1,k, N-i-1,1) * beta_star_i;
						}
									//std::cout<<"beta_i: "<<std::endl<<beta_i<<std::endl;
					}
					else{
						//Step 3: Compute beta_star_i = (Omega_Ni_Ni)^-1*Sigma_Ni_i. beta_star_i in R^|Ni|
							MatRow Omega_Ni_Ni( SubMatrix<Symmetric::True>(nbd_i, Omega) );
								//std::cout<<"Omega_Ni_Ni: "<<std::endl<<Omega_Ni_Ni<<std::endl;
							Eigen::VectorXd Sigma_Ni_i( SubMatrix(nbd_i, i, Sigma) ); //-->questa SubMatrix() meglio farla con block() di eigen -> ma è sbatti
								//std::cout<<"Sigma_Ni_i: "<<std::endl<<Sigma_Ni_i<<std::endl;
							Eigen::VectorXd beta_star_i = Omega_Ni_Ni.llt().solve(Sigma_Ni_i);
								//std::cout<<"beta_star_i: "<<std::endl<<beta_star_i<<std::endl;
						//Step 4: Define beta_hat_i in R^N-1 such that:
						//- Entries not associated to elements in Ni are 0
						//- Entries associated to elements in Ni are given by beta_star_i
						//- Has no element associated to i
							Eigen::VectorXd beta_hat_i(Eigen::VectorXd::Zero(N-1));
							for(citerator j_it = nbd_i.cbegin(); j_it < nbd_i.cend(); ++j_it){
									//std::cout<<"*j_it = "<<*j_it<<std::endl;
								if(*j_it < i)
									beta_hat_i(*j_it) = beta_star_i( j_it - nbd_i.cbegin() );
								else if(*j_it > i)
									beta_hat_i(*j_it - 1) = beta_star_i( j_it - nbd_i.cbegin() );
							}
									//std::cout<<"beta_hat_i: "<<std::endl<<beta_hat_i<<std::endl;
						//Step 5: Set i-th row and col of Omega equal to Omega_noti_noti*beta_hat_i
							MatRow Omega_noti_noti( SubMatrix<Symmetric::True>(i , Omega) );
									//std::cout<<"Omega_noti_noti: "<<std::endl<<Omega_noti_noti<<std::endl;
							beta_i = Omega_noti_noti * beta_hat_i;
									//std::cout<<"beta_i: "<<std::endl<<beta_i<<std::endl;
					}
					//Plug beta_i into the i-th row / col except from diagonal element
					if(i == 0)//plug in first row (no first element)
						Omega.block(0,1,1,N-1) = beta_i.transpose();
					else if(i == N-1)//plug in last column (no last element)
						Omega.block(0,N-1,N-1,1) = beta_i;
					else{
								//std::cout<<"Omega block 1:"<<std::endl<<Omega.block(0,i,i,1)<<std::endl;
								//std::cout<<"Omega block 2:"<<std::endl<<Omega.block(i,i+1,1,beta_i.size()-i)<<std::endl;
								//std::cout<<"beta_i: "<<std::endl<<beta_i<<std::endl;
						Omega.block(0,i,i,1) 	   			 = beta_i.head(i);
						Omega.block(i,i+1,1,beta_i.size()-i) = beta_i.transpose().tail(beta_i.size()-i);
					}
								//std::cout<<"Omega: "<<std::endl<<Omega<<std::endl;
				}
			//Step 6: Compute the norm of differences
				norm_res = NormType::norm(Omega.selfadjointView<Eigen::Upper>(), Omega_old.selfadjointView<Eigen::Upper>());
				Omega_old = Omega;
			//Step 7: Check stop criteria
					//std::cout<<"Norm res = "<<norm_res<<std::endl;
				if(norm_res < threshold){
					converged = true;
				}
			}
		//Step 8: check if everything is fine
			//std::cout<<"Converged? "<<converged<<std::endl;
			//std::cout<<"#iterazioni = "<<it<<std::endl;
		//Step 9: return K = Omega^-1
			return std::make_tuple(Omega.selfadjointView<Eigen::Upper>().llt().solve(MatRow::Identity(N, N)),converged, it);
			//Omega = Omega.selfadjointView<Eigen::Upper>();
			//return Omega.inverse();
	}

	//------------------------------------------------------------------------------------------------------------------------------------------------------

	//GraphStructure may be GraphType / CompleteViewAdj / CompleteView
	template<template <typename> class GraphStructure = GraphType, typename Type = unsigned int >
	long double log_normalizing_constat(GraphStructure<Type> const & G, double const & b, Eigen::MatrixXd const & D, unsigned int const & MCiteration, unsigned int seed = 0){
		//Typedefs
		using MatRow  	  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		using MatCol      = Eigen::MatrixXd;
		using CholTypeCol = Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
		using iterator    = std::vector<unsigned int>::iterator;
		using citerator   = std::vector<unsigned int>::const_iterator;
		//Check
		if(b <= 2)
			throw std::runtime_error("Shape parameter has to be larger than 2");
		if(D != D.transpose()){
			throw std::runtime_error("Inv_Scale matrix is not symetric");
		}
		CholTypeCol cholD(D);
		if( cholD.info() != Eigen::Success)
			throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
		//Step 1: Preliminaries
		const unsigned int n_links(G.get_n_links());
		const unsigned int N(G.get_size());
		const unsigned int max_n_links(0.5*N*(N-1));
		const unsigned int N_free_elements(n_links + N);
		const long double min_numeric_limits_ld = std::numeric_limits<long double>::min() * 1000;
		long double result_MC{0};
		if(seed == 0)
			seed = std::chrono::system_clock::now().time_since_epoch().count();
		sample::GSL_RNG engine(seed);
		sample::rnorm rnorm;
		//Start
		//nu[i] = #1's in i-th row, from position i+1 up to end. Note that it is also  che number of off-diagonal elements in each row
		std::vector<unsigned int> nu(N);
		#pragma omp parallel for shared(nu)
		for(IdxType i = 0; i < N; ++i){
			std::vector<unsigned int> nbd_i = G.get_nbd(i);
			nu[i] = std::count_if(nbd_i.cbegin(), nbd_i.cend(), [i](const unsigned int & idx){return idx > i;});
		}
		if(n_links == max_n_links){
			long double res_gamma{0};
			for(IdxType i = 0; i < N; ++i){
				res_gamma += std::lgammal( (long double)(0.5*(b + nu[i])) );
			}
			return ( 	(N/2)*utils::log_pi +
						(0.5*N*(b+N-1))*utils::log_2 +
		        	  	res_gamma -
		        	 	0.5*(b+N-1)*std::log(D.determinant())   );
		}
		else if(n_links == 0){

			long double sum_log_diag{0};
			for(IdxType i = 0; i < N; ++i){
				sum_log_diag += std::log( D(i,i) );
			}
			return ( 	0.5*N*b*utils::log_2 +
						N*std::lgamma(0.5*b) -
						(0.5*b) *sum_log_diag );
		}
		else{
			//- Compute T = chol(D^-1), T has to be upper diagonal
						//std::cout<<"D = "<<std::endl<<D<<std::endl;
			MatCol T(cholD.solve(Eigen::MatrixXd::Identity(N,N)).llt().matrixU()); //T is colwise because i need to extract its columns
			//- Define H st h_ij = t_ij/t_jj
			MatCol H(MatCol::Zero(N,N)); //H is colwise because i would need to scan its col
			for(unsigned int j = 0; j < N ; ++j)
				H.col(j) = T.col(j) / T(j,j);

						//std::cout<<"H = "<<std::endl<<H<<std::endl;
						//#pragma omp parallel
						//{
							//std::cout<<"Hello parallel"<<std::endl;
						//}
			int thread_id;
			//Qui inizia il ciclo
			#pragma omp parallel for private(thread_id) reduction (+:result_MC)
			for(IdxType iter = 0; iter < MCiteration; ++ iter){
				#ifdef PARALLELEXEC
				thread_id = omp_get_thread_num();
				#endif
				MatRow Psi(MatRow::Zero(N,N));
				//In the end it has to be exp(-1/2 sum( psi_nonfree_ij^2 )). I accumulate the sum of non free elements squared every time they are generated
				double sq_sum_nonfree{0};
				//Step 2: Sampling the free elements
				//- Sample the diagonal elements from chisq(b + nu_i) (or equivalently from a gamma((b+nu_i)/2, 1/2))
				//- Sample the off-diagonal elements from N(0,1)

				//I could avoid to compute the last element because it won't be used for sure but still this way is more clear
				std::vector<double> vector_free_element(N_free_elements);
				unsigned int time_to_diagonal{0};
				citerator it_nu = nu.cbegin();
				for(unsigned int i = 0; i < N_free_elements; ++i){
					if(time_to_diagonal == 0){
						//vector_free_element[i] = std::sqrt(std::gamma_distribution<> (0.5*(b+(*it_nu)),2.0)(engine));
						vector_free_element[i] = std::sqrt(sample::rchisq()(engine, (double)(b+(*it_nu)) ));
					}
					else
						vector_free_element[i] = rnorm(engine);

					if(time_to_diagonal++ == *it_nu){
						time_to_diagonal = 0;
						it_nu++;
					}
				}

				std::vector<double>::const_iterator it_fe = vector_free_element.cbegin();

				if(H.isIdentity()){

					//Step 3: Complete Psi (upper part)
					//- If i==0 -> all null
					Psi(0,0) = *it_fe;
					it_fe++;
					for(unsigned int j = 1; j < N; ++j){
						if(G(0,j) == true){
							Psi(0,j) = *it_fe;
							it_fe++;
						}
					}
							//std::cout<<"Psi dopo i = 0: "<<std::endl<<Psi<<std::endl;

					Psi(1,1) = *it_fe;
					it_fe++;
					for(unsigned int j = 2; j < N; ++j){ //Counting also the diagonal element because they have to be updated too
						if(G(1,j) == false){
							Psi(1,j) = - Psi(0,1)*Psi(0,j)/Psi(1,1);
							sq_sum_nonfree += Psi(1,j)*Psi(1,j);
						}
						else{
							Psi(1,j) = *it_fe;
							it_fe++;
						}
					}

								//std::cout<<"Psi dopo i = 1: "<<std::endl<<Psi<<std::endl;
					//- If i>1 -> formula 2
					for(unsigned int i = 2; i < N-1; ++i){ // i = N is useless
						Eigen::VectorXd S = Psi.block(0,i,i,1);
						for(unsigned int j = i; j < N; ++j){ //Counting also the diagonal element because they have to be updated too
							if(G(i,j) == false){
								Eigen::VectorXd S_star = Psi.block(0,j,i,1);
								Psi(i,j) = - S.dot(S_star)/Psi(i,i) ;
								sq_sum_nonfree += Psi(i,j)*Psi(i,j);
							}
							else{
								Psi(i,j) = *it_fe;
								it_fe++;
							}
						}
								//std::cout<<"Psi dopo i = "<<i<<": "<<std::endl<<Psi<<std::endl;
					}

				}
				else{
					auto CumSum = [&Psi, &H](unsigned int const & a, unsigned int const & b){ //Computes sum_(k in a:b-1)(Psi_ak*H_kb)

						if(a >= Psi.rows() || b >= H.rows())
							throw std::runtime_error("Cum Sum invalid index request");
						if(a>=b)
							throw std::runtime_error("Wrong dimension inserted");
						else if(a == (b-1))
							return Psi(a,b-1)*H(a,b);
						else
							return (Eigen::RowVectorXd(Psi.block(a,a,1,b-a)).dot(Eigen::VectorXd(H.block(a,b,b-a,1))));
					};
					//Step 3: Complete Psi (upper part)
					//- If i==0 -> formula 1
					// Sums is a NxN-1 matrix (probabilmente il numero di righe sarà N-1 ma vabbe) such that Sums(a,b) = CumSum(a,b+1), i.e CumSum(a,b) = Sums(a,b-1)
					Psi(0,0) = *it_fe;
					it_fe++;
					MatRow Sums(MatRow::Zero(N,N-1));
					for(unsigned int j = 1; j < N; ++j){
						Sums(0,j-1) = CumSum(0,j);
						if(G(0,j) == false){
							Psi(0,j) = -Sums(0,j-1);
							sq_sum_nonfree += Psi(0,j)*Psi(0,j);
						}
						else{
							Psi(0,j) = *it_fe;
							it_fe++;
						}
					}
						//std::cout<<"Psi dopo i = 0: "<<std::endl<<Psi<<std::endl;
						//std::cout<<"Sums dopo i = 0: "<<std::endl<<Sums<<std::endl;
					//- If i==1 -> simplified case of formula 2
					Psi(1,1) = *it_fe;
					it_fe++;
					for(unsigned int j = 2; j < N; ++j){
						Sums(1,j-1) = CumSum(1,j);
						if(G(1,j) == false){
							Psi(1,j) = - ( Sums(1,j-1) + (Psi(0,1) + Psi(0,0)*H(0,1))*(Psi(0,j) + Sums(0,j-1))/Psi(1,1) );
							sq_sum_nonfree += Psi(1,j)*Psi(1,j);
						}
						else{
							Psi(1,j) = *it_fe;
							it_fe++;
						}
					}
							//std::cout<<"Psi dopo i = 1: "<<std::endl<<Psi<<std::endl;
					//std::cout<<"Sums dopo i = 1: "<<std::endl<<Sums<<std::endl;
					//- If i>1 -> formula 2
					for(unsigned int i = 2; i < N-1; ++i){
						Psi(i,i) = *it_fe;
						it_fe++;
						Eigen::VectorXd S = Psi.block(0,i,i,1) + Sums.block(0,i-1,i,1); //non cache friendly operation perché Psi e Sums sono per righe e qua sto estraendo delle colonne
						for(unsigned int j = i+1; j < N; ++j){
							Sums(i,j-1) = CumSum(i,j);
							if(G(i,j) == false){
								Eigen::VectorXd S_star = Psi.block(0,j,i,1) + Sums.block(0,j-1,i,1); //non cache friendly operation perché Psi e Sums sono per righe e qua sto estraendo delle colonne
								Psi(i,j) = - ( Sums(i,j-1) + S.dot(S_star)/Psi(i,i) );
								sq_sum_nonfree += Psi(i,j)*Psi(i,j);
							}
							else{
								Psi(i,j) = *it_fe;
								it_fe++;
							}
						}
							//std::cout<<"Psi dopo i = "<<i<<": "<<std::endl<<Psi<<std::endl;
					}
				}
				//Step 4: Compute exp( -0.5 * sum_NonFreeElements(Psi_ij)^2 )
				long double temp = std::exp((long double)(-0.5 * sq_sum_nonfree));
				if(temp < min_numeric_limits_ld)
					temp = min_numeric_limits_ld;
				result_MC += std::exp((long double)(-0.5 * sq_sum_nonfree));
				//result_MC += std::exp(-0.5 * sq_sum_nonfree);

			}

			result_MC /= MCiteration;
							//std::cout<<"result_MC prima del log = "<<result_MC<<std::endl;
			result_MC = std::log(result_MC);
							//std::cout<<"result_MC = "<<result_MC<<std::endl;
			//Step 5: Compute constant term and return
			long double result_const_term{0};


			#pragma parallel for reduction(+:result_const_term)
			for(IdxType i = 0; i < N; ++i){
				result_const_term += (double)nu[i]/2.0*log_2pi +
									 (double)(b+nu[i])/2.0*log_2 +
									 (double)(b+G.get_nbd(i).size())*std::log(T(i,i)) + //Se T è l'identità posso anche evitare di calcolare questo termine
									 std::lgammal((long double)(0.5*(b + nu[i])));
			}//This computation requires the best possible precision because il will generate a very large number
						//std::cout<<"Constant term = "<<result_const_term<<std::endl;
			return result_MC + result_const_term;
		}
	}



	//------------------------------------------------------------------------------------------------------------------------------------------------------

	using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatCol   	= Eigen::MatrixXd;
	using VecCol    = Eigen::VectorXd;
	std::tuple<MatCol, MatCol, VecCol, double, MatRow, std::vector<bool> > 
	SimulateData_Block(unsigned int const & p, unsigned int const & n, unsigned int const & r,
					   MatCol const & BaseMat, std::shared_ptr<const Groups> const & Gr,  unsigned int seed = 0, double const & sparsity = 0.1)
	{
	
		if(seed==0){
		  std::random_device rd;
		  seed=rd();
		}
		sample::GSL_RNG engine(seed);
		sample::rmvnorm rmv; //Covariance parametrization
		MatRow Ip(MatRow::Identity(p,p));
		MatRow Ir(MatRow::Identity(r,r));
		//Graph
		BlockGraphAdj<bool> G(Gr);
		G.fillRandom(sparsity, seed);
				//std::cout<<"G:"<<std::endl<<G<<std::endl;
		//Precision
		MatRow K = utils::rgwish(G.completeview(), 3.0, MatRow::Identity(p,p), seed);
				//std::cout<<"K:"<<std::endl<<K<<std::endl;
		//mu
		VecCol mu(VecCol::Zero(p));
		for(unsigned int i = 0; i < p; ++i)
			mu(i) = sample::rnorm()(engine, 0, 0.1);
				//std::cout<<"mu:"<<std::endl<<mu<<std::endl;
		//tau_eps
		double tau_eps = sample::rnorm()(engine, 100, 0.1);
				//std::cout<<"tau_eps = "<<tau_eps<<std::endl;
		//Beta and data
		MatCol Beta(MatCol::Zero(p,n));
		MatCol data(MatCol::Zero(r,n));
		MatRow Sigma = K.inverse();
		for(unsigned int i = 0; i < n; ++i){
			Beta.col(i) = rmv(engine, mu, Sigma);
			VecCol media(BaseMat*Beta.col(i));
			for(unsigned int j = 0; j < r; ++j){
				data(j,i) = sample::rnorm()(engine, media(j), std::sqrt(1/tau_eps));
			}
		}
				//std::cout<<"Beta:"<<std::endl<<Beta<<std::endl;
				//std::cout<<"data:"<<std::endl<<data<<std::endl;
		
		return std::make_tuple(data, Beta, mu, tau_eps, K, G.get_adj_list());
	}
	std::tuple<MatCol, MatCol, VecCol, double, MatRow, std::vector<bool> > 
	SimulateData_Block(unsigned int const & p, unsigned int const & n, unsigned int const & r, unsigned int const & M, 
					   MatCol const & BaseMat, unsigned int seed = 0, double const & sparsity = 0.1)
	{
		return SimulateData_Block(p, n, r, BaseMat, std::make_shared<const Groups>(M,p), seed, sparsity);
	}
	std::tuple<MatCol, MatCol, VecCol, double, MatRow, std::vector<bool> > 
	SimulateData_Block(unsigned int const & p, unsigned int const & n, unsigned int const & r,
					   MatCol const & BaseMat, BlockGraph<bool> & G,  unsigned int seed = 0)
	{
	
		if(seed==0){
		  std::random_device rd;
		  seed=rd();
		}
		sample::GSL_RNG engine(seed);
		sample::rmvnorm rmv; //Covariance parametrization
		MatRow Ip(MatRow::Identity(p,p));
		MatRow Ir(MatRow::Identity(r,r));

				//std::cout<<"G:"<<std::endl<<G<<std::endl;
		//Precision
		MatRow K = utils::rgwish(G.completeview(), 3.0, MatRow::Identity(p,p), seed);
				//std::cout<<"K:"<<std::endl<<K<<std::endl;
		//mu
		VecCol mu(VecCol::Zero(p));
		for(unsigned int i = 0; i < p; ++i)
			mu(i) = sample::rnorm()(engine, 0, 0.1);
				//std::cout<<"mu:"<<std::endl<<mu<<std::endl;
		//tau_eps
		double tau_eps = sample::rnorm()(engine, 100, 0.1);
				//std::cout<<"tau_eps = "<<tau_eps<<std::endl;
		//Beta and data
		MatCol Beta(MatCol::Zero(p,n));
		MatCol data(MatCol::Zero(r,n));
		MatRow Sigma = K.inverse();
		for(unsigned int i = 0; i < n; ++i){
			Beta.col(i) = rmv(engine, mu, Sigma);
			VecCol media(BaseMat*Beta.col(i));
			for(unsigned int j = 0; j < r; ++j){
				data(j,i) = sample::rnorm()(engine, media(j), std::sqrt(1/tau_eps));
			}
		}
				//std::cout<<"Beta:"<<std::endl<<Beta<<std::endl;
				//std::cout<<"data:"<<std::endl<<data<<std::endl;
		
		return std::make_tuple(data, Beta, mu, tau_eps, K, G.get_adj_list());
	}
	std::tuple<MatCol, MatCol, VecCol, double, MatRow, std::vector<bool> > 
	SimulateData_Complete(unsigned int const & p, unsigned int const & n, unsigned int const & r,
					  	  MatCol const & BaseMat, unsigned int seed = 0, double const & sparsity = 0.1)
	{

		if(seed==0){
		  std::random_device rd;
		  seed=rd();
		}
		sample::GSL_RNG engine(seed);
		sample::rmvnorm rmv; //Covariance parametrization
		MatRow Ip(MatRow::Identity(p,p));
		MatRow Ir(MatRow::Identity(r,r));
		//Graph
		GraphType<bool> G(p);
		G.fillRandom(sparsity, seed);
				//std::cout<<"G:"<<std::endl<<G<<std::endl;
		//Precision
		MatRow K = utils::rgwish(G.completeview(), 3.0, MatRow::Identity(p,p), seed);
				//std::cout<<"K:"<<std::endl<<K<<std::endl;
		//mu
		VecCol mu(VecCol::Zero(p));
		for(unsigned int i = 0; i < p; ++i)
			mu(i) = sample::rnorm()(engine, 0, 0.1);
				//std::cout<<"mu:"<<std::endl<<mu<<std::endl;
		//tau_eps
		double tau_eps = sample::rnorm()(engine, 100, 0.1);
				//std::cout<<"tau_eps = "<<tau_eps<<std::endl;
		//Beta and data
		MatCol Beta(MatCol::Zero(p,n));
		MatCol data(MatCol::Zero(r,n));
		MatRow Sigma = K.inverse();
		for(unsigned int i = 0; i < n; ++i){
			Beta.col(i) = rmv(engine, mu, Sigma);
			VecCol media(BaseMat*Beta.col(i));
			for(unsigned int j = 0; j < r; ++j){
				data(j,i) = sample::rnorm()(engine, media(j), std::sqrt(1/tau_eps));
			}
		}
				//std::cout<<"Beta:"<<std::endl<<Beta<<std::endl;
				//std::cout<<"data:"<<std::endl<<data<<std::endl;
		
		return std::make_tuple(data, Beta, mu, tau_eps, K, G.get_adj_list());
	}
	std::tuple<MatCol, MatCol, VecCol, double, MatRow, std::vector<bool> > 
	SimulateData_Complete(unsigned int const & p, unsigned int const & n, unsigned int const & r,
					  	  MatCol const & BaseMat, GraphType<bool> & G, unsigned int seed = 0)
	{

		if(seed==0){
		  std::random_device rd;
		  seed=rd();
		}
		sample::GSL_RNG engine(seed);
		sample::rmvnorm rmv; //Covariance parametrization
		MatRow Ip(MatRow::Identity(p,p));
		MatRow Ir(MatRow::Identity(r,r));
		
				//std::cout<<"G:"<<std::endl<<G<<std::endl;
		//Precision
		MatRow K = utils::rgwish(G.completeview(), 3.0, MatRow::Identity(p,p), seed);
				//std::cout<<"K:"<<std::endl<<K<<std::endl;
		//mu
		VecCol mu(VecCol::Zero(p));
		//for(unsigned int i = 0; i < p; ++i)
			//mu(i) = sample::rnorm()(engine, 0, 0.1);
				//std::cout<<"mu:"<<std::endl<<mu<<std::endl;
		//tau_eps
		double tau_eps = sample::rnorm()(engine, 100, 0.1);
				//std::cout<<"tau_eps = "<<tau_eps<<std::endl;
		//Beta and data
		MatCol Beta(MatCol::Zero(p,n));
		MatCol data(MatCol::Zero(r,n));
		MatRow Sigma = K.inverse();
		for(unsigned int i = 0; i < n; ++i){
			Beta.col(i) = rmv(engine, mu, Sigma);
			VecCol media(BaseMat*Beta.col(i));
			for(unsigned int j = 0; j < r; ++j){
				data(j,i) = sample::rnorm()(engine, media(j), std::sqrt(1/tau_eps));
			}
		}
				//std::cout<<"Beta:"<<std::endl<<Beta<<std::endl;
				//std::cout<<"data:"<<std::endl<<data<<std::endl;
		
		return std::make_tuple(data, Beta, mu, tau_eps, K, G.get_adj_list());
	}


}


	




#endif
