#ifndef __GSLWRAPPERS_H__
#define __GSLWRAPPERS_H__

#include "include_headers.h"
//GSL
#include <gsl/gsl_rng.h>     //For random number generators
#include <gsl/gsl_randist.h> //For random variates and probability density functions
#include <gsl/gsl_cdf.h> 	 //For cumulative density functions
#include <gsl/gsl_bspline.h> //For spline operations
#include <gsl/gsl_linalg.h>


//Della normale MV ci sta fare due versioni, una che prende la precisione e ne calcola dentro il chol 
//e una che prende direttamente chol ma inteso come cholType e non gia la matrice upper

namespace sample{

	using MatRow 		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatCol 		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using VecRow 		= Eigen::RowVectorXd;
	using VecCol 		= Eigen::VectorXd;
	using CholTypeRow 	= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol	= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	enum class isChol
	{
		Upper, Lower, False
	};


	//This function simply wraps in c++ code the construction and desctruction of a gsl_rng obj
	class GSL_RNG{ 
		public:
			GSL_RNG(unsigned int const & _seed){
				gsl_rng_env_setup();
				r = gsl_rng_alloc(gsl_rng_default);
				seed = _seed;
				gsl_rng_set(r,_seed);
			}
			GSL_RNG(){
				gsl_rng_env_setup();
				r = gsl_rng_alloc(gsl_rng_default);
				seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
				gsl_rng_set(r,seed);
			}
			~GSL_RNG(){
				gsl_rng_free(r);
			}
			void print_info()const{
				printf ("generator type: %s\n", gsl_rng_name(r));
				std::cout<<"seed = "<<seed<<std::endl;
				printf ("first value = %lu\n", gsl_rng_get(r));
			}
			gsl_rng* operator()()const{
				return r;
			}
			inline void set_seed(unsigned int const & s){
				seed = s;
				gsl_rng_set(r,seed);
			}
			inline unsigned int get_seed() const{
				return seed;
			}
			//It is public
			gsl_rng * r;
		private:
			unsigned int seed;
	};

	struct rnorm
	{
		//Gets standard deviation!
		double operator()(GSL_RNG const & engine, double const & mean, double const & sd)const{
			return gsl_ran_gaussian_ziggurat(engine(),sd) + mean;
		}	
		double operator()(GSL_RNG const & engine)const{
			return gsl_ran_gaussian_ziggurat(engine(), 1.0);
		}
		double operator()()const{
			return gsl_ran_gaussian_ziggurat(GSL_RNG ()(),1.0); //the first () is for the constructor, the second il for the call operator
		}
	};
	 
	struct rgamma{
		double operator()(GSL_RNG const & engine, double const & shape, double const & scale)const{
			return gsl_ran_gamma(engine(),shape,scale);
		}
		double operator()(double const & shape, double const & scale)const{
			return gsl_ran_gamma(GSL_RNG ()(),shape,scale);
		}
	};

	struct rchisq{
		double operator()(GSL_RNG const & engine, double const & k)const{
			return gsl_ran_chisq(engine(),k);
		}
		double operator()(double const & k)const{
			return gsl_ran_chisq(GSL_RNG ()(), k);
		}
	};


	template<isChol isCholType = isChol::False>
	struct rmvnorm_prec{
		template<typename EigenType>
		VecCol operator()(GSL_RNG const & engine, VecCol const & mean, EigenType const & Prec)
		{
			static_assert(isCholType == isChol::False || 
						  isCholType == isChol::Upper ||
						  isCholType == isChol::Lower ,
						  "Error, invalid sample::isChol field inserted. It has to be equal to Upper, Lower or False");

			if(Prec.rows() != Prec.cols() )
				throw std::runtime_error("Error, precision matrix is not squared");
			if(mean.size() != Prec.cols())
				throw std::runtime_error("Error, dimensions of mean vector and precision matrix are not compatible");

			MatRow A(MatRow::Identity(mean.size(), mean.size())); //Deve essere upper triangular
			if constexpr( isCholType == isChol::False){
					//std::cout<<"Non è chol"<<std::endl;
				MatRow U(MatRow::Identity(Prec.rows(), Prec.cols()));
				if constexpr(std::is_same_v< EigenType, MatRow>){
					CholTypeRow chol(Prec);
					if(chol.info() != Eigen::Success)
						throw std::runtime_error("Error, the precision matrix is not symmetric positive definite");
					U = chol.matrixU();
							//std::cout<<"U: "<<std::endl<<U<<std::endl;
				}
				else{
					CholTypeCol chol(Prec);
					if(chol.info() != Eigen::Success)
						throw std::runtime_error("Error, the precision matrix is not symmetric positive definite");
					U = chol.matrixU();
				}
				A = U.triangularView<Eigen::Upper>().solve(MatRow::Identity(Prec.rows(), Prec.cols())); //U is upper trinagular, this is a backsolve
			}
			else if constexpr(isCholType == isChol::Upper){
						//std::cout<<"è chol Upper"<<std::endl;
				MatRow U(Prec);
						//std::cout<<"U: "<<std::endl<<U<<std::endl;
				A = U.triangularView<Eigen::Upper>().solve(MatRow::Identity(Prec.rows(), Prec.cols())); 
			}
			else if constexpr(isCholType == isChol::Lower){
						//std::cout<<"è chol Lower"<<std::endl;
				MatRow U(Prec.transpose());
						//std::cout<<"U: "<<std::endl<<U<<std::endl;
				A = U.triangularView<Eigen::Upper>().solve(MatRow::Identity(Prec.rows(), Prec.cols())); 	
			}
						//std::cout<<"A: "<<std::endl<<A<<std::endl;
			

			VecCol z(VecCol::Zero(Prec.cols()));
			#pragma omp parallel for shared(z) // non sta andando
			for(unsigned int i = 0; i < Prec.cols(); ++i)
				z(i) = rnorm()(engine);

					//std::cout<<"z: "<<std::endl<<z<<std::endl;
			return mean + A*z;
		}
		template<typename EigenType>
		VecCol operator()(VecCol const & mean, EigenType const & Prec){
			return rmvnorm_prec<isCholType>()(GSL_RNG (), mean, Prec);
		}
	};

	struct rmvnorm{
		template<typename EigenType>
		VecCol operator()(GSL_RNG const & engine, VecCol & mean, EigenType & Cov)
		{
			if(Cov.rows() != Cov.cols() )
				throw std::runtime_error("Error, covariance matrix is not squared");
			if(mean.size() != Cov.cols())
				throw std::runtime_error("Error, dimensions of mean vector and precision matrix are not compatible");
			//Map in GSL matrix
			gsl_matrix * V 	    = gsl_matrix_calloc(Cov.rows(), Cov.rows());
			gsl_matrix *cholMat = gsl_matrix_alloc (Cov.rows(), Cov.rows());
			gsl_vector *mu  	= gsl_vector_alloc (Cov.rows());
			gsl_vector *result  = gsl_vector_alloc (Cov.rows());
			
			V->data  = Cov.data();
			mu->data = mean.data();
			//std::cout<<"V "<<std::endl;
			//for(auto i = 0; i < Cov.rows(); ++i){
				//for(auto j = 0; j < Cov.rows(); ++j){
					//std::cout<<gsl_matrix_get(V,i,j)<<" ";
				//}
				//std::cout<<std::endl;	
			//}
			//Chol decomposition
			gsl_matrix_memcpy(cholMat, V);
			gsl_linalg_cholesky_decomp1(cholMat);
			//Sample with GSL
			gsl_ran_multivariate_gaussian(engine(), mu, cholMat, result);
			//Map back in Eigen form
			Eigen::Map<VecCol> temp(&(result->data[0]), Cov.rows()); 
			//Qua si crea una situazione un po' paradossale. result viene messa dentro temp nel modo corretto MA quando libero result poi non posso ritornare la matrice
			//perché alcuni valori si perdono. sono costretto a fare una copia e dare quella.
			VecCol ReturnVect = temp;
			//Free and return
			gsl_matrix_free(cholMat);
			gsl_vector_free(result);
			gsl_vector_free(mu);
			gsl_matrix_free(V);
			return ReturnVect;	 
		}
		template<typename EigenType>
		VecCol operator()(VecCol & mean, EigenType & Cov)
		{
			return rmvnorm()(GSL_RNG (), mean, Cov);
		}
	};


	

	
	template<typename RetType = MatCol>
	struct rwish_old{
		template<typename EigenType>
		RetType operator()( GSL_RNG const & engine, double const & b, EigenType &Psi )const
		{
			if(Psi.rows() != Psi.cols())
				throw std::runtime_error("Non squared matrix inserted");
			//Map in GSL matrix
			gsl_matrix * V 	    = gsl_matrix_calloc(Psi.rows(), Psi.rows());
			gsl_matrix *cholMat = gsl_matrix_alloc (Psi.rows(), Psi.rows());
			gsl_matrix *result  = gsl_matrix_alloc (Psi.rows(), Psi.rows());
			gsl_matrix * work   = gsl_matrix_calloc(Psi.rows(), Psi.rows());
			
			V->data = Psi.data();
			//std::cout<<"V "<<std::endl;
			//for(auto i = 0; i < Psi.rows(); ++i){
				//for(auto j = 0; j < Psi.rows(); ++j){
					//std::cout<<gsl_matrix_get(V,i,j)<<" ";
				//}
				//std::cout<<std::endl;	
			//}
			//Chol decomposition
			gsl_matrix_memcpy(cholMat, V);
			gsl_linalg_cholesky_decomp1(cholMat);
			//Sample with GSL
			gsl_ran_wishart(engine(), b+Psi.rows()-1, cholMat, result, work);
			//Map back in Eigen form
			Eigen::Map<RetType> temp(&(result->data[0]), Psi.rows(), Psi.rows()); 
			//Qua si crea una situazione un po' paradossale. result viene messa dentro temp nel modo corretto MA quando libero result poi non posso ritornare la matrice
			//perché alcuni valori si perdono. sono costretto a fare una copia e dare quella.
			RetType ReturnMatrix = temp;
			//Free and return
			gsl_matrix_free(cholMat);
			gsl_matrix_free(result);
			gsl_matrix_free(work);
			gsl_matrix_free(V);
			return ReturnMatrix;	  	
		}
		template<typename EigenType>
		RetType operator()(double const & b, EigenType &Psi)const{
			return rwish_old<RetType>()(GSL_RNG (), b,Psi);
		}
			
	};
	

	//What is better? 
	//The best results are obtained when the input is ColMajor. In this case return type is not influent
	//Results are slightly worse if the input is RowMajor, worst case is Row input and Col output.
	//What about CholType? CholLower is optimal with RowMajor input and CholUpper is optimal with ColMajor input.
	//In the suboptimal case a temporary obj is needed to change the storage order which causes a little overhead.
	template<typename RetType = MatCol, isChol isCholType = isChol::False>
	struct rwish{
		template<typename EigenType>
		RetType operator()( GSL_RNG const & engine, double const & b, EigenType &Psi )const
		{	
			static_assert(isCholType == isChol::False || 
						  isCholType == isChol::Upper ||
						  isCholType == isChol::Lower ,
						  "Error, invalid sample::isChol field inserted. It has to be equal to Upper, Lower or False");
			if(Psi.rows() != Psi.cols())
				throw std::runtime_error("Non squared matrix inserted");

			//Declaration of gsl objects
			gsl_matrix *cholMat = gsl_matrix_alloc (Psi.rows(), Psi.rows());
			gsl_matrix *result  = gsl_matrix_alloc (Psi.rows(), Psi.rows());
			gsl_matrix * work   = gsl_matrix_calloc(Psi.rows(), Psi.rows());
			

			//auto start = std::chrono::high_resolution_clock::now();
			
			if constexpr( isCholType == isChol::False){
				//Map in GSL matrix
				gsl_matrix * V 	    = gsl_matrix_calloc(Psi.rows(), Psi.rows());
				V->data = Psi.data();
				//Chol decomposition
				gsl_matrix_memcpy(cholMat, V);
				gsl_linalg_cholesky_decomp1(cholMat);
				gsl_matrix_free(V); //Free V
			}
			else if constexpr(isCholType == isChol::Upper){
				//Psi has to be ColMajor
				if constexpr( std::is_same_v< EigenType, MatRow> ){
					MatCol PsiCol(Psi);
					for(int i = 0; i < Psi.rows()*Psi.rows(); ++i)
						cholMat->data[i] = PsiCol.data()[i];
				}
				else{
					cholMat->data = Psi.data();
				}
				
			}
			else if constexpr(isCholType == isChol::Lower){ 
				//Psi has to be RowMajor
				if constexpr( std::is_same_v< EigenType, MatCol> ){
					MatRow PsiRow(Psi);
					for(int i = 0; i < Psi.rows()*Psi.rows(); ++i) //In questo caso devo essere esplicito. buffer rovinato?? non so ma mi salta la prima riga altrimenti
						cholMat->data[i] = PsiRow.data()[i];
				}
				else{
					cholMat->data = Psi.data();
				}
			} 		
							//std::cout<<"cholMat "<<std::endl;
			 				//for(auto i = 0; i < Psi.rows(); ++i){
			 					//for(auto j = 0; j < Psi.rows(); ++j){
			 						//std::cout<<gsl_matrix_get(cholMat,i,j)<<" ";
			 					//}
			 				//std::cout<<std::endl;	
			 				//}	
			//auto stop = std::chrono::high_resolution_clock::now();
			//std::chrono::duration<double, std::milli> timer = stop - start;
			//std::cout << "Tempo per creare cholMat :  " << timer.count()<<" ms"<< std::endl;
 			
			//Sample with GSL
			gsl_ran_wishart(engine(), b+Psi.rows()-1, cholMat, result, work);
			//Map back in Eigen form
			Eigen::Map<RetType> temp(&(result->data[0]), Psi.rows(), Psi.rows()); 

			//Qua si crea una situazione un po' paradossale. result viene messa dentro temp nel modo corretto MA quando libero result poi non posso ritornare la matrice
			//perché alcuni valori si perdono. sono costretto a fare una copia e dare quella.
			RetType ReturnMatrix = temp;
			
			//Free and return
			gsl_matrix_free(cholMat);
			gsl_matrix_free(result);
			gsl_matrix_free(work);
			
			return ReturnMatrix;	  	
		}
		template<typename EigenType>
		RetType operator()(double const & b, EigenType &Psi)const{
			return rwish<RetType, isCholType>()(GSL_RNG (), b,Psi);
		}
			
	};
	//Both input and output Type are template parameters. Here is some usage example:
	
	//1) Create explicitely the object before (ColMajor by default)
		//sample::rwish<MatCol, sample::isChol::Lower>   WishC;
		//sample::rwish<MatRow, sample::isChol::Upper>   WishR;
		//res_GSL_col = WishC(engine_gsl, 3, D);
		//res_GSL_row = WishR(engine_gsl, 3, D);

	//2) Create the callable object at fly but specifing the type
		//res_GSL_row = sample::rwish<MatRow, sample::isChol::False>()(engine_gsl, 3, D);

	//3) Creation at fly exploiting default value type
		//res_GSL_col = sample::rwish()(engine_gsl, 3, D);



}



//Per il momento non è un template perché immagino che questa scelta si faccia una volta e basta
namespace spline{
	using MatRow  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatCol  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using MatType = MatCol;

	// It generates n_basis B-splines of order equal to order in the interval [a,b]. 
	// This function assumes uniformly spaced breakpoints on [a,b] and constructs the corresponding knot vector 
	// using a number of breaks equal to n_basis + 2 - order
	// Set order=4 for cubic splines.
	// grid_points is a vector containing the points where the splines has to be evaluated. It then returns a matrix of dimension
	// grid_points.size() x n_basis. This means that r-th rows contains all the spline computed in grid_points[r] and j-th column
	// contains the j-th spline evaluated in all points

	MatType generate_design_matrix(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  std::vector<double> const & grid_points);
	//Same as before but assumes that the grid points are r uniformly space points in [a,b], assuming the first to be equal to a and the last equal to b
	MatType generate_design_matrix(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  unsigned int const & r);
	// It returns a vector of length nderiv+1, each element is a grid_points.size() x n_basis matrix, the k-th element is the evaluation of 
	// all the k-th derivatives of all the splines in all the grid points. 
	// The first elemenst is the design matrix
	std::vector<MatType> evaluate_spline_derivative(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  				   std::vector<double> const & grid_points, unsigned int const & nderiv);
}



#endif
