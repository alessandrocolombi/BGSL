#ifndef __GSLWRAPPERS_H__
#define __GSLWRAPPERS_H__

#include "include_headers.h"
//GSL
#include <gsl/gsl_rng.h>     //For random number generators
#include <gsl/gsl_randist.h> //For random variates and probability density functions
#include <gsl/gsl_cdf.h> 	 //For cumulative density functions
#include <gsl/gsl_bspline.h> //For spline operations
#include <gsl/gsl_linalg.h>

//Important note:
// rmvnorm_prec / rmvnorm / rwish_old / rwish functions access directly to data() buffer of eigen matrix. This implies that the matrix passed as input
// cannot be a temporary object constructed inside the function call.
// Things that works:
// MatCol Icol = MatCol::Identity(p,p); res = sample::rmvnorm()(engine, mean, Irow); -> OK
// Thing that does not work:
// res = sample::rmvnorm()(engine, mean, MatCol::Identity(p,p)); -> NO
// res = sample::rmvnorm()(engine, mean, A*B); -> NO


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
	//Had to remove std::random_device because there is a bug when compiling in window (returs always the same value).
	//std::chrono is fine but seed are very similar when generated in the sampler, but at least they are different. Would be nice to find a solution generating more entropy
	// reference for bug -> https://en.cppreference.com/w/cpp/numeric/random/random_device, https://sourceforge.net/p/mingw-w64/bugs/338/
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
				//std::random_device rd;
				//seed=rd();
				seed = static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count());
				std::seed_seq seq = {seed};
				std::vector<unsigned int> seeds(1);
				seq.generate(seeds.begin(), seeds.end());
				seed = seeds[0];
				gsl_rng_set(r,seed);
				this->print_info();
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
		double operator()(double const & mean, double const & sd)const{
			return rnorm()(GSL_RNG (), mean, sd);
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

	struct rmvnorm_not_template{
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
			return rmvnorm_not_template()(GSL_RNG (), mean, Cov);
		}
	};

	//Quella vecchia, facevo ancora la copia del risultato. Da tenere come esempio
	template<isChol isCholType = isChol::False>
	struct rmvnorm_old{
		template<typename EigenType>
		VecCol operator()(GSL_RNG const & engine, VecCol mean, EigenType const & Cov)
		{
			static_assert(isCholType == isChol::False || 
						  isCholType == isChol::Upper ||
						  isCholType == isChol::Lower ,
						  "Error, invalid sample::isChol field inserted. It has to be equal to Upper, Lower or False");
			if(Cov.rows() != Cov.cols())
				throw std::runtime_error("Non squared matrix inserted");

			//Declaration of gsl objects
				gsl_matrix *cholMat = gsl_matrix_alloc (Cov.rows(), Cov.rows());
				gsl_vector *result  = gsl_vector_alloc (Cov.rows());
				gsl_vector *mu  	= gsl_vector_alloc (Cov.rows());
				mu->data = mean.data();
			//auto start = std::chrono::high_resolution_clock::now();
			if(Cov.isIdentity()){
				gsl_matrix_set_identity(cholMat);
			}
			else {
				if constexpr( isCholType == isChol::False){
				//Map in GSL matrix
				gsl_matrix * V 	= gsl_matrix_calloc(Cov.rows(), Cov.rows());
				for(int i = 0; i < Cov.rows()*Cov.rows(); ++i)
					V->data[i] = Cov.data()[i];
				//V->data = Cov.data();
				//Chol decomposition
				gsl_matrix_memcpy(cholMat, V);
				gsl_linalg_cholesky_decomp1(cholMat);
				gsl_matrix_free(V); //Free V
				}
				else if constexpr(isCholType == isChol::Upper){
					//Cov has to be ColMajor
					if constexpr( std::is_same_v< EigenType, MatRow> ){
					MatCol CovCol(Cov);
					for(int i = 0; i < Cov.rows()*Cov.rows(); ++i)
						cholMat->data[i] = CovCol.data()[i];
					}
					else{
						for(int i = 0; i < Cov.rows()*Cov.rows(); ++i)
							cholMat->data[i] = Cov.data()[i];
						//cholMat->data = Cov.data();
					}
				}
				else if constexpr(isCholType == isChol::Lower){ 
				//Cov has to be RowMajor
					if constexpr( std::is_same_v< EigenType, MatCol> ){
						MatRow CovRow(Cov);
						for(int i = 0; i < Cov.rows()*Cov.rows(); ++i) //In questo caso devo essere esplicito. buffer rovinato?? non so ma mi salta la prima riga altrimenti
							cholMat->data[i] = CovRow.data()[i];
						}
						else{
							for(int i = 0; i < Cov.rows()*Cov.rows(); ++i)
								cholMat->data[i] = Cov.data()[i];
							//cholMat->data = Cov.data();
						}
				} 		
			}
			gsl_ran_multivariate_gaussian(engine(), mu, cholMat, result);
			//Map back in Eigen form
			Eigen::Map<VecCol> temp(&(result->data[0]), Cov.rows()); 
			//std::cout<<"temp:"<<std::endl<<temp<<std::endl;
			//Qua si crea una situazione un po' paradossale. result viene messa dentro temp nel modo corretto MA quando libero result poi non posso ritornare la matrice
			//perché alcuni valori si perdono. sono costretto a fare una copia e dare quella.
			//Anche se facico una soluzione come nella nuova funzione, ovvero mettendo result->data = nullptr, non possono comunque ritornare temp. ho lo stesso problema dei primi due elementi invalidati.
			VecCol ReturnVect = temp;
			//Free and return
			gsl_matrix_free(cholMat);
			gsl_vector_free(result);
			gsl_vector_free(mu);
			return ReturnVect;	 
		}
		template<typename EigenType>
		VecCol operator()(VecCol & mean, EigenType & Cov)
		{
			return rmvnorm_old()(GSL_RNG (), mean, Cov);
		}
	};


	template<isChol isCholType = isChol::False>
	struct rmvnorm{
		template<typename EigenType>
		VecCol operator()(GSL_RNG const & engine, VecCol & mean, EigenType & Cov)
		{
			//Cov has to be symmetric. Not checked for efficiency
			static_assert(isCholType == isChol::False || 
						  isCholType == isChol::Upper ||
						  isCholType == isChol::Lower ,
						  "Error, invalid sample::isChol field inserted. It has to be equal to Upper, Lower or False");
			if(Cov.rows() != Cov.cols())
				throw std::runtime_error("Non squared matrix inserted");
			if(Cov.rows() != mean.size())
				throw std::runtime_error("Matrix and mean do not have compatible size");
	
			//Declare return object
			/* The simplest way to proceed is to create a gsl_vector for storing the sampled values and then to copy it into an eigen vector. If doing this way the copy cannot be avoided 
			   even if Eigen::Map is used. Indeed Eigen map wraps the gsl_vector buffer of data to be an eigen object but when the vector is returned, the gsl_vector has to be freed and the
			   buffer is corrupted, returning the wrong result.
			   What is implemented below is different, first the Eigen object is allocated and then the gsl_vector is defined such that it writes on the same buffer of the Eigen vector. No copies
			   are done before returning.
			   Note that the gsl_vector is not defined by means of gsl_vector_alloc(). Indeed that function allocates some space in memory that is lost when result.data is set to be the same
			   of return_obj, which of course generates a memory leak. This happens because the gsl_vector does not own the buffer and do not deallocate that space by calling gsl_vector_free(). */
			VecCol return_obj(VecCol::Zero(mean.size()));		//Eigen obj that has to be returned
			gsl_vector result;									//gsl_vector where the sampled values will be stored. 
			result.size   = mean.size();						//size of the vector
			result.stride = 1;									//how close in memory the elements of result.data are. 
			result.owner  = 0;									//result does not own the buffer where data are stored 
			result.data   = return_obj.data();					//set data buffer of gsl_vector to the exactly the same of the Eigen vector. 
															 	//From now on they share the same buffer, writing on result.data is like writing on return_obj.data() and viceversa
			
			gsl_matrix *cholMat = gsl_matrix_alloc (Cov.rows(), Cov.rows()); //gsl_ran_multivariate_gaussian requires the Cholesky decompoition and has to be a gsl_matrix
			gsl_vector mu;									
			mu.size   = mean.size();						
			mu.stride = 1;									
			mu.owner  = 0;									
			mu.data   = mean.data();					
			// Declaration of some Eigen quantities that may be needed. They have to be declared here because they will share their buffer with cholMat 
			// and if they go out of scope, the buffer is corrupted.
			MatRow Chol_cov;
			MatRow TCov_row;
			MatCol TCov_col;
			//auto start = std::chrono::high_resolution_clock::now();
			if(Cov.isIdentity()){
				//std::cout<<"Identity"<<std::endl;
				gsl_matrix_set_identity(cholMat);
			}
			else {
				if constexpr( isCholType == isChol::False)
				{
					//std::cout<<"Not chol"<<std::endl;
					Chol_cov = Cov.llt().matrixL(); //Use Eigen factorization because Cov is passed by ref and it is not const. If the gsl version is used, Cov is modified!
					cholMat->data = Chol_cov.data();										
					//gsl_linalg_cholesky_decomp1(cholMat); //gsl for cholesky decoposition
				}
				else if constexpr(isCholType == isChol::Upper){
					if constexpr( std::is_same_v< EigenType, MatRow> ){
						//std::cout<<"RowMajor + Upper"<<std::endl;
						TCov_row = Cov.transpose(); //Do not use Cov.transposeInPlace() otherwise Cov is modified.
						cholMat->data = TCov_row.data();
					}
					else{
						//std::cout<<"ColMajor + Upper"<<std::endl;	
						cholMat->data = Cov.data();
					}
				}
				else if constexpr(isCholType == isChol::Lower){ 
					if constexpr( std::is_same_v< EigenType, MatCol> ){
						//std::cout<<"ColMajor + Lower"<<std::endl;	
						TCov_col = Cov.transpose(); //Do not use Cov.transposeInPlace() otherwise Cov is modified.
						cholMat->data = TCov_col.data();
					}
					else{
						//std::cout<<"RowMajor + Lower"<<std::endl;	
						cholMat->data = Cov.data();
					}
				} 		
			}

							//std::cout<<"cholMat "<<std::endl;
			 				//for(auto i = 0; i < cholMat->size1; ++i){
			 					//for(auto j = 0; j < cholMat->size1; ++j){
			 						//std::cout<<gsl_matrix_get(cholMat,i,j)<<" ";
			 					//}
			 				//std::cout<<std::endl;	
			 				//}	
			gsl_ran_multivariate_gaussian(engine(), &mu, cholMat, &result);
					//std::cout<<"finito, stampo result"<<std::endl;
					//for(int i = 0; i < result.size; ++i)
						//std::cout<<gsl_vector_get(&result, i)<<", ";
				//std::cout<<std::endl;
			//Free and return
			gsl_matrix_free(cholMat);
			return return_obj;	
			//Runnig with valgrind_memcheck:
			/*
			==4546== HEAP SUMMARY:
			==4546==     in use at exit: 0 bytes in 0 blocks
			==4546==   total heap usage: 26 allocs, 26 frees, 80,000 bytes allocated
			==4546== 
			==4546== All heap blocks were freed -- no leaks are possible
			*/ 
		}
		template<typename EigenType>
		VecCol operator()(VecCol & mean, EigenType & Cov)
		{
			return rmvnorm()(GSL_RNG (), mean, Cov);
		}
	};

	template<typename RetType = MatCol>
	struct rwish_non_chol{
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
			return rwish_non_chol<RetType>()(GSL_RNG (), b,Psi);
		}
			
	};
	//What is better? 
	//The best results are obtained when the input is ColMajor. In this case return type is not influent
	//Results are slightly worse if the input is RowMajor, worst case is Row input and Col output.
	//What about CholType? CholLower is optimal with RowMajor input and CholUpper is optimal with ColMajor input.
	//In the suboptimal case a temporary obj is needed to change the storage order which causes a little overhead.
	template<typename RetType = MatCol, isChol isCholType = isChol::False>
	struct rwish_old{
		template<typename EigenType>
		RetType operator()( GSL_RNG const & engine, double const & b, EigenType &Psi )const
		{	
			static_assert(isCholType == isChol::False || 
						  isCholType == isChol::Upper ||
						  isCholType == isChol::Lower ,
						  "Error, invalid sample::isChol field inserted. It has to be equal to utils::isChol::Upper, utils::isChol::Lower or utils::isChol::False");
			if(Psi.rows() != Psi.cols())
				throw std::runtime_error("Non squared matrix inserted");

			//Declaration of gsl objects
			gsl_matrix *cholMat = gsl_matrix_calloc (Psi.rows(), Psi.rows());
			gsl_matrix *result  = gsl_matrix_alloc  (Psi.rows(), Psi.rows());
			gsl_matrix *work    = gsl_matrix_calloc (Psi.rows(), Psi.rows());
			
			// Declaration of some Eigen quantities that may be needed. They have to be declared here because they will share their buffer with cholMat 
			// and if they go out of scope, the buffer is corrupted.
			MatRow Chol_psi;
			MatRow Tpsi_row;
			MatCol Tpsi_col;

			//auto start = std::chrono::high_resolution_clock::now();
			if(Psi.isIdentity()){
				gsl_matrix_set_identity(cholMat);
			}
			else {
				if constexpr( isCholType == isChol::False)
				{
					//std::cout<<"Not chol"<<std::endl;
					Chol_psi = Psi.llt().matrixL(); //Use Eigen factorization because Cov is passed by ref and it is not const. If the gsl version is used, Cov is modified!
					cholMat->data = Chol_psi.data();										
					//gsl_linalg_cholesky_decomp1(cholMat); //gsl for cholesky decoposition

												////Map in GSL matrix
												//gsl_matrix * V 	= gsl_matrix_calloc(Psi.rows(), Psi.rows());
																	////for(int i = 0; i < Psi.rows()*Psi.rows(); ++i)
																		////V->data[i] = Psi.data()[i];
												//V->data = Psi.data();
												////Chol decomposition
												//gsl_matrix_memcpy(cholMat, V);
												//gsl_linalg_cholesky_decomp1(cholMat);
												//gsl_matrix_free(V); //Free V	
				}
				else if constexpr(isCholType == isChol::Upper)
				{
					if constexpr( std::is_same_v< EigenType, MatRow> ){
						std::cout<<"RowMajor + Upper"<<std::endl;
						Tpsi_row = Psi.transpose(); //Do not use Psi.transposeInPlace() otherwise Psi is modified.
						cholMat->data = Tpsi_row.data();
												//MatCol PsiCol(Psi);
												//for(int i = 0; i < Psi.rows()*Psi.rows(); ++i)
													//cholMat->data[i] = PsiCol.data()[i];
					}
					else{
						std::cout<<"ColMajor + Upper"<<std::endl;
						cholMat->data = Psi.data();
					}
				}
				else if constexpr(isCholType == isChol::Lower)
				{ 
					if constexpr( std::is_same_v< EigenType, MatCol> ){
						std::cout<<"ColMajor + Lower"<<std::endl;	
						Tpsi_col = Psi.transpose(); //Do not use Cov.transposeInPlace() otherwise Cov is modified.
						cholMat->data = Tpsi_col.data();
										//MatRow PsiRow(Psi);
										//for(int i = 0; i < Psi.rows()*Psi.rows(); ++i) //In questo caso devo essere esplicito. buffer rovinato?? non so ma mi salta la prima riga altrimenti
											//cholMat->data[i] = PsiRow.data()[i];
					}
					else{
						std::cout<<"ColMajor + Upper"<<std::endl;
						cholMat->data = Psi.data();
					}
				} 		
			}


							//std::cout<<"cholMat "<<std::endl;
			 				//for(auto i = 0; i < cholMat->size1; ++i){
			 					//for(auto j = 0; j < cholMat->size1; ++j){
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
			return rwish_old<RetType, isCholType>()(GSL_RNG (), b,Psi);
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

	template<typename RetType = MatCol, isChol isCholType = isChol::False>
	struct rwish{
		template<typename EigenType>
		RetType operator()( GSL_RNG const & engine, double const & b, EigenType &Psi )const
		{	
			static_assert(isCholType == isChol::False || 
						  isCholType == isChol::Upper ||
						  isCholType == isChol::Lower ,
						  "Error, invalid sample::isChol field inserted. It has to be equal to utils::isChol::Upper, utils::isChol::Lower or utils::isChol::False");
			if(Psi.rows() != Psi.cols())
				throw std::runtime_error("Non squared matrix inserted");
			
			RetType return_obj(RetType::Zero(Psi.rows(), Psi.cols()));		//Eigen obj that has to be returned
			gsl_matrix result;												//gsl_matrix where the sampled values will be stored. 
			result.size1   = Psi.rows();									//row of the matrix
			result.size2   = Psi.cols();									//cols of the matrix
			result.tda 	= Psi.rows();										//it is not a submatrix, so this parameter is equal to the number of rows.
			result.owner  = 0;												//result does not own the buffer where data are stored 
			result.data   = return_obj.data();								//set data buffer of gsl_matrix to be exactly the same of the Eigen matrix. 
															 				//From now on they share the same buffer, writing on result.data is like writing on return_obj.data() and viceversa
			//Declaration of other gsl objects
			gsl_matrix *cholMat = gsl_matrix_calloc(Psi.rows(), Psi.rows());
			gsl_matrix *work    = gsl_matrix_calloc(Psi.rows(), Psi.rows());
			
			// Declaration of some Eigen quantities that may be needed. They have to be declared here because they will share their buffer with cholMat 
			// and if they go out of scope, the buffer is corrupted.
			MatRow Chol_psi;
			MatRow Tpsi_row;
			MatCol Tpsi_col;

			if(Psi.isIdentity()){
				gsl_matrix_set_identity(cholMat);
			}
			else {
				if constexpr( isCholType == isChol::False)
				{
					//std::cout<<"Not chol"<<std::endl;
					Chol_psi = Psi.llt().matrixL(); //Use Eigen factorization because Cov is passed by ref and it is not const. If the gsl version is used, Cov is modified!
					cholMat->data = Chol_psi.data();										
					//gsl_linalg_cholesky_decomp1(cholMat); //gsl for cholesky decoposition
				}
				else if constexpr(isCholType == isChol::Upper)
				{
					if constexpr( std::is_same_v< EigenType, MatRow> ){
						//std::cout<<"RowMajor + Upper"<<std::endl;
						Tpsi_row = Psi.transpose(); //Do not use Psi.transposeInPlace() otherwise Psi is modified.
						cholMat->data = Tpsi_row.data();
					}
					else{
						//std::cout<<"ColMajor + Upper"<<std::endl;
						cholMat->data = Psi.data();
					}
				}
				else if constexpr(isCholType == isChol::Lower)
				{ 
					if constexpr( std::is_same_v< EigenType, MatCol> ){
						//std::cout<<"ColMajor + Lower"<<std::endl;	
						Tpsi_col = Psi.transpose(); //Do not use Cov.transposeInPlace() otherwise Cov is modified.
						cholMat->data = Tpsi_col.data();
					}
					else{
						//std::cout<<"ColMajor + Upper"<<std::endl;
						cholMat->data = Psi.data();
					}
				} 		
			}
							//std::cout<<"cholMat "<<std::endl;
			 				//for(auto i = 0; i < cholMat->size1; ++i){
			 					//for(auto j = 0; j < cholMat->size1; ++j){
			 						//std::cout<<gsl_matrix_get(cholMat,i,j)<<" ";
			 					//}
			 				//std::cout<<std::endl;	
			 				//}	
			//Sample with GSL
			gsl_ran_wishart(engine(), b+Psi.rows()-1, cholMat, &result, work);
							//std::cout<<"result "<<std::endl;
			 				//for(auto i = 0; i < result->size1; ++i){
			 					//for(auto j = 0; j < result->size1; ++j){
			 						//std::cout<<gsl_matrix_get(result,i,j)<<" ";
			 					//}
			 				//std::cout<<std::endl;	
			 				//}
			//Free and return
			gsl_matrix_free(cholMat);
			gsl_matrix_free(work);
			return return_obj;	 
			//Running with valgrind
			/*
			==5155== HEAP SUMMARY:
			==5155==     in use at exit: 0 bytes in 0 blocks
			==5155==   total heap usage: 40 allocs, 40 frees, 220,224 bytes allocated
			==5155== 
			==5155== All heap blocks were freed -- no leaks are possible
			*/ 	
		}
		template<typename EigenType>
		RetType operator()(double const & b, EigenType & Psi)const{
			return rwish<RetType, isCholType>()(GSL_RNG (), b,Psi);
		}
	};

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

	std::tuple<MatType, std::vector<double> >
	generate_design_matrix(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
							std::vector<double> const & grid_points);
	//Same as before but assumes that the grid points are r uniformly space points in [a,b], assuming the first to be equal to a and the last equal to b
	std::tuple<MatType, std::vector<double> >
	generate_design_matrix(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
							unsigned int const & r);
	// It returns a vector of length nderiv+1, each element is a grid_points.size() x n_basis matrix, the k-th element is the evaluation of 
	// all the k-th derivatives of all the splines in all the grid points. 
	// The first elemenst is the design matrix
	std::vector<MatType> evaluate_spline_derivative(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  				     std::vector<double> const & grid_points, unsigned int const & nderiv);
	std::vector<MatType> evaluate_spline_derivative(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  				     unsigned int const & r, unsigned int const & nderiv);
}



#endif
