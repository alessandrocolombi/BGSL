#ifndef __GWISHART_H__
#define __GWISHART_H__

#include "include_headers.h"
#include "include_graphs.h"
#include "utils.h"

struct GWishartTraits{
	using IdxType  	  = std::size_t;
	using Shape    	  = double;
	using MatRow  	  =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatCol   	  = Eigen::MatrixXd;
	using InvScale 	  = MatCol; 
	using UpperTriRow = MatRow;
	using UpperTriCol = MatCol;
	using InnerData   = MatRow;
	using CholType 	  = Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol = Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	using ColType     = Eigen::Matrix<double, Eigen::Dynamic, 1>;
	using RowType     = Eigen::Matrix<double, 1, Eigen::Dynamic>;
};

/*
	This class represents a random matrix distributed as a GWishart(b,D) where b is the Shape parameter and D in an Inverse_Scale matrix. Note that it is parametrized with respect to Inverse_Scale 
	matrix and not Scale matrix.
	It is not a template class but some methods are template methods. This choice allows more flexibility with respect to the type of graph. The same GWishart object can operate with all possible
	complete graph types.
	Layout of data member:
	- b, Shape parameter. It is a double and has to be > 2
	- D, Inverse_Scale matrix stored Columnwise. It is used only to generate chol_invD.
	- chol_invD, Cholesky decomposition of D^-1, i.e the Cholesky decomposition of Scale matrix. It has to be upper triangular such that D^-1 = chol_invD^T * chol_invD. It is stored by columns.
	  It is the prior parameter actually used in rgwish() and log_normalizing_constant() functions.  
	- data, the actual random matrix. It is stored by rows.
	- U, Cholesky decomposition of data, i.e data = U^T * U. It has to be upper triangular and stored by rows. It is possible to store only data and not U. In that case boolean parameter isFactorized
	  is false. It turns into true as soon as U is computed, for example via compute_Chol() function.
	-> It is copy-constructable and move-constructable and assignable
	-> WARNING: constructors and set methods ensure that only the upper part of chol_invD and U are actually used. They do not check that matrices are actually upper triangular, it is up to the user to 
	   ensure that. 
*/

class GWishart : public GWishartTraits{
	public:
		//Constructors
		GWishart()=default;
		//Parameters are defaulted
		GWishart(unsigned int const & p):b(3), D(InvScale::Identity(p, p)), chol_invD(D), isFactorized(false) {}
		//Receive Parameters
		GWishart(Shape _b, InvScale const & _DD):b(_b), D(_DD), isFactorized(false) 
		{
			if(_b <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			if(_DD != _DD.transpose()){
				throw std::runtime_error("Inv_Scale matrix is not symetric");
			}
			CholTypeCol cholD(_DD);
			if( cholD.info() != Eigen::Success)
				throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
			else
				chol_invD = cholD.solve(InvScale::Identity(D.rows(), D.cols())).llt().matrixU();
		}

		template<template <typename> class CompleteStructure = GraphType, typename T = unsigned int, typename Derived>
		//Receive Matrix and Parameters. Graph is not saved but it is needed to be sure that data satisfies a certain structure
		GWishart(CompleteStructure<T> const & G, Eigen::MatrixBase<Derived> const & _data, Shape _b, InvScale const & _DD):
				 data(_data.template selfadjointView<Eigen::Upper>()), b(_b), D(_DD), isFactorized(false) 
		{
			static_assert(	internal_type_traits::isCompleteGraph<CompleteStructure,T>::value,
						"___ERROR:_GWISHART_REQUIRES_A_GRAPH_IN_COMPLETE_FORM. HINT -> EVERY_GRAPH_SHOULD_PROVIDE_A_METHOD_CALLED completeview() THAT_CONVERTS_IT_IN_THE_COMPLETE_FORM");				
			//Check b
			if(_b <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			if(_DD != _DD.transpose()){
				throw std::runtime_error("Inv_Scale matrix is not symetric");
			}
			//Check D
			CholTypeCol cholD(_DD);
			if( cholD.info() != Eigen::Success)
				throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
			else
				chol_invD = cholD.solve(InvScale::Identity(G.get_size(),G.get_size())).llt().matrixU();
			//Check structure
			if(!this->check_structure(G)){
				std::cout<<"Structures of matrix and graph are not compatible"<<std::endl;
				//throw std::runtime_error("Structures of matrix and graph are not compatible");
			}
		}

		template< typename Derived >
		//Receive Parameters and a upper triangular matrix representing chol(D^-1)
		GWishart(Shape _b, InvScale const & _DD, Eigen::MatrixBase<Derived> const & _chol_invD):b(_b), D(_DD), chol_invD(_chol_invD.template triangularView<Eigen::Upper>()), isFactorized(false) 
		{
			if(_b <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			if(_DD != _DD.transpose()){
				throw std::runtime_error("Inv_Scale matrix is not symetric");
			}
		}


		template<template <typename> class CompleteStructure = GraphType, typename T = unsigned int, typename Derived1, typename Derived2>
		//Receive Matrix, Parameters and a upper triangular matrix representing chol(D^-1)
		GWishart(CompleteStructure<T> const & G, Eigen::MatrixBase<Derived1> const & _data, Shape _b, InvScale const & _DD, Eigen::MatrixBase<Derived2> const & _chol_invD):
				 data(_data.template selfadjointView<Eigen::Upper>()), b(_b), D(_DD), chol_invD(_chol_invD.template triangularView<Eigen::Upper>()), isFactorized(false)
		{
			static_assert(	internal_type_traits::isCompleteGraph<CompleteStructure,T>::value,
						"___ERROR:_GWISHART_REQUIRES_A_GRAPH_IN_COMPLETE_FORM. HINT -> EVERY_GRAPH_SHOULD_PROVIDE_A_METHOD_CALLED completeview() THAT_CONVERTS_IT_IN_THE_COMPLETE_FORM");
			//Check b
			if(_b <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			//Check D
			if(_DD != _DD.transpose()){
				throw std::runtime_error("Inv_Scale matrix is not symetric");
			}
			//Check structure
			if(!this->check_structure(G)){
				std::cout<<"Structures of matrix and graph are not compatible"<<std::endl;
				//throw std::runtime_error("Structures of matrix and graph are not compatible");
			}
		}

		template< typename Derived>
		GWishart(Eigen::MatrixBase<Derived> const & _U, Shape _b, InvScale const & _DD ):b(_b),D(_DD), U(_U.template triangularView<Eigen::Upper>()), isFactorized(true)
		{ 
			if(_b <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			data = U.transpose() * U;
			if(_DD != _DD.transpose()){
				throw std::runtime_error("Inv_Scale matrix is not symetric");
			}
			CholTypeCol cholD(_DD);
			if( cholD.info() != Eigen::Success)
				throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
			else
				chol_invD = cholD.solve(InvScale::Identity(D.rows(),D.rows() )).llt().matrixU();
		}

		template< typename Derived1, typename Derived2>
		GWishart(Eigen::MatrixBase<Derived1> const & _U, Shape _b, InvScale const & _DD, Eigen::MatrixBase<Derived2>  const & _chol_invD)
				 :b(_b),D(_DD), chol_invD(_chol_invD.template triangularView<Eigen::Upper>()), U(_U.template triangularView<Eigen::Upper>()), isFactorized(true)
		{ 
			if(_b <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			data = U.transpose() * U;
		}
		
		//Getters
		inline MatRow& get_upper_Chol(){
			return U;
		}
		inline MatRow get_upper_Chol()const{
			return U;
		}
		inline MatRow get_lower_Chol()const{
			return U.transpose();
		}
		inline ColType get_col(IdxType const & j) const{
		  return data.row(j); //it's symetric
		}
		inline RowType get_row(IdxType const & i) const{
		  return data.row(i);
		}
		inline InnerData& get_matrix(){
			return data;
		}
		inline InnerData get_matrix() const{
			return data;
		}
		inline Shape get_shape() const{
			return b;
		}
		inline InvScale get_inv_scale()const{
			return D;
		}
		inline UpperTriCol get_chol_invD()const{
			return chol_invD;
		}

		//Setter
		template<template <typename> class CompleteStructure = GraphType, typename T = unsigned int>
		void set_matrix(CompleteStructure<T> const & G, InnerData const & Mat){ 
			static_assert(	internal_type_traits::isCompleteGraph<CompleteStructure,T>::value,
						"___ERROR:_GWISHART_REQUIRES_A_GRAPH_IN_COMPLETE_FORM. HINT -> EVERY_GRAPH_SHOULD_PROVIDE_A_METHOD_CALLED completeview() THAT_CONVERTS_IT_IN_THE_COMPLETE_FORM");
			if(Mat.cols() != Mat.rows())
				throw std::runtime_error("Non squared matrix inserted");
			data = Mat.selfadjointView<Eigen::Upper>();
			//Check structure
			if(!this->check_structure(G)){
				std::cout<<"Setting a new matrix: Structures of matrix and graph are not compatible"<<std::endl;
				//throw std::runtime_error("Structures of matrix and graph are not compatible");
			}
			//compute_Chol();
			isFactorized = false;
		}
		template<template <typename> class CompleteStructure = GraphType, typename T = unsigned int> //--> forse meglio toglierla direttamente oppure implementarla wrt rgwish member function
		void set_random(const CompleteStructure<T> & G, double const threshold = 1e-8, sample::GSL_RNG const & engine = sample::GSL_RNG()){
			static_assert(	internal_type_traits::isCompleteGraph<CompleteStructure,T>::value,
						"___ERROR:_GWISHART_REQUIRES_A_GRAPH_IN_COMPLETE_FORM. HINT -> EVERY_GRAPH_SHOULD_PROVIDE_A_METHOD_CALLED completeview() THAT_CONVERTS_IT_IN_THE_COMPLETE_FORM");
			data = utils::rgwish(G, b, D, threshold, engine); 
			isFactorized = false;
		}
		void set_shape(Shape const & _bb){
			if(_bb <= 2)
				throw std::runtime_error("Shape parameter has to be larger than 2");
			b = _bb;
		}
		void set_inv_scale(InvScale const & _DD){
			if(_DD != _DD.transpose()){
				throw std::runtime_error("Inv_Scale matrix is not symetric");
			}
			CholTypeCol cholD(_DD);
			if( cholD.info() != Eigen::Success)
				throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
			else{
				D 	 		 = _DD;
				chol_invD    = cholD.solve(InvScale::Identity(D.rows(),D.rows())).llt().matrixU();
			}
		}
		template< typename Derived >
		void set_chol_invD(Eigen::MatrixBase<Derived> const & _chol_invD){
			chol_invD = _chol_invD. template triangularView<Eigen::Upper>();
		}
		//Cholesky operations
		inline void compute_Chol(){
			CholType chol(this->data);
			if( chol.info() != Eigen::Success)
				throw std::runtime_error("Chol decomposition failed, probably the matrix is not sdp");
			U = chol.matrixU();
			isFactorized = true;
		}
		//main methods
		template<template <typename> class CompleteStructure = GraphType, typename T = unsigned int, typename NormType = utils::MeanNorm>
		void rgwish(const CompleteStructure<T> & G, double const threshold = 1e-8, sample::GSL_RNG const & engine = sample::GSL_RNG() );
		template<template <typename> class CompleteStructure = GraphType, typename T = unsigned int>
		bool check_structure(const CompleteStructure<T> & G)const{
			static_assert(	internal_type_traits::isCompleteGraph<CompleteStructure,T>::value,
						"___ERROR:_GWISHART_REQUIRES_A_GRAPH_IN_COMPLETE_FORM. HINT -> EVERY_GRAPH_SHOULD_PROVIDE_A_METHOD_CALLED completeview() THAT_CONVERTS_IT_IN_THE_COMPLETE_FORM");
			if(data.rows() != G.get_size())
				return false;
			double threshold = 1e-6;
			for(IdxType i = 1; i < data.rows()-1; ++i){
				for(IdxType j = i+1; j < data.cols(); ++j){
					if( ( G(i,j) == 0 && std::abs(data(i,j))>threshold ) || (G(i,j) == 1 && std::abs(data(i,j))<threshold)){
						return false;	
					}						
				}
			}
				
			return true;	
		}
		template<template <typename> class CompleteStructure = GraphType, typename Type = unsigned int>
		long double log_normalizing_constat(const CompleteStructure<Type> & G, unsigned int const & MCiteration = 100, sample::GSL_RNG const & engine = sample::GSL_RNG() ); 
		//Public member stating if the matrix is factorized or not, i.e if U is such that data=U.transpose()*U
		bool 		isFactorized;
	private:
		Shape 		b;
		InvScale 	D;
		UpperTriCol chol_invD;  //Upper triangular, chol(D^-1) --> chol_invD^T * chol_invD = D^-1
		InnerData 	data;	
		UpperTriRow U; 			// U = chol(data)^T, i.e, data= U^TU 

};



template<template <typename> class CompleteStructure, typename Type>
long double GWishart::log_normalizing_constat(const CompleteStructure<Type> & G, unsigned int const & MCiteration, sample::GSL_RNG const & engine){
	
	static_assert(	internal_type_traits::isCompleteGraph<CompleteStructure,Type>::value,
				"___ERROR:_GWISHART_REQUIRES_A_GRAPH_IN_COMPLETE_FORM. HINT -> EVERY_GRAPH_SHOULD_PROVIDE_A_METHOD_CALLED completeview() THAT_CONVERTS_IT_IN_THE_COMPLETE_FORM");
	//Typedefs
	using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol   	= Eigen::MatrixXd;
	using IdxType  	= GWishartTraits::IdxType;
	using iterator  = std::vector<unsigned int>::iterator; 	
	using citerator = std::vector<unsigned int>::const_iterator; 	
	using Graph 	= CompleteStructure<Type>;
	//Step 1: Preliminaries
	const unsigned int n_links(G.get_n_links());
	const unsigned int N(G.get_size());
	const unsigned int max_n_links(0.5*N*(N-1));
	const unsigned int N_free_elements(n_links + N);
	const long double min_numeric_limits_ld = std::numeric_limits<long double>::min() * 1000;
	long double result_MC{0};
	unsigned int number_nan{0};
			//if(seed == 0){
				////std::random_device rd;
				////seed=rd();
				//seed=static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			//}
			//sample::GSL_RNG engine(seed);
			//std::default_random_engine engine(seed);
			//std::normal_distribution<> rnorm{0.0,1.0}; //For sampling from normal distribution
	sample::rnorm rnorm;
	sample::rchisq rchisq;
	
	

	//Start
	//nu[i] = #1's in i-th row, from position i+1 up to end. Note that it is also the number of off-diagonal elements in each row 
	std::vector<unsigned int> nu(N); 
	#pragma omp parallel for shared(nu)
	for(IdxType i = 0; i < N; ++i){
		std::vector<unsigned int> nbd_i = G.get_nbd(i);
		nu[i] = std::count_if(nbd_i.cbegin(), nbd_i.cend(), [i](const unsigned int & idx){return idx > i;});
	} 
		//std::cout<<"nbd:"<<std::endl;
			//for(auto v : nu)
				//std::cout<<v<<", ";
			//std::cout<<std::endl;

	if(n_links == max_n_links){
		//std::cout<<"Pieno"<<std::endl;
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
		//std::cout<<"Vuoto"<<std::endl;
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
		const MatCol &T(chol_invD); //T is colwise because i need to extract its columns. Take a reference to keep same notation of free function
							//std::cout<<"T = "<<std::endl<<T<<std::endl;
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
		std::vector<double> vec_ss_nonfree;
		std::vector<double> vec_ss_nonfree_result; //useful for parallel case, does not harm in sequential case (no unuseful copies are done)
		

		#pragma omp parallel private(thread_id),private(vec_ss_nonfree),shared(vec_ss_nonfree_result)
		{
			int n_threads{1};
			#ifdef PARALLELEXEC
				n_threads = omp_get_num_threads();
						//std::cout<<"n_threads = "<<n_threads<<std::endl;
			#endif
			vec_ss_nonfree.reserve(MCiteration/n_threads);
			//Start MC for loop

			for(IdxType iter = 0; iter < MCiteration/n_threads; ++ iter){
				#ifdef PARALLELEXEC
				thread_id = omp_get_thread_num();
				#endif
				MatRow Psi(MatRow::Zero(N,N));
				//In the end it has to be exp(-1/2 sum( psi_nonfree_ij^2 )). I accumulate the sum of non free elements squared every time they are generated
				double sq_sum_nonfree{0};
				//Step 2: Sampling the free elements
				//- Sample the diagonal elements from chisq(b + nu_i) (or equivalently from a gamma((b+nu_i)/2, 1/2))
				//- Sample the off-diagonal elements from N(0,1)

				std::vector<double> vector_free_element(N_free_elements);
				unsigned int time_to_diagonal{0};
				citerator it_nu = nu.cbegin();
				for(unsigned int i = 0; i < N_free_elements; ++i){
					if(time_to_diagonal == 0){
						//vector_free_element[i] = std::sqrt(std::gamma_distribution<> (0.5*(b+(*it_nu)),2.0)(engine));
						vector_free_element[i] = std::sqrt(rchisq(engine, (double)(b+(*it_nu)) ));
						//if(vector_free_element[i] < min_numeric_limits_ld)
							//throw std::runtime_error("Impossibile, un elemento diagonale è nullo");
					}
					else
						vector_free_element[i] = rnorm(engine);

					if(time_to_diagonal++ == *it_nu){
						time_to_diagonal = 0;
						it_nu++;
					}
				}

				std::vector<double>::const_iterator it_fe = vector_free_element.cbegin();

				if(H.isIdentity()){ //Takes into account also the case D diagonal but not identity
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

					Psi(1,1) = *it_fe; //Counting also the diagonal element because they have to be updated too
					it_fe++;
					for(unsigned int j = 2; j < N; ++j){ 
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
					for(unsigned int i = 2; i < N-1; ++i){ // i = N-1 (last element) is useless
						Eigen::VectorXd S = Psi.block(0,i,i,1); //Non cache friendly 
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
							return (Eigen::RowVectorXd(Psi.block(a,a,1,b-a)).dot(Eigen::VectorXd(H.block(a,b,b-a,1)))); //Cache friendly
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
						Eigen::VectorXd S = Psi.block(0,i,i,1) + Sums.block(0,i-1,i,1); //non cache friendly 
						for(unsigned int j = i+1; j < N; ++j){
							Sums(i,j-1) = CumSum(i,j);
							if(G(i,j) == false){
								Eigen::VectorXd S_star = Psi.block(0,j,i,1) + Sums.block(0,j-1,i,1); //non cache friendly 
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
											//long double temp = std::exp((long double)(-0.5 * sq_sum_nonfree));
												//if(temp < min_numeric_limits_ld)
												//temp = min_numeric_limits_ld;
				if( sq_sum_nonfree != sq_sum_nonfree){
					//throw std::runtime_error("!!!!!!!!!!!!!!! Molto probabile che ci sia un nan !!!!!!!!!!!!!!!!!!!");
					//std::cout<<"!!!!!!!!!!!!!!! Molto probabile che ci sia un nan !!!!!!!!!!!!!!!!!!!"<<std::endl;
					number_nan++;
				}
				else{
					vec_ss_nonfree.emplace_back(-0.5*sq_sum_nonfree);
				}
			}

									//std::cout<<"vec_ss_nonfree.size() = "<<vec_ss_nonfree.size()<<std::endl;
			#ifdef PARALLELEXEC
				#pragma omp barrier
				#pragma omp critical
				{
							//std::cout<<"I'm td number #"<<omp_get_thread_num()<<std::endl;
					vec_ss_nonfree_result.insert(vec_ss_nonfree_result.end(), 
												 std::make_move_iterator(vec_ss_nonfree.begin()), std::make_move_iterator(vec_ss_nonfree.end())
												);
				}	
			#endif				
		}
		#ifndef PARALLELEXEC
			std::swap(vec_ss_nonfree_result, vec_ss_nonfree);
		#endif
										//std::cout<<"vec_ss_nonfree_result.size() = "<<vec_ss_nonfree_result.size()<<std::endl;
		result_MC = -std::log(vec_ss_nonfree_result.size()) + utils::logSumExp(vec_ss_nonfree_result);
										//std::cout<<"result_MC = "<<result_MC<<std::endl;
		//Step 5: Compute constant term and return
		long double result_const_term{0};
		#pragma parallel for reduction(+:result_const_term)
		for(IdxType i = 0; i < N; ++i){
			result_const_term += (long double)nu[i]/2.0*utils::log_2pi +
								 (long double)(b+nu[i])/2.0*utils::log_2 +
								 (long double)(b+G.get_nbd(i).size())*std::log(T(i,i)) + //Se T è l'identità posso anche evitare di calcolare questo termine
								 std::lgammal((long double)(0.5*(b + nu[i])));
		}//This computation requires the best possible GWishart because il will generate a very large number
					//std::cout<<"Constant term = "<<result_const_term<<std::endl;
		return result_MC + result_const_term;
	}
	
}

template<template <typename> class CompleteStructure, typename T, typename NormType>
void GWishart::rgwish(const CompleteStructure<T> & G, double const threshold, sample::GSL_RNG const & engine){
	std::tie(this->data, std::ignore, std::ignore) = utils::rgwish_core<CompleteStructure,T,utils::ScaleForm::CholUpper_InvScale,NormType>(G,this->b,this->chol_invD,threshold,engine,500);

}

#endif



