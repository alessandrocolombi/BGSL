#ifndef __GWISHART_H__
#define __GWISHART_H__

#include "include_headers.h"
#include "include_graphs.h"
#include "utils.h"

struct GWishartTraits{
	using IdxType  	 = std::size_t;
	using Shape    	 = double;
	using MatRow  	 = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using MatCol   	 = Eigen::MatrixXd;
	using InvScale 	 = MatCol; //<---- AL MOMENTO è PER COLONNE perché nella rwish serve cosi
	using UpperTri   = Eigen::TriangularView< MatRow , Eigen::Upper>;
	using InnerData  = MatRow;
	using CholMatrix = MatRow;
	using CholType 	 = Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	using ColType    = Eigen::Matrix<double, Eigen::Dynamic, 1>;
	using RowType    = Eigen::Matrix<double, 1, Eigen::Dynamic>;
};

// E' un template rispetto al tipo di grafo che passo in input.
// Posso passare o un grafo completo (quindi di tipo GraphType) o una CompleteView proveniente da un grafo a blocchi
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int>
class GWishart : GWishartTraits{
public:
	using Graph    = GraphStructure<T>;	
	using GraphPtr = std::shared_ptr<const Graph>;
	//Constructors
	//Parameters are defaulted
	GWishart(GraphPtr _G):ptr_graph(_G), b(3), D(InvScale::Identity(ptr_graph->get_size(), ptr_graph->get_size())), chol_invD(D), isFactorized(false) {};
	//Receive Graphs and Parameters
	GWishart(GraphPtr _G, Shape _b, InvScale const & _DD):ptr_graph(_G), b(_b), D(_DD), isFactorized(false) {
		if(_b <= 2)
			throw std::runtime_error("Shape parameter has to be larger than 2");
		if(_DD != _DD.transpose()){
			throw std::runtime_error("Inv_Scale matrix is not symetric");
		}
		CholTypeCol cholD(_DD);
		if( cholD.info() != Eigen::Success)
			throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
		else
			chol_invD = cholD.solve(InvScale::Identity(ptr_graph->get_size(),ptr_graph->get_size())).llt().matrixL();
	};
	//Receive Graphs, Matrix and Parameters
	GWishart(GraphPtr _G, InnerData const & _data, Shape _b, InvScale const & _DD):
			 ptr_graph(_G), data(_data.selfadjointView<Eigen::Upper>()), b(_b), D(_DD), isFactorized(false) 
	{
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
			chol_invD = cholD.solve(InvScale::Identity(ptr_graph->get_size(),ptr_graph->get_size())).llt().matrixL();
		//Check structure
		if(!this->check_structure())
			throw std::runtime_error("Structures of matrix and graph are not compatible");
	};
	//Receive Graphs, Parameters and a lower triangular matrix representing chol(D^-1)
	GWishart(GraphPtr _G, Shape _b, InvScale const & _DD, InvScale const & _chol_invD):ptr_graph(_G), b(_b), D(_DD), chol_invD(_chol_invD), isFactorized(false) 
	{
		if(_b <= 2)
			throw std::runtime_error("Shape parameter has to be larger than 2");
		if(_DD != _DD.transpose()){
			throw std::runtime_error("Inv_Scale matrix is not symetric");
		}
	};
	//Receive Graphs, Matrix, Parameters and a lower triangular matrix representing chol(D^-1)
	GWishart(GraphPtr _G, InnerData const & _data, Shape _b, InvScale const & _DD, InvScale const & _chol_invD):
			 ptr_graph(_G), data(_data.selfadjointView<Eigen::Upper>()), b(_b), D(_DD), chol_invD(_chol_invD), isFactorized(false)
	{
		//Check b
		if(_b <= 2)
			throw std::runtime_error("Shape parameter has to be larger than 2");
		//Check D
		if(_DD != _DD.transpose()){
			throw std::runtime_error("Inv_Scale matrix is not symetric");
		}
		//Check structure
		if(!this->check_structure())
			throw std::runtime_error("Structures of matrix and graph are not compatible");
	};

	GWishart(GraphPtr _G, UpperTri const & _U, Shape _b, InvScale const & _DD ):ptr_graph(_G), b(_b),D(_DD), U(_U), isFactorized(true)
	{ 
		if(_b <= 2)
			throw std::runtime_error("Shape parameter has to be larger than 2");
		data = U * U.transpose();
		if(_DD != _DD.transpose()){
			throw std::runtime_error("Inv_Scale matrix is not symetric");
		}
		CholTypeCol cholD(_DD);
		if( cholD.info() != Eigen::Success)
			throw std::runtime_error("Chol decomposition of Inv Scale matrix failed, probably the matrix is not sdp");
		else
			chol_invD = cholD.solve(InvScale::Identity(ptr_graph->get_size(),ptr_graph->get_size())).llt().matrixL();
	};//devo passargli (..., mat.triangularView<Eigen::Upper>())
	
	GWishart(GraphPtr _G, UpperTri const & _U, Shape _b, InvScale const & _DD, InvScale const & _chol_invD)
			 :ptr_graph(_G), b(_b),D(_DD), chol_invD(_chol_invD), U(_U), isFactorized(true)
	{ 
		if(_b <= 2)
			throw std::runtime_error("Shape parameter has to be larger than 2");
		data = U * U.transpose();
	};//devo passargli (..., mat.triangularView<Eigen::Upper>())
	//Getters
	inline CholMatrix get_upper_Chol()const{
		return U;
	}
	inline CholMatrix get_lower_Chol()const{
		return U.transpose();
	}
	inline ColType get_col(IdxType const & j) const{
	  return data.row(j); //it's symetric
	}
	inline RowType get_row(IdxType const & i) const{
	  return data.row(i);
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
	inline InvScale get_chol_invD()const{
		return chol_invD;
	}
	inline std::vector<unsigned int> get_nbd(IdxType const & i)const{
		return ptr_graph->get_nbd(i);
	}
	//Setter
	void set_matrix(InnerData const & Mat){ 
		if(Mat.cols() != Mat.rows())
			throw std::runtime_error("Non squared matrix inserted");
		data = Mat.selfadjointView<Eigen::Upper>();
		//Check structure
		if(!this->check_structure())
			throw std::runtime_error("Structures of matrix and graph are not compatible");
		//compute_Chol();
		isFactorized = false;
	}
	void set_random(unsigned int const & seed = 0){
		data = utils::rgwish(*ptr_graph, b, D, seed); //Funziona con valori di default?
		isFactorized = false;
	}

	template<typename NormType = utils::MeanNorm>
	void rgwish(unsigned int seed = 0);
	
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
			chol_invD    = cholD.solve(InvScale::Identity(ptr_graph->get_size(),ptr_graph->get_size())).llt().matrixU();
		}
	}
	void set_chol_invD(InvScale const & _chol_invD){
		chol_invD = _chol_invD;
	}
	//Cholesky operations
	inline void compute_Chol(){
		CholType chol(this->data);
		if( chol.info() != Eigen::Success)
			throw std::runtime_error("Chol decomposition failed, probably the matrix is not sdp");
		U = chol.matrixU();
		isFactorized = true;
	}
	CholMatrix Chol()const{ // QUESTA CHE È? A CHE SERVE?
		CholType chol(this->data);
		if( chol.info() != Eigen::Success)
			throw std::runtime_error("Chol decomposition failed, probably the matrix is not sdp");
		return chol.matrixU(); 
	}
	bool check_structure()const{
		double threshold = 1e-6;
		for(IdxType i = 1; i < data.rows()-1; ++i){
			for(IdxType j = i+1; j < data.cols(); ++j){
				if( ( (*ptr_graph)(i,j) == 0 && std::abs(data(i,j))>threshold ) || ((*ptr_graph)(i,j) == 1 && std::abs(data(i,j))<threshold)){
					return false;	
				}
						
			}
		}
			
		return true;	
	}

	long double log_normalizing_constat(unsigned int const & MCiteration = 100, unsigned int seed=0); //Forse addirittura long double come return type
	long double log_normalizing_constat_DId(unsigned int const & MCiteration = 100, unsigned int seed=0);
	//void print()const{
		//std::cout<<std::endl<<data<<std::endl;
	//}

	//Public member stating if the matrix is factorized or not, i.e if U is such that data=U.transpose()*U
	bool 		isFactorized;
private:
	GraphPtr 	ptr_graph;
	Shape 		b;
	InvScale 	D;
	InvScale 	chol_invD;  //Upper triangular, chol(D^-1)
	InnerData 	data;	
	CholMatrix  U; // U = chol(K)^T, i.e, K = U^TU 

};



template<template <typename> class GraphStructure, typename Type>
long double GWishart<GraphStructure, Type>::log_normalizing_constat(unsigned int const & MCiteration, unsigned int seed){
	//Typedefs
	using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol   	= Eigen::MatrixXd;
	using IdxType  	= GWishartTraits::IdxType;
	using iterator  = std::vector<unsigned int>::iterator; 	
	using citerator = std::vector<unsigned int>::const_iterator; 	
	
	//Step 1: Preliminaries
	const unsigned int n_links(ptr_graph->get_n_links());
	const unsigned int N(ptr_graph->get_size());
	const unsigned int max_n_links(0.5*N*(N-1));
	const unsigned int N_free_elements(n_links + N);
	const long double min_numeric_limits_ld = std::numeric_limits<long double>::min() * 1000;
	long double result_MC{0};
	if(seed == 0)
		seed = std::chrono::system_clock::now().time_since_epoch().count();	
	sample::GSL_RNG engine(seed);
	sample::rnorm rnorm;
	//std::default_random_engine engine(seed);
	//std::normal_distribution<> rnorm{0.0,1.0}; //For sampling from normal distribution
	

	//Start
	//nu[i] = #1's in i-th row, from position i+1 up to end. Note that it is also  che number of off-diagonal elements in each row 
	std::vector<unsigned int> nu(N); 
	#pragma omp parallel for shared(nu)
	for(IdxType i = 0; i < N; ++i){
		std::vector<unsigned int> nbd_i = ptr_graph->get_nbd(i);
		nu[i] = std::count_if(nbd_i.cbegin(), nbd_i.cend(), [i](const unsigned int & idx){return idx > i;});
	} 
			//for(auto v : nu)
				//std::cout<<v<<", ";
			//std::cout<<std::endl;

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
		const MatCol &T(chol_invD); //T is colwise because i need to extract its columns. Operazione inutile perché l'ho gia calcolata ma non voglio cambiare nome
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
					if((*ptr_graph)(0,j) == true){
						Psi(0,j) = *it_fe;
						it_fe++;
					}
				}
						//std::cout<<"Psi dopo i = 0: "<<std::endl<<Psi<<std::endl;	

				Psi(1,1) = *it_fe;
				it_fe++;	
				for(unsigned int j = 2; j < N; ++j){ //Counting also the diagonal element because they have to be updated too
					if((*ptr_graph)(1,j) == false){
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
						if((*ptr_graph)(i,j) == false){
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
						throw std::runtime_error("Invalid index request");
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
					if((*ptr_graph)(0,j) == false){
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
					if((*ptr_graph)(1,j) == false){
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
						if((*ptr_graph)(i,j) == false){
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

			//#pragma omp critical
			//{
				//std::cout<<"I'm thread #"<<thread_id<<", my result is = "<<result_MC<<std::endl;
			//}

		}

		result_MC /= MCiteration;
						//std::cout<<"result_MC prima del log = "<<result_MC<<std::endl;
		result_MC = std::log(result_MC);
						//std::cout<<"result_MC = "<<result_MC<<std::endl;
		//Step 5: Compute constant term and return
		long double result_const_term{0};

		
		#pragma parallel for reduction(+:result_const_term)
		for(IdxType i = 0; i < N; ++i){
			result_const_term += (double)nu[i]/2.0*utils::log_2pi + 
								 (double)(b+nu[i])/2.0*utils::log_2 +
								 (double)(b+ptr_graph->get_nbd(i).size())*std::log(T(i,i)) + //Se T è l'identità posso anche evitare di calcolare questo termine
								 std::lgammal((long double)(0.5*(b + nu[i])));
		}//This computation requires the best possible precision because il will generate a very large number
						//std::cout<<"Constant term = "<<result_const_term<<std::endl;
		return result_MC + result_const_term; 
	}
	
}


template<template <typename> class GraphStructure, typename T>
template< typename NormType >
void GWishart<GraphStructure, T>::rgwish(unsigned int seed){

	std::cout<<"Dentro ad rgwish"<<std::endl;
	//Typedefs
	using iterator  = std::vector<unsigned int>::iterator;
	using citerator = std::vector<unsigned int>::const_iterator;
	//Set parameters
	unsigned int const N = ptr_graph->get_size();
	std::cout<<"Esiste un grafo"<<std::endl;
	unsigned int const n_links = ptr_graph->get_n_links();
	unsigned int const max_iter = 500;
	double const threshold = 1e-8;
	bool converged = false;
	unsigned int it{0};
	double norm_res{1.0};
	if(seed == 0){
		  std::random_device rd;
		  seed=rd();
		//seed = std::chrono::system_clock::now().time_since_epoch().count();	
	}
	sample::GSL_RNG engine(seed);

	if(n_links == 0){
				std::cout<<"Grafo vuoto"<<std::endl;
				std::cout<<"seed = "<<seed<<std::endl;
				//std::cout<<"Empty Graph"<<std::endl;
		MatRow K_return(MatRow::Identity(N,N));
		for(unsigned int i = 0; i < N; ++i) //Non cache friendly
			K_return(i,i) = std::sqrt(  sample::rchisq()(engine, (double)(b + N - i - 1))  );

				//std::cout<<"K_return:"<<std::endl<<K_return<<std::endl;
		data = K_return;
		return;
	}

	//Step 1: Draw K from Wish(b,D) = wish(D^-1, b+N-1) 
	//MatCol Inv_D(chol_invD*chol_invD.transpose());          //Questo è un conto che vorrei evitare
	MatCol K( sample::rwish<MatCol, sample::isChol::Lower>()(engine, b, chol_invD) );  
			//std::cout<<"K: "<<std::endl<<K<<std::endl;
	if(n_links == ptr_graph->get_possible_links()){
				std::cout<<"Complete Graph"<<std::endl;
		data = K;
		return; 
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
			std::vector<unsigned int> nbd_i = ptr_graph->get_nbd(i);
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
					MatRow Omega_Ni_Ni( utils::SubMatrix<utils::Symmetric::True>(nbd_i, Omega) );
						//std::cout<<"Omega_Ni_Ni: "<<std::endl<<Omega_Ni_Ni<<std::endl;
					Eigen::VectorXd Sigma_Ni_i( utils::SubMatrix(nbd_i, i, Sigma) ); //-->questa SubMatrix() meglio farla con block() di eigen -> ma è sbatti
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
					MatRow Omega_noti_noti( utils::SubMatrix<utils::Symmetric::True>(i , Omega) );
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
	std::cout<<"Converged? "<<converged<<std::endl;
	std::cout<<"#iterazioni = "<<it<<std::endl;
	//Step 9: return K = Omega^-1
	data = Omega.selfadjointView<Eigen::Upper>().llt().solve(MatRow::Identity(N, N));
	return; 
	//Omega = Omega.selfadjointView<Eigen::Upper>();
	//return Omega.inverse();
}



//Questa non è da usare
template<template <typename> class GraphStructure, typename T>
long double GWishart<GraphStructure, T>::log_normalizing_constat_DId(unsigned int const & MCiteration, unsigned int seed){
	//Typedefs
	using MatRow  	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol   	= Eigen::MatrixXd;
	using IdxType  	= GWishartTraits::IdxType;
	using iterator  = std::vector<unsigned int>::iterator; 	
	using citerator = std::vector<unsigned int>::const_iterator; 	
	//Check

	//Step 1: Preliminaries
	const unsigned int n_links(ptr_graph->get_n_links());
		//std::cout<<"n_links in the graph = "<<n_links<<std::endl;
	const unsigned int N(ptr_graph->get_size());
		//std::cout<<"N = "<<N<<std::endl;
	const unsigned int max_n_links(0.5*N*(N-1));
	long double result_MC{0};

	if(seed == 0)
		seed = std::chrono::system_clock::now().time_since_epoch().count();
	
	std::default_random_engine engine(seed);
	
	//For sampling from normal distribution
	std::normal_distribution<> rnorm{0.0,1.0};

	//Start
	if(n_links == max_n_links){
		return -1.0;
	}
	else if(n_links == 0){
		return -2.0;
	}
	else{
		
		std::vector<unsigned int> nu(N); //nu[i] = #1's in i-th row, from position i+1 up to end
			//std::cout<<"nu.size() = "<<nu.size()<<std::endl;
		//std::vector<unsigned int> kk(N); //kk[i] = #1's in i-th col, from position 0 up to i-1

		//- Define nu_i = #1's in i-th row
		#pragma omp parallel for shared(nu)
		for(IdxType i = 0; i < N; ++i){
			std::vector<unsigned int> nbd_i = ptr_graph->get_nbd(i);
			nu[i] = std::count_if(nbd_i.cbegin(), nbd_i.cend(), [i](const unsigned int & idx){return idx > i;});
			//kk[i] = ptr_graph->get_nbd(i).size() - nu[i];
		} 
			//std::cout<<"nu[i] = #1's in i-th row, from position i+1 up to end: "<<std::endl;
			//for(auto v : nu)
				//std::cout<<v<<", ";
			//std::cout<<std::endl;

		//Qui inizia il ciclo 
		for(IdxType iter = 0; iter < MCiteration; ++ iter){
			MatRow Psi(MatRow::Zero(N,N));
			//In the end it has to be exp(-1/2 sum( psi_nonfree_ij^2 )). I accumulate the sum of non free elements squared every time they are generated
			double sq_sum_nonfree{0}; 
			//Step 2: Sampling the free elements
			//- Sample the diagonal elements from chisq(b + nu_i) (or equivalently from a gamma((b+nu_i)/2, 1/2))
			//- Sample the off-diagonal elements from N(0,1)
			#pragma omp parallel for shared(Psi)
			for(unsigned int i = 0; i < N; ++i){
				// In order to sample from a chisq(dof) i need to sample from a gamma(dof/2,1/2).
				// However c++ notation for gamma_distribution requires the inverse of the second parameter
				std::gamma_distribution<> chisq(0.5*(b+nu[i]),2.0); 
				for(unsigned int j = i; j < N; ++j){
					if(i == j)
						Psi(i,i) = std::sqrt(chisq(engine));
					else if((*ptr_graph)(i,j) == true){
							//std::cout<<"G("<<i<<", "<<j<<") = "<<(*ptr_graph)(i,j)<<std::endl;
						Psi(i,j) = rnorm(engine);
					}
				}
			}
				//std::cout<<"Psi dopo i free elements: "<<std::endl<<Psi<<std::endl;

			//Step 3: Complete Psi (upper part)
			

			//- If i==0 -> all null			
			//- If i==1 -> simplified case of formula 2
			if((*ptr_graph)(0,1) == true){
				for(unsigned int j = 2; j < N; ++j){
					if((*ptr_graph)(1,j) == false){
						Psi(1,j) = - Psi(0,1)*Psi(0,j)/Psi(1,1);
						sq_sum_nonfree += Psi(1,j)*Psi(1,j);
					}
				}	
			}
				//std::cout<<"Psi dopo i = 1: "<<std::endl<<Psi<<std::endl;
			//- If i>1 -> formula 2
			for(unsigned int i = 2; i < N-1; ++i){
				Eigen::VectorXd S = Psi.block(0,i,i,1); //non cache friendly operation perché Psi e Sums sono per righe e qua sto estraendo delle colonne
				for(unsigned int j = i+1; j < N; ++j){
					if((*ptr_graph)(i,j) == false){
						Eigen::VectorXd S_star = Psi.block(0,j,i,1); //non cache friendly operation perché Psi e Sums sono per righe e qua sto estraendo delle colonne
						Psi(i,j) = - S.dot(S_star)/Psi(i,i) ;
						sq_sum_nonfree += Psi(i,j)*Psi(i,j);
					}
				}
					//std::cout<<"Psi dopo i = "<<i<<": "<<std::endl<<Psi<<std::endl;
			}
			//Step 4: Compute exp( -0.5 * sum_NonFreeElements(Psi_ij)^2 )
				//std::cout<<"Sum squared Non free elements = "<<sq_sum_nonfree<<std::endl;
				//std::cout<<"Result iteration = "<<std::exp(-0.5 * sq_sum_nonfree)<<std::endl;
			result_MC += std::exp(-0.5 * sq_sum_nonfree);
		}

		result_MC /= MCiteration;
		//std::cout<<"result_MC prima del log = "<<result_MC<<std::endl;
		result_MC = std::log(result_MC);
		//std::cout<<"result_MC = "<<result_MC<<std::endl;
		//Step 5: Compute constant term and return
		long double result_const_term{0};

		#pragma parallel for reduction(+:result_const_term)
		for(IdxType i = 0; i < N; ++i){
			result_const_term += (double)nu[i]/2.0*utils::log_2pi + 
								 (double)(b+nu[i])/2.0*utils::log_2 +
								 std::lgammal((long double)(0.5*(b + nu[i])));
		}//This computation requires the best possible precision because il will generate a very large number
	    //utils::power(utils::sqrt_2pi, nu[i]) --> sto elevando a potenza una constexpr, il problema è che non so a priori quale sarà l'esponente. sarebbe divertente
		//calcolare un super array lunghissimo di constexpr tipo per sqrt_2pi elevato da 2 a 100
		//lgammal automatically computes the log of gamma function	
		//std::cout<<"result Constant termi = "<<result_const_term<<std::endl;	
		return result_MC + result_const_term; 
	}
	
}


#endif



