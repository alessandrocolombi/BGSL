#ifndef __RJMH_HPP__
#define __RjMH_HPP__

#include "GGM.h"

//This is for block graphs
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class ReversibleJumpsMH : public GGM<GraphStructure, T> {
	public:
		//Typedefs
		using IdxType  	  		= typename GGMTraits<GraphStructure, T>::IdxType;
		using MatRow      		= typename GGMTraits<GraphStructure, T>::MatRow;
		using MatCol      		= typename GGMTraits<GraphStructure, T>::MatCol;
		using VecRow      		= typename GGMTraits<GraphStructure, T>::VecRow;
		using VecCol      		= typename GGMTraits<GraphStructure, T>::VecCol;
		using CholTypeRow 		= typename GGMTraits<GraphStructure, T>::CholTypeRow;
		using CholTypeCol 		= typename GGMTraits<GraphStructure, T>::CholTypeCol;
		using GroupsPtr   		= typename GGMTraits<GraphStructure, T>::GroupsPtr;
		using PriorPtr			= typename GGMTraits<GraphStructure, T>::PriorPtr;
		using Graph 			= typename GGMTraits<GraphStructure, T>::Graph;
		using CompleteType 		= typename GGMTraits<GraphStructure, T>::CompleteType;
		using PrecisionType     = typename GGMTraits<GraphStructure, T>::PrecisionType;
		using ReturnType 		= typename GGMTraits<GraphStructure, T>::ReturnType;
		//Constructors
		ReversibleJumpsMH(	PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _sigma, 
						  	unsigned int const & _MCiterPrior = 100):
						  		GGM<GraphStructure, T>(_ptr_prior, _b, _D), sigma(_sigma), MCiterPrior(_MCiterPrior) {}
		ReversibleJumpsMH(	PriorPtr& _ptr_prior,unsigned int const & _p, double const & _sigma, 
							unsigned int const & _MCiterPrior = 100 ):
						  		GGM<GraphStructure, T>(_ptr_prior, _p), sigma(_sigma), MCiterPrior(_MCiterPrior) {}
		//Methods
		std::tuple<PrecisionType, double, double> RJ(CompleteType const & Gnew_CompleteView, PrecisionType& Kold_prior, MoveType Move);
	
		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, unsigned int seed = 0);
	protected:
		double const sigma; //it is a standard deviation
		unsigned int MCiterPrior;
};



//Specialization for GraphType (i.e CompleteGraphs). No default in specializations
template<typename T >
class ReversibleJumpsMH<GraphType, T> : public GGM<GraphType, T> {
	public:
		//Typedefs
		using IdxType  	  		= typename GGMTraits<GraphType, T>::IdxType;
		using MatRow      		= typename GGMTraits<GraphType, T>::MatRow;
		using MatCol      		= typename GGMTraits<GraphType, T>::MatCol;
		using VecRow      		= typename GGMTraits<GraphType, T>::VecRow;
		using VecCol      		= typename GGMTraits<GraphType, T>::VecCol;
		using CholTypeRow 		= typename GGMTraits<GraphType, T>::CholTypeRow;
		using CholTypeCol 		= typename GGMTraits<GraphType, T>::CholTypeCol;
		using GroupsPtr   		= typename GGMTraits<GraphType, T>::GroupsPtr;
		using PriorPtr			= typename GGMTraits<GraphType, T>::PriorPtr;
		using Graph 			= typename GGMTraits<GraphType, T>::Graph;
		using CompleteType 		= typename GGMTraits<GraphType, T>::CompleteType;
		using PrecisionType     = typename GGMTraits<GraphType, T>::PrecisionType;
		using ReturnType 		= typename GGMTraits<GraphType, T>::ReturnType;
		//Constructors
				ReversibleJumpsMH(	PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _sigma, 
								  	unsigned int const & _MCiterPrior = 100 ):
								  		GGM<GraphType, T>(_ptr_prior, _b, _D), sigma(_sigma), MCiterPrior(_MCiterPrior) {}
				ReversibleJumpsMH(	PriorPtr& _ptr_prior,unsigned int const & _p, double const & _sigma, 
									unsigned int const & _MCiterPrior = 100 ):
								  		GGM<GraphType, T>(_ptr_prior, _p), sigma(_sigma), MCiterPrior(_MCiterPrior){}
		//Methods
		//Create the proposed matrix K'						  
		std::tuple<PrecisionType, double, double> RJ(CompleteType const & Gnew_CompleteView, PrecisionType& Kold_prior, MoveType Move);

		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, unsigned int seed = 0);
	protected:
		double const sigma; //it is a standard deviation
		unsigned int MCiterPrior;
};


//--------------------------------------------------------------------------------------------------------------------------------------------
// Code for Block Graphs
template< template <typename> class GraphStructure , typename T >
std::tuple<typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType, double, double>
ReversibleJumpsMH<GraphStructure, T>::RJ(typename ReversibleJumpsMH<GraphStructure, T>::CompleteType const & Gnew_CompleteView,
										 typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType& Kold_prior, MoveType Move)
{
	using Graph 		= GraphStructure<T>;
	using Container		= std::vector< std::pair<unsigned int, unsigned int> >;
	using Citerator 	= Container::const_iterator;

	//std::cout<<"Sono dentro RJ per quelli a blocchi"<<std::endl;

	std::random_device rd;
    unsigned int seed=rd();
	sample::GSL_RNG engine_gsl(seed);
	unsigned int p(Kold_prior.get_matrix().rows());
				//std::cout<<"p = "<<p<<std::endl;
	//2) Find all the links that are changing in Complete form 
	const std::pair<unsigned int, unsigned int>& changed_link = this->selected_link; //For lighter notation
	Container L(Gnew_CompleteView.map_to_complete(changed_link.first, changed_link.second));
	Citerator it_L = L.cbegin();
				std::cout<<"Link che cambiano (L):"<<std::endl;
				for(auto __v : L)
					std::cout<<"("<<__v.first<<", "<<__v.second<<")"<<" || ";
				std::cout<<std::endl;
	//a) Fill new Phi
	MatRow Phi_new(MatRow::Zero(p,p));
	if(!Kold_prior.isFactorized){
		std::cout<<"Precision is not factorized"<<std::endl;
		Kold_prior.compute_Chol();
	}
	MatRow Phi_old(Kold_prior.get_upper_Chol()); 

	double log_element_proposal{0};
	double log_jacobian{0};
			//std::cout<<"Phi_old:"<<std::endl<<Phi_old<<std::endl;
	//i = 0 case)
	Phi_new(0,0) = Phi_old(0,0);
	for(unsigned int j = 1; j < p; ++j){

		if(Gnew_CompleteView(0,j) == false){ //There is no link. Is it one of the removed one?
			Phi_new(0,j) = 0; 	//Even if it is, it is not a free element and has to be computed by completion operation
			if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair((unsigned int)0,j) == *it_L){
				log_element_proposal +=  (Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j)) ;
				log_jacobian += Phi_old(0,0);
				it_L++;
			}
		}
		else{ //There is a link. Is it the new one?
			if(Move == MoveType::Add && it_L != L.cend() && std::make_pair((unsigned int)0,j) == *it_L){
				Phi_new(0,j) = sample::rnorm()(engine_gsl, Phi_old(0,j), this->sigma); //The new element is a free element
				log_element_proposal += (Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j));
				log_jacobian += Phi_old(0,0);
				it_L++;
			}
			else
				Phi_new(0,j) = Phi_old(0,j);
		}
	}	
			//std::cout<<"Finito i = 0"<<std::endl;
			//std::cout<<"Phi_new:"<<std::endl<<Phi_new<<std::endl;
	//i = 1 case)
	Phi_new(1,1) = Phi_old(1,1);
	for(unsigned int j = 2; j < p; ++j){

		if(Gnew_CompleteView(1,j) == false){ //There is no link. Is it the removed one?
			Phi_new(1,j) = - ( Phi_new(0,1)*Phi_new(0,j) )/Phi_new(1,1); //Even if it is, it is not a free element and has to be computed by completion operation
			if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair((unsigned int)1,j) == *it_L){
				log_element_proposal += (Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j));
				log_jacobian += Phi_old(1,1);
				it_L++;
			}

		}
		else{ //There is a link. Is it the new one?

			if(Move == MoveType::Add && it_L != L.cend() && std::make_pair((unsigned int)1,j) == *it_L){
				Phi_new(1,j) = sample::rnorm()(engine_gsl, Phi_old(1,j), this->sigma); //The new element is a free element
				log_element_proposal +=	(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j));
				log_jacobian += Phi_old(1,1);
				it_L++;
			}
			else
				Phi_new(1,j) = Phi_old(1,j);
		}
	}
			//std::cout<<"Finito i = 1"<<std::endl;
			//std::cout<<"Phi_new:"<<std::endl<<Phi_new<<std::endl;
	//i > 1 case)
	for(unsigned int i = 2; i < p-1; ++i){

		Phi_new(i,i) = Phi_old(i,i);
		//Phi_new = Phi_new.template selfadjointView<Eigen::Upper>(); //Needed in order to be cache friendly 
		//VecRow Psi_i(Phi_new.block(i,0,1,i));
		VecCol Psi_i(Phi_new.block(0,i,i,1));
		for(unsigned int j = i+1; j < p; ++j){
			if(Gnew_CompleteView(i,j) == false){ //There is no link. Is it the removed one?
				//Even if it is, it is not a free element and has to be computed by completion operation
				//Phi_new = Phi_new.template selfadjointView<Eigen::Upper>(); 
				Phi_new(i,j) = - ( Psi_i.dot(VecCol (Phi_new.block(0,j,i,1))) )/Phi_new(i,i);
				if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair(i,j) == *it_L){
					log_element_proposal += (Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j));
					log_jacobian += Phi_old(i,i);
					it_L++;
				}

			}
			else{ //There is a link. Is it the new one?

				if(Move == MoveType::Add && it_L != L.cend() && std::make_pair(i,j) == *it_L){ 
					Phi_new(i,j) = sample::rnorm()(engine_gsl, Phi_old(i,j), this->sigma); //The new element is a free element
					log_element_proposal += (Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j));
					log_jacobian += Phi_old(i,i);
					it_L++;
				}
				else
					Phi_new(i,j) = Phi_old(i,j);
			}
		}
				//std::cout<<"Finito i = "<<i<<std::endl;
				//std::cout<<"Phi_new:"<<std::endl<<Phi_new<<std::endl;
	}
	Phi_new(p-1,p-1) = Phi_old(p-1,p-1); //Last element
			std::cout<<"Phi_new:"<<std::endl<<Phi_new<<std::endl;
			std::cout<<"Phi_old:"<<std::endl<<Phi_old<<std::endl;


	log_element_proposal /= 2*this->sigma*this->sigma;
	log_element_proposal += static_cast<double>(L.size())*(0.5*utils::log_2pi + std::log(std::abs(this->sigma)));
	log_jacobian 		 *= Gnew_CompleteView.get_group_size(changed_link.second);
	return std::make_tuple( 
			PrecisionType (Phi_new.template triangularView<Eigen::Upper>(), Kold_prior.get_shape(), Kold_prior.get_inv_scale(), Kold_prior.get_chol_invD() ),
			log_element_proposal, log_jacobian );
}


template< template <typename> class GraphStructure , typename T >
typename ReversibleJumpsMH<GraphStructure, T>::ReturnType
ReversibleJumpsMH<GraphStructure, T>::operator()(MatCol const & data, unsigned int const & n, 
										  	     typename GGMTraits<GraphStructure, T>::Graph & Gold,  
										         double alpha, unsigned int seed)
{

	using Graph  = GraphStructure<T>;
	
	if(seed==0){
	  std::random_device rd;
	  seed=rd();
	}
	std::default_random_engine engine(seed);
	std::uniform_real_distribution< double > rand(0.,1.);
	double log_acceptance_ratio{0}; 
	//1) Propose new Graph
	auto [Gnew, log_GraphMove_proposal, mv_type] = this->propose_new_graph(Gold, alpha, seed) ;
				std::cout<<std::endl;
				std::cout<<"Gold:"<<std::endl<<Gold<<std::endl;
				std::cout<<"Gnew:"<<std::endl<<Gnew<<std::endl;
	//2) Perform RJ according to the proposed move and graph
	auto [Knew_prior, log_rj_proposal, log_jacobian_mv ] = this->RJ(Gnew.completeview(), this->Kprior, mv_type) ;	
				std::cout<<"Kold_prior.get_matrix():"<<std::endl<<this->Kprior.get_matrix()<<std::endl;
				std::cout<<"Knew_prior.get_matrix():"<<std::endl<<Knew_prior.get_matrix()<<std::endl;
	//3) Compute acceptance probability ratio
	PrecisionType& Kold_prior = this->Kprior; //lighter notation to avoid this every time
	double log_GraphPr_ratio(this->ptr_prior->log_ratio(Gnew, Gold));
	double log_GWishPrConst_ratio(Kold_prior.log_normalizing_constat(Gold.completeview(),MCiterPrior, seed) - 
								  Knew_prior.log_normalizing_constat(Gnew.completeview(),MCiterPrior, seed) );
	//double log_LL_GWishPr_ratio( -0.5 * ((Knew_prior.get_matrix() - Kold_prior.get_matrix())*(Kold_prior.get_inv_scale() + data)).trace()  ); //--> only the diagonal elements are needed in order to compute the diagonal elements. p elements and not p*p
	double log_LL_GWishPr_ratio{0};
	MatRow Knew_Kold(Knew_prior.get_matrix() - Kold_prior.get_matrix());
	MatCol Ddata(Kold_prior.get_inv_scale() + data);
	#pragma omp parallel for reduction(+:log_LL_GWishPr_ratio)
	for(unsigned int i = 0; i < Knew_Kold.rows(); ++i)
		log_LL_GWishPr_ratio += Knew_Kold.row(i)*Ddata.col(i);
	log_LL_GWishPr_ratio /= -2.0;
				std::cout<<"log_GWishPrConst_ratio = "<<log_GWishPrConst_ratio<<std::endl;
				std::cout<<"GraphMove_proposal = "<<std::exp(log_GraphMove_proposal)<<std::endl;
				std::cout<<"K'-K:"<<std::endl<<Knew_prior.get_matrix() - Kold_prior.get_matrix()<<std::endl;
				std::cout<<"log_LL_GWishPr_ratio = "<<log_LL_GWishPr_ratio<<std::endl;
				std::cout<<"log_rj_proposal = "<<log_rj_proposal<<std::endl;
				std::cout<<"log_jacobian_mv = "<<log_jacobian_mv<<std::endl;
	if(mv_type == MoveType::Add)
		log_acceptance_ratio = log_GWishPrConst_ratio + log_GraphPr_ratio + log_GraphMove_proposal + log_LL_GWishPr_ratio + log_rj_proposal + log_jacobian_mv;
	else if(mv_type == MoveType::Remove)
		log_acceptance_ratio = log_GWishPrConst_ratio + log_GraphPr_ratio + log_GraphMove_proposal + log_LL_GWishPr_ratio - log_rj_proposal - log_jacobian_mv;

	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 
				std::cout<<"acceptance_ratio = "<<acceptance_ratio<<std::endl;
	//4) Perform the move and return
	int accepted;
	if( rand(engine) < acceptance_ratio ){
				std::cout<<"Accettato"<<std::endl;
		Gold = Gnew;
		//Gold = std::move(Gnew);
		//this->Kprior = std::move(Knew_prior); //Questo non serve a niente perché tanto devo estrarre e aggiornare la nuova matrice
		accepted = 1;
	}
	else{
				std::cout<<"Rifiutato"<<std::endl;
		accepted = 0;
	}
	//MatRow newPrecisionMat = utils::rgwish(Gold.completeview(), Kold_prior.get_shape() + n, Kold_prior.get_inv_scale() + data , seed );
	this->Kprior.set_matrix(Gold.completeview(), utils::rgwish(Gold.completeview(), Kold_prior.get_shape() + n, Kold_prior.get_inv_scale() + data , seed ) ); //Copia inutile, costruiscila qua dentro
	this->Kprior.compute_Chol();
	return std::make_tuple(this->Kprior.get_matrix(), accepted);
	//return std::make_tuple(utils::rgwish(Gold.completeview(), Kold_prior.get_shape() + n , Kold_prior.get_inv_scale() + data , seed ), accepted );
	//It is easier to use the free function becuase the inv_scale matrix has to be factorized for sure.
	//If the move is accepted, Gold is the new graph
}

//--------------------------------------------------------------------------------------------------------------------------------------------
// Code for Complete Graphs

template< typename T >
std::tuple< typename ReversibleJumpsMH<GraphType, T>::PrecisionType, double, double>
ReversibleJumpsMH<GraphType, T>::RJ(typename ReversibleJumpsMH<GraphType, T>::CompleteType const & Gnew,
									typename ReversibleJumpsMH<GraphType, T>::PrecisionType& Kold_prior, MoveType Move)
{
	using Graph = GraphType<T>;

			//std::cout<<"Sono in RJ per i completi "<<std::endl;
	std::random_device rd;
	unsigned int seed=rd();
	sample::GSL_RNG engine_gsl(seed);
	const unsigned int p(Kold_prior.get_matrix().rows());
	const std::pair<unsigned int, unsigned int>& changed_link = this->selected_link; //For lighter notation
	double log_element_proposal{0};
	//a) Fill new Phi
	MatRow Phi_new(MatRow::Zero(p,p));

	if(!Kold_prior.isFactorized){
		std::cout<<"Precision is not Factorized"<<std::endl;
		Kold_prior.compute_Chol();
	}

	MatRow Phi_old(Kold_prior.get_upper_Chol()); 

	bool found=false;

	//i = 0 case)
	Phi_new(0,0) = Phi_old(0,0);
	for(unsigned int j = 1; j < p; ++j){

		if(Gnew(0,j) == false){ //There is no link. Is it the removed one?
			Phi_new(0,j) = 0; //Even if it is, it is not a free element and has to be computed by completion operation
			if(Move == MoveType::Remove && !found && std::make_pair((unsigned int)0,j) == changed_link){
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j))/(2*this->sigma*this->sigma);
			}

		}
		else{ //There is a link. Is it the new one?
			if(Move == MoveType::Add && !found && std::make_pair((unsigned int)0,j) == changed_link){
				Phi_new(0,j) = sample::rnorm()(engine_gsl, Phi_old(0,j), this->sigma); //The new element is a free element
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j))/(2*this->sigma*this->sigma);
			}
			else
				Phi_new(0,j) = Phi_old(0,j);
		}
	}
				//std::cout<<"fatto i = 0"<<std::endl;
	//i = 1 case)
	Phi_new(1,1) = Phi_old(1,1);
	for(unsigned int j = 2; j < p; ++j){

		if(Gnew(1,j) == false){ //There is no link. Is it the removed one?
			Phi_new(1,j) = - ( Phi_new(0,1)*Phi_new(0,j) )/Phi_new(1,1); //Even if it is, it is not a free element and has to be computed by completion operation
			if(Move == MoveType::Remove && !found && std::make_pair((unsigned int)1,j) == changed_link){
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j))/(2*this->sigma*this->sigma);
			}

		}
		else{ //There is a link. Is it the new one?

			if(Move == MoveType::Add && !found && std::make_pair((unsigned int)1,j) == changed_link){
				found = true;
				Phi_new(1,j) = sample::rnorm()(engine_gsl, Phi_old(1,j), this->sigma); //The new element is a free element
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j))/(2*this->sigma*this->sigma);
			}
			else
				Phi_new(1,j) = Phi_old(1,j);
		}//changed_link è davvero una pair
	}
				//std::cout<<"fatto i = 1"<<std::endl;
				//std::cout<<"Psi_new:"<<std::endl<<Psi_new<<std::endl;
	//i > 1 case)
	for(unsigned int i = 2; i < p-1; ++i){

		Phi_new(i,i) = Phi_old(i,i);
		//Phi_new = Phi_new.template selfadjointView<Eigen::Upper>(); //Needed in order to be cache friendly ---> NOO POI NON È PIU UPPER TRIANGULAR
		//VecRow Psi_i(Phi_new.block(i,0,1,i)); //cache friendly but not longer upper triangular
		VecCol Psi_i(Phi_new.block(0,i,i,1));
		for(unsigned int j = i+1; j < p; ++j){
			if(Gnew(i,j) == false){ //There is no link. Is it the removed one?
				//Phi_new = Phi_new.template selfadjointView<Eigen::Upper>(); //Even if it is, it is not a free element and has to be computed by completion operation
				Phi_new(i,j) = - ( Psi_i.dot(VecCol (Phi_new.block(0,j,i,1))) )/Phi_new(i,i);
				if(Move == MoveType::Remove && !found && std::make_pair(i,j) == changed_link){
					found = true;
					log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
											(Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j))/(2*this->sigma*this->sigma);
				}

			}
			else{ //There is a link. Is it the new one?

				if(Move == MoveType::Add && !found && std::make_pair(i,j) == changed_link){ 
					found = true;
					Phi_new(i,j) = sample::rnorm()(engine_gsl, Phi_old(i,j), this->sigma); //The new element is a free element
					log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
											(Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j))/(2*this->sigma*this->sigma);
				}
				else
					Phi_new(i,j) = Phi_old(i,j);
			}//changed_link è davvero una pair
		}
	}
	Phi_new(p-1,p-1) = Phi_old(p-1,p-1); //Last element
	//b) Compute jacobian
	//double log_jacobian ( std::log(Phi_old(changed_link.first, changed_link.first)) );
	//c) Construct proposed GWishart and return
				//std::cout<<"Phi_old:"<<std::endl<<Phi_old<<std::endl;
				//std::cout<<"Phi_new:"<<std::endl<<Phi_new<<std::endl;
	return std::make_tuple( PrecisionType (Phi_new.template triangularView<Eigen::Upper>(), Kold_prior.get_shape(), Kold_prior.get_inv_scale(), Kold_prior.get_chol_invD() ),
			log_element_proposal, std::log(Phi_old(changed_link.first, changed_link.first)) );
}


template< typename T >
typename ReversibleJumpsMH<GraphType, T>::ReturnType
ReversibleJumpsMH<GraphType, T>::operator()(MatCol const & data, unsigned int const & n, 
										   typename GGMTraits<GraphType, T>::Graph & Gold,  
										   double alpha, unsigned int seed)
{
		
	using Graph = GraphType<T>;

	if(seed==0){
	  std::random_device rd;
	  seed=rd();
	}
	std::default_random_engine engine(seed);
	std::uniform_real_distribution< double > rand(0.,1.);
	double log_acceptance_ratio{0};
	//1) Propose new Graph
	auto [Gnew, log_GraphMove_proposal, mv_type] = this->propose_new_graph(Gold, alpha, seed) ;
			//std::cout<<std::endl;
			//std::cout<<"Gold:"<<std::endl<<Gold<<std::endl;
			//std::cout<<"Gnew:"<<std::endl<<Gnew<<std::endl;
	//2) Perform RJ according to the proposed move and graph
	auto [Knew_prior, log_rj_proposal, log_jacobian_mv ] = this->RJ(Gnew.completeview(), this->Kprior, mv_type) ;	
			//std::cout<<"Kold_prior.get_matrix():"<<std::endl<<this->Kprior.get_matrix()<<std::endl;
			//std::cout<<"Knew_prior.get_matrix():"<<std::endl<<Knew_prior.get_matrix()<<std::endl;
	//3) Compute acceptance probability ratio
	PrecisionType& Kold_prior 			 = this->Kprior; //lighter notation to avoid this every time
	double log_GraphPr_ratio(this->ptr_prior->log_ratio(Gnew, Gold));
	double log_GWishPrConst_ratio(Kold_prior.log_normalizing_constat(Gold.completeview(),MCiterPrior, seed) - 
								  Knew_prior.log_normalizing_constat(Gnew.completeview(),MCiterPrior, seed) );
	//double log_LL_GWishPr_ratio( -0.5 * ((Knew_prior.get_matrix() - Kold_prior.get_matrix())*(Kold_prior.get_inv_scale() + data)).trace()  ); //--> only the diagonal elements are needed in order to compute the diagonal elements. p elements and not p*p
	double log_LL_GWishPr_ratio{0};
	MatRow Knew_Kold(Knew_prior.get_matrix() - Kold_prior.get_matrix());
	MatCol Ddata(Kold_prior.get_inv_scale() + data);
	#pragma omp parallel for reduction(+:log_LL_GWishPr_ratio)
	for(unsigned int i = 0; i < Knew_Kold.rows(); ++i)
		log_LL_GWishPr_ratio += Knew_Kold.row(i)*Ddata.col(i);
	log_LL_GWishPr_ratio /= -2.0;
			//std::cout<<"log_GWishPrConst_ratio = "<<log_GWishPrConst_ratio<<std::endl;
			//std::cout<<"GraphMove_proposal = "<<std::exp(log_GraphMove_proposal)<<std::endl;
			//std::cout<<"K'-K:"<<std::endl<<Knew_prior.get_matrix() - Kold_prior.get_matrix()<<std::endl;
			//std::cout<<"log_LL_GWishPr_ratio = "<<log_LL_GWishPr_ratio<<std::endl;
			//std::cout<<"log_LL_GWishPr_ratio = "<<log_LL_GWishPr_ratio<<std::endl;
			//std::cout<<"log_rj_proposal = "<<log_rj_proposal<<std::endl;
			//std::cout<<"log_jacobian_mv = "<<log_jacobian_mv<<std::endl;
			
	if(mv_type == MoveType::Add)
		log_acceptance_ratio = log_GWishPrConst_ratio + log_GraphPr_ratio + log_GraphMove_proposal + log_LL_GWishPr_ratio + log_rj_proposal + log_jacobian_mv;
	else if(mv_type == MoveType::Remove)
		log_acceptance_ratio = log_GWishPrConst_ratio + log_GraphPr_ratio + log_GraphMove_proposal + log_LL_GWishPr_ratio - log_rj_proposal - log_jacobian_mv;

	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 
				//std::cout<<"acceptance_ratio = "<<acceptance_ratio<<std::endl;
	//4) Perform the move and return
	int accepted;
	if( rand(engine) < acceptance_ratio ){
				//std::cout<<"Accettato"<<std::endl;
		Gold = Gnew;
		//this->Kprior = std::move(Knew_prior); //Questo non serve a niente perché tanto devo estrarre e aggiornare la nuova matrice
		accepted = 1;
	}
	else{
				//std::cout<<"Rifiutato"<<std::endl;
		accepted = 0;
	}
	
	this->Kprior.set_matrix(Gold.completeview(), utils::rgwish(Gold.completeview(), Kold_prior.get_shape() + n, Kold_prior.get_inv_scale() + data , seed ) ); 
	this->Kprior.compute_Chol();
	return std::make_tuple(this->Kprior.get_matrix(),accepted);
	//It is easier to use the free function becuase the inv_scale matrix has to be factorized for sure.
	//If the move is accepted, Gold is the new graph
}






#endif

