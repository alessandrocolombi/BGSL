#ifndef __ADMH_HPP__
#define __ADMH_HPP__

#include "GGM.h"

template<template <typename> class GraphStructure = GraphType, typename T = unsigned int>
class AddRemoveMH : public GGM<GraphStructure, T> {
	
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

		AddRemoveMH(PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _trGwishSampler, unsigned int const & _MCiterPrior = 100, unsigned int const & _MCiterPost = 100): 
					GGM<GraphStructure, T>(_ptr_prior, _b, _D, _trGwishSampler), MCiterPrior(_MCiterPrior), MCiterPost(_MCiterPost){}
		AddRemoveMH(PriorPtr& _ptr_prior,unsigned int const & _p, double const & _trGwishSampler, unsigned int const & _MCiterPrior = 100, unsigned int const & _MCiterPost = 100): 
					GGM<GraphStructure, T>(_ptr_prior, _p, _trGwishSampler), MCiterPrior(_MCiterPrior), MCiterPost(_MCiterPost){}

		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, sample::GSL_RNG const & engine = sample::GSL_RNG() ) override ;
	protected:
		unsigned int MCiterPrior;
		unsigned int MCiterPost;
		double negative_infinity = -std::numeric_limits<double>::infinity();
		double infinity = std::numeric_limits<double>::infinity();
}; 


template<template <typename> class GraphStructure, typename T>
typename GGMTraits<GraphStructure, T>::ReturnType
AddRemoveMH<GraphStructure, T>::operator()(MatCol const & data, unsigned int const & n, 
										   typename GGMTraits<GraphStructure, T>::Graph & Gold, double alpha, sample::GSL_RNG const & engine ) 
{

	using Graph = GraphStructure<T>;

	sample::runif rand;
	//1) Propose new Graph
	auto [Gnew, log_proposal_Graph, mv_type] = this->propose_new_graph(Gold, alpha, engine) ; //mv_type probably is not used here

	//2) Create GWishart wrt Gnew and wrt posterior parameters
	if(!this->data_factorized){
		this->D_plus_U = this->Kprior.get_inv_scale() + data;	
		this->chol_inv_DplusU = this->D_plus_U.llt().solve(MatCol::Identity(data.rows(),data.rows())).llt().matrixU();
		this->data_factorized = true;
	}
	PrecisionType Kpost(this->Kprior.get_shape() + n , this->D_plus_U, this->chol_inv_DplusU );
	//3) Compute log acceptance ratio
	double old_prior_member =  this->Kprior.log_normalizing_constat(Gold.completeview(),MCiterPrior, engine);
	double old_post_member  =  Kpost.log_normalizing_constat(Gold.completeview(),MCiterPost, engine);
	double new_post_member  =  Kpost.log_normalizing_constat(Gnew.completeview(),MCiterPost, engine);
	double new_prior_member =  this->Kprior.log_normalizing_constat(Gnew.completeview(),MCiterPrior, engine);

	double log_acceptance_ratio = this->ptr_prior->log_ratio(Gnew, Gold) + log_proposal_Graph + 
								old_prior_member - old_post_member + new_post_member - new_prior_member;

	
	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 
	//4) Perform the move and return
	int accepted;
	if(rand(engine) < acceptance_ratio){ //move is accepted
		Gold = std::move(Gnew); //Move semantic is available for graphs
		accepted = 1;
	}
	else{ //move is refused
		accepted = 0;
	}
	Kpost.rgwish(Gold.completeview(), this->trGwishSampler, engine); //Sample new matrix. If the move was accepted, Gold is the new graph
	return std::make_tuple(Kpost.get_matrix(), accepted); 
}



#endif


