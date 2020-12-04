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

		AddRemoveMH(PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, unsigned int const & _MCiterPrior = 100, unsigned int const & _MCiterPost = 100): 
					GGM<GraphStructure, T>(_ptr_prior, _b, _D), MCiterPrior(_MCiterPrior), MCiterPost(_MCiterPost){}
		AddRemoveMH(PriorPtr& _ptr_prior,unsigned int const & _p, unsigned int const & _MCiterPrior = 100, unsigned int const & _MCiterPost = 100): 
					GGM<GraphStructure, T>(_ptr_prior, _p), MCiterPrior(_MCiterPrior), MCiterPost(_MCiterPost){}

		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, unsigned int seed = 0) override ;
	protected:
		unsigned int MCiterPrior;
		unsigned int MCiterPost;
		double negative_infinity = -std::numeric_limits<double>::infinity();
		double infinity = std::numeric_limits<double>::infinity();
}; 


template<template <typename> class GraphStructure, typename T>
typename GGMTraits<GraphStructure, T>::ReturnType
AddRemoveMH<GraphStructure, T>::operator()(MatCol const & data, unsigned int const & n, 
										   typename GGMTraits<GraphStructure, T>::Graph & Gold, double alpha, unsigned int seed) 
{

	using Graph = GraphStructure<T>;
	
			//std::cout<<"Sono dentro ad AddRemoveMH ()"<<std::endl;
	if(seed==0){
	  std::random_device rd;
	  seed=rd();
	}
	std::default_random_engine engine(seed);
	std::uniform_real_distribution<double> rand(0.,1.);
	//1) Propose new Graph
	auto [Gnew, log_proposal_Graph, mv_type] = this->propose_new_graph(Gold, alpha, seed) ; //mv_type probably wont be used here
			//std::cout<<std::endl;
			//std::cout<<"Gold:"<<std::endl<<Gold<<std::endl;
			//std::cout<<"Gnew:"<<std::endl<<Gnew<<std::endl;

	//2) Create GWishart wrt Gnew and wrt posterior parameters
	PrecisionType Kpost(this->Kprior.get_shape() + n , this->Kprior.get_inv_scale() + data );
	//3) Compute log acceptance ratio
	/*
	double old_prior = utils::log_normalizing_constat(Gold.completeview(), this->Kprior.get_shape(), this->Kprior.get_inv_scale(), MCiterPrior);
	double old_post = utils::log_normalizing_constat(Gold.completeview(),  this->Kprior.get_shape()+n, this->Kprior.get_inv_scale()+data, MCiterPrior);
	double new_post = utils::log_normalizing_constat(Gnew.completeview(), this->Kprior.get_shape()+n, this->Kprior.get_inv_scale()+data, MCiterPost);
	double new_prior =  utils::log_normalizing_constat(Gnew.completeview(), this->Kprior.get_shape(), this->Kprior.get_inv_scale(), MCiterPrior);
	double log_acceptance_ratio( new_post - new_prior - old_post + old_prior ); 
	*/
	/*
	double old_prior_member =  this->Kprior.log_normalizing_constat(Gold.completeview(),MCiterPrior, seed);
	double old_post_member  =  Kpost.log_normalizing_constat(Gold.completeview(),MCiterPost, seed);
	double new_post_member  =  Kpost.log_normalizing_constat(Gnew.completeview(),MCiterPost, seed);
	double new_prior_member =  this->Kprior.log_normalizing_constat(Gnew.completeview(),MCiterPrior, seed);
	*/


	double log_acceptance_ratio(this->ptr_prior->log_ratio(Gnew, Gold) + log_proposal_Graph +
								this->Kprior.log_normalizing_constat(Gold.completeview(),MCiterPrior, seed)  - 
								Kpost.log_normalizing_constat(Gold.completeview(),MCiterPost, seed)   +
								Kpost.log_normalizing_constat(Gnew.completeview(),MCiterPost, seed)   - 
								this->Kprior.log_normalizing_constat(Gnew.completeview(),MCiterPrior, seed)  );
	
	

				//std::cout<<"proposal_Graph = "<< std::exp( log_proposal_Graph ) <<std::endl;
	/*
	std::cout<<std::endl;
	std::cout<<"old post = "<<old_post<<std::endl;
	std::cout<<"old post member = "<<old_post_member<<std::endl;
	std::cout<<"old prior = "<<old_prior<<std::endl;
	std::cout<<"old prior member = "<<old_prior_member<<std::endl;
	std::cout<<"new post = "<<new_post<<std::endl;
	std::cout<<"new post member = "<<new_post_member<<std::endl;
	std::cout<<"new prior = "<<new_prior<<std::endl;
	std::cout<<"new prior member = "<<new_prior_member<<std::endl;
	std::cout<<"Old ratio = "<<old_post - old_prior<<std::endl;
	std::cout<<"Old ratio member = "<<old_post_member - old_prior_member<<std::endl;
	std::cout<<"New ratio = "<<new_post - new_prior<<std::endl;
	std::cout<<"New ratio member = "<<new_post_member - new_prior_member<<std::endl;
	*/

	
	
				//std::cout<<"log acceptance ratio = "<<log_acceptance_ratio<<std::endl;
	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 
	//4) Perform the move and return
	if(rand(engine) < acceptance_ratio){
				//std::cout<<"Accepted"<<std::endl;
		Gold = Gnew;
		Kpost.rgwish(Gnew.completeview(), seed); 
		return std::make_tuple(Kpost.get_matrix(), 1); //<------------------------------------------------------------------------------
		//return std::make_tuple(this->Kprior.get_matrix(), 1); //<------------------------------------------------------------------------------
	}
	else{
			//std::cout<<"Refused"<<std::endl;
		Kpost.rgwish(Gold.completeview(), seed); 
		return std::make_tuple(Kpost.get_matrix(), 0);
		//return std::make_tuple(this->Kprior.get_matrix(), 0);
	}
}



#endif


