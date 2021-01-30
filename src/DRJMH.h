#ifndef __DRJMH_HPP__
#define __DRjMH_HPP__

#include "GGM.h"

//This is for block graphs
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class DoubleReversibleJumpsMH : public ReversibleJumpsMH<GraphStructure, T> {
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
		template <typename S>
		using CompleteSkeleton  = typename GGMTraits<GraphStructure, T>::template CompleteSkeleton<S>; //Need the skelethon of CompleteForm
		using CompleteType 		= typename GGMTraits<GraphStructure, T>::CompleteType;
		using PrecisionType     = typename GGMTraits<GraphStructure, T>::PrecisionType;
		using ReturnType 		= typename GGMTraits<GraphStructure, T>::ReturnType;
		//Constructors
		DoubleReversibleJumpsMH(	PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _trGwishSampler, double const & _sigma, unsigned int const & _MCiterPrior = 0):
						 		 		ReversibleJumpsMH<GraphStructure, T>(_ptr_prior, _b, _D, _trGwishSampler, _sigma, _MCiterPrior), Waux( _b, _D ) {}
		DoubleReversibleJumpsMH(	PriorPtr& _ptr_prior,unsigned int const & _p, double const & _trGwishSampler, double const & _sigma, unsigned int const & _MCiterPrior = 0):
						 			 ReversibleJumpsMH<GraphStructure, T>(_ptr_prior, _p, _trGwishSampler, _sigma, _MCiterPrior), Waux(_p){}
		//Methods
		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, sample::GSL_RNG const & engine = sample::GSL_RNG() );
	protected:
		PrecisionType Waux; 
};




template< template <typename> class GraphStructure , typename T >
typename DoubleReversibleJumpsMH<GraphStructure, T>::ReturnType
DoubleReversibleJumpsMH<GraphStructure, T>::operator()(MatCol const & data, unsigned int const & n, 
										  	    	   typename GGMTraits<GraphStructure, T>::Graph & Gold,  
										         	   double alpha, sample::GSL_RNG const & engine )
{

	sample::runif rand;
	double log_acceptance_ratio{0}; 
	//1) Propose a new graph
	auto [Gnew, log_GraphMove_proposal, mv_type] = this->propose_new_graph(Gold, alpha, engine);
	MoveType inverse_mv_type;
	(mv_type == MoveType::Add) ? (inverse_mv_type = MoveType::Remove) : (inverse_mv_type = MoveType::Add);
	//2) Sample auxiliary matrix according to Gnew
	CompleteType Gnew_complete(Gnew.completeview());
	CompleteType Gold_complete(Gold.completeview());

	this->Waux.rgwish(Gnew_complete, this->trGwishSampler, engine);
	this->Waux.compute_Chol();
	//3) Perform the first jump with respect to the actual precision matrix K -> K'
						//auto [Knew, log_rj_proposal_K, log_jacobian_mv_K ] = this->RJ(Gnew_complete, this->Kprior, mv_type) ;
	auto [Knew, log_rj_proposal_K, log_jacobian_mv_K ] = this->RJ_new(Gnew_complete, this->Kprior, mv_type, engine) ;
	//4) Perform the second jump with respect to auxiliary matrix Waux -> W0	
						//auto [W0, log_rj_proposal_W, log_jacobian_mv_W ] = this->RJ(Gold_complete, this->Waux, inverse_mv_type) ;
	auto [W0, log_rj_proposal_W, log_jacobian_mv_W ] = this->RJ_new(Gold_complete, this->Waux, inverse_mv_type, engine) ;
	//NOTE: Step 3 is the jump (K,G) --> (K',G') while step 4 is the jump (Waux,G') --> (W0,G0)

	//5) Compute acceptance probability ratio
	auto TraceProd = [](MatRow const & A, MatCol const & B){
		double res{0};
		#pragma omp parallel for reduction(+:res)
		for(unsigned int i = 0; i < A.rows(); ++i)
			res += A.row(i)*B.col(i);
		return( -0.5*res );
	}; //This lambda function computes trace(A*B)

	//Check if data is changing or not. if not, do not need to factorize every time
	if(!this->data_factorized){
		this->D_plus_U = this->Kprior.get_inv_scale() + data;	
		this->chol_inv_DplusU = this->D_plus_U.llt().solve(MatCol::Identity(data.rows(),data.rows())).llt().matrixU();
		this->data_factorized = true;
	}
	
	double log_GraphPr_ratio(this->ptr_prior->log_ratio(Gnew, Gold));
	double log_LL_GWishPr_ratio(  TraceProd( Knew.get_matrix() - this->Kprior.get_matrix() , this->D_plus_U)  );
	double log_aux_ratio( TraceProd(this->Waux.get_matrix() - W0.get_matrix(), this->Waux.get_inv_scale() ) );

	if(mv_type == MoveType::Add){
		log_acceptance_ratio = log_GraphPr_ratio + log_GraphMove_proposal + 
							   log_LL_GWishPr_ratio - log_aux_ratio 	  + 
							   log_rj_proposal_K - log_rj_proposal_W	  + 
							   log_jacobian_mv_K - log_jacobian_mv_W	  ;
	}

	else if(mv_type == MoveType::Remove){
		log_acceptance_ratio = log_GraphPr_ratio + log_GraphMove_proposal + 
							   log_LL_GWishPr_ratio - log_aux_ratio 	  + 
							   log_rj_proposal_W - log_rj_proposal_K	  + 
							   log_jacobian_mv_W - log_jacobian_mv_K	  ;
	}

	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 
	//6) Perform the move and return
	int accepted;
	if( rand(engine) < acceptance_ratio ){//Accepted
		//Gold = Gnew;
		Gold = std::move(Gnew); //Move semantic is available for graphs
		accepted = 1;
	}
	else{//Not accepted
		accepted = 0;
	}
	//If the move is accepted, Gold is the new graph
	this->Kprior.set_matrix(Gold_complete, 
		utils::rgwish<CompleteSkeleton, T, utils::ScaleForm::CholUpper_InvScale, utils::MeanNorm>(Gold_complete, this->Kprior.get_shape() + n, this->chol_inv_DplusU , this->trGwishSampler, engine ) ); 
	this->Kprior.compute_Chol();
	return std::make_tuple(this->Kprior.get_matrix(),accepted);

	
}


#endif

