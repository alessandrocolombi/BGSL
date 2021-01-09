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
	//std::cout<<"Sono dentro DoubleReversibleJumpsMH"<<std::endl;
	//std::cout<<"  "<<std::endl;
			//if(seed==0){
				////std::random_device rd;
				////seed=rd();
				//seed=static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			//}
			//std::default_random_engine engine(seed);
			//std::uniform_real_distribution< double > rand(0.,1.);
	sample::runif rand;
	double log_acceptance_ratio{0}; 
	//1) Propose a new graph
	auto [Gnew, log_GraphMove_proposal, mv_type] = this->propose_new_graph(Gold, alpha, engine);
	MoveType inverse_mv_type;
	(mv_type == MoveType::Add) ? (inverse_mv_type = MoveType::Remove) : (inverse_mv_type = MoveType::Add);
				//if(mv_type == MoveType::Add)
					//std::cout<<"Aggiungo"<<std::endl;
				//else
					//std::cout<<"Tolgo"<<std::endl;
					//std::cout<<"Gold:"<<std::endl<<Gold<<std::endl;
					//std::cout<<"Gnew:"<<std::endl<<Gnew<<std::endl;
	//2) Sample auxiliary matrix according to Gnew
	CompleteType Gnew_complete(Gnew.completeview());
	CompleteType Gold_complete(Gold.completeview());

	this->Waux.rgwish(Gnew_complete, this->trGwishSampler, engine);
	this->Waux.compute_Chol();
	//3) Perform the first jump with respect to the actual precision matrix K -> K'
			//std::cout<<"Salto di K"<<std::endl;
	//auto [Knew, log_rj_proposal_K, log_jacobian_mv_K ] = this->RJ(Gnew_complete, this->Kprior, mv_type) ;
	auto [Knew, log_rj_proposal_K, log_jacobian_mv_K ] = this->RJ_new(Gnew_complete, this->Kprior, mv_type, engine) ;
					//std::cout<<"Kold:"<<std::endl<<this->Kprior.get_matrix()<<std::endl;
					//std::cout<<"Knew:"<<std::endl<<Knew.get_matrix()<<std::endl;
	//4) Perform the second jump with respect to auxiliary matrix Waux -> W0	
			//std::cout<<"Salto di W"<<std::endl;
	//auto [W0, log_rj_proposal_W, log_jacobian_mv_W ] = this->RJ(Gold_complete, this->Waux, inverse_mv_type) ;
	auto [W0, log_rj_proposal_W, log_jacobian_mv_W ] = this->RJ_new(Gold_complete, this->Waux, inverse_mv_type, engine) ;
					//std::cout<<"Waux:"<<std::endl<<Waux.get_matrix()<<std::endl;
					//std::cout<<"W0:"<<std::endl<<W0.get_matrix()<<std::endl;

	//NOTE: Step 3 is the jump (K,G) --> (K',G') while step 4 is the jump (Waux,G') --> (W0,G0)

	//5) Compute acceptance probability ratio
	auto TraceProd = [](MatRow const & A, MatCol const & B){
		double res{0};
		#pragma omp parallel for reduction(+:res)
		for(unsigned int i = 0; i < A.rows(); ++i)
			res += A.row(i)*B.col(i);
		return( -0.5*res );
	}; //This lambda function computes trace(A*B)
	if(!this->data_factorized){
		this->D_plus_U = this->Kprior.get_inv_scale() + data;	
		this->chol_inv_DplusU = this->D_plus_U.llt().solve(MatCol::Identity(data.rows(),data.rows())).llt().matrixU();
		this->data_factorized = true;
	}
	
	double log_GraphPr_ratio(this->ptr_prior->log_ratio(Gnew, Gold));
	//double log_LL_GWishPr_ratio( -0.5 * ((Knew.get_matrix() - this->Kprior.get_matrix())*(this->Kprior.get_inv_scale() + data)).trace()  );
	double log_LL_GWishPr_ratio(  TraceProd( Knew.get_matrix() - this->Kprior.get_matrix() , this->D_plus_U)  );
	//double log_aux_ratio( -0.5 * ((this->Waux.get_matrix() - W0.get_matrix())*( this->Waux.get_inv_scale() )).trace() );
	double log_aux_ratio( TraceProd(this->Waux.get_matrix() - W0.get_matrix(), this->Waux.get_inv_scale() ) );

			//Questo lo metto solo per fare un confronto
			//double log_GWishPrConst_ratio(this->Kprior.log_normalizing_constat(Gold_complete,500, seed) - 
			//							  Knew.log_normalizing_constat(Gnew_complete,500, seed) );


					//std::cout<<"GraphMove_proposal = "<<std::exp(log_GraphMove_proposal)<<std::endl;
					//std::cout<<"K'-K:"<<std::endl<<Knew.get_matrix() - this->Kprior.get_matrix()<<std::endl;
					//std::cout<<"Waux - W0:"<<std::endl<<this->Waux.get_matrix() - W0.get_matrix()<<std::endl;
					//std::cout<<"log_aux_ratio = "<<log_aux_ratio<<std::endl;
					//std::cout<<"log_LL_GWishPr_ratio = "<<log_LL_GWishPr_ratio<<std::endl;
					//std::cout<<"log_rj_proposal_K = "<<log_rj_proposal_K<<std::endl;
					//std::cout<<"log_rj_proposal_W = "<<log_rj_proposal_W<<std::endl;
					//std::cout<<"log_jacobian_mv_K = "<<log_jacobian_mv_K<<std::endl;
					//std::cout<<"log_jacobian_mv_W = "<<log_jacobian_mv_W<<std::endl;


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
				//std::cout<<"acceptance_ratio = "<<acceptance_ratio<<std::endl;
	//6) Perform the move and return
	int accepted;
	if( rand(engine) < acceptance_ratio ){
				//std::cout<<"Ho accettato"<<std::endl;
		//Gold = Gnew;
		Gold = std::move(Gnew); //Move semantic is available for graphs
		//this->Kprior = std::move(Knew);
		accepted = 1;
	}
	else{
				//std::cout<<"Ho rifiutato"<<std::endl;
		accepted = 0;
	}
	static_assert(	internal_type_traits::isCompleteGraph<CompleteSkeleton, T>::value,
					"QUESTA É LA COSA STRANA");	

	this->Kprior.set_matrix(Gold_complete, 
		utils::rgwish<CompleteSkeleton, T, utils::ScaleForm::CholUpper_InvScale, utils::MeanNorm>(Gold_complete, this->Kprior.get_shape() + n, this->chol_inv_DplusU , this->trGwishSampler, engine ) ); 
	this->Kprior.compute_Chol();
	return std::make_tuple(this->Kprior.get_matrix(),accepted);
	//return std::make_tuple(utils::rgwish(Gold.completeview(), this->Kprior.get_shape(), this->Kprior.get_inv_scale() + data , seed ), accepted );
	//It is easier to use the free function becuase the inv_scale matrix has to be factorized for sure. 
	//--> deve essere fattorizzata ogni volta solo nel FGM sampler, nel GGM sampler in realtà no
	//If the move is accepted, Gold is the new graph
}










//In questo caso non serve specializzare per i GraphType

/*
//Specialization for GraphType (i.e CompleteGraphs). No default in specializations
template<typename T >
class DoubleReversibleJumpsMH<GraphType, T> : public ReversibleJumpsMH<GraphType, T> {
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
		using CompleteType 			= typename GGMTraits<GraphType, T>::CompleteType;
		using GWishartType 		= typename GGMTraits<GraphType, T>::GWishartType;
		using PrecisionType     = typename GGMTraits<GraphType, T>::PrecisionType;
		using ReturnType 		= typename GGMTraits<GraphType, T>::ReturnType;
		//Constructors
		DoubleReversibleJumpsMH(PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _sigma):
						 		 ReversibleJumpsMH<GraphType, T>(_ptr_prior, _b, _D, _sigma), Waux( _b, _D ) {}
		DoubleReversibleJumpsMH(PriorPtr& _ptr_prior,unsigned int const & _p, double const & _sigma):
						 		 ReversibleJumpsMH<GraphType, T>(_ptr_prior, _p, _sigma), Waux(_p){}
		//Methods
		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, unsigned int const & MCiter, unsigned int seed = 0);
	protected:
		PrecisionType Waux; 
};
*/

/*
 // Code for Complete Graphs
template< typename T >
typename DoubleReversibleJumpsMH<GraphType, T>::ReturnType
DoubleReversibleJumpsMH<GraphType, T>::operator()(MatCol const & data, unsigned int const & n, 
										  		  typename GGMTraits<GraphType, T>::Graph & Gold,  
										 		  double alpha, unsigned int const & MCiter, unsigned int seed)
{
	std::cout<<"Sono dentro DoubleReversibleJumpsMH per completi"<<std::endl;
	if(seed==0){
	  std::random_device rd;
	  seed=rd();
	}
	std::default_random_engine engine(seed);
	std::uniform_real_distribution< double > rand(0.,1.);
	double log_acceptance_ratio{0}; //questo deve rimanere
	//1) Propose a new graph
	auto [Gnew, log_GraphMove_proposal] = this->propose_new_graph(Gold, alpha, seed) ;
	//2) Sample auxiliary matrix according to Gnew
	this->Waux.rgwish(Gnew_complete, seed);
	std::cout<<"chi dentro Waux?"<<std::endl;
	std::cout<<"this->Waux.get_matrix():"<<std::endl<<this->Waux.get_matrix()<<std::endl;
	this->Waux.compute_Chol();
	std::cout<<"this->Waux.get_upper_Chol():"<<std::endl<<this->Waux.get_upper_Chol()<<std::endl;

	//3) Perform the first jump with respect to the actual precision matrix K -> K'
	auto [Knew, log_rj_proposal_K, log_jacobian_mv_K ] = this->RJ(Gnew_complete, this->Kprior) ;
	std::cout<<"Uscito da primo salto"<<std::endl;
	//4) Perform the second jump with respect to auxiliary matrix W_tilde -> W0	
	auto [W0, log_rj_proposal_W, log_jacobian_mv_W ] = this->RJ(Gold.completeview(), this->Waux) ;
	std::cout<<"Uscito dal secondo salto"<<std::endl;
	//NOTE: Step 3 is the jump (K, G) --> (K', G') while step 4 is the jump (Waux, G') --> (W0, G0)

	//5) Compute acceptance probability ratio
	double log_GraphPr_ratio(this->ptr_prior->log_ratio(Gnew, Gold));
	std::cout<<"calcolato prior ratio"<<std::endl;
	double log_LL_GWishPr_ratio( -0.5 * ((Knew.get_matrix() - this->Kprior.get_matrix())*(this->Kprior.get_inv_scale() + data)).trace()  );
	std::cout<<"calcolato likelihood * prior K "<<std::endl;
	double log_aux_ratio( -0.5 * ((this->Waux.get_matrix() - W0.get_matrix())*( this->Waux.get_inv_scale() )).trace() );
	std::cout<<" calcolato prior W"<<std::endl;

	if(this->Move == MoveType::Add){
		log_acceptance_ratio = log_GraphPr_ratio + log_GraphMove_proposal + 
							   log_LL_GWishPr_ratio - log_aux_ratio 	  + 
							   log_rj_proposal_K - log_rj_proposal_W	  + 
							   log_jacobian_mv_K - log_jacobian_mv_W	  ;
	}

	else if(this->Move == MoveType::Remove){
		log_acceptance_ratio = log_GraphPr_ratio + log_GraphMove_proposal + 
							   log_LL_GWishPr_ratio - log_aux_ratio 	  + 
							   log_rj_proposal_W - log_rj_proposal_K	  + 
							   log_jacobian_mv_W - log_jacobian_mv_K	  ;
	}
	std::cout<<"calcolato log acceptance_ratio"<<std::endl;
	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 

	//6) Perform the move and return
	int accepted;
	if( rand(engine) < acceptance_ratio ){
		std::cout<<"Ho accettato"<<std::endl;
		Gold = Gnew;
		//Gold = std::move(Gnew);
		this->Kprior = Knew; //oppure piu semplicemente, basta aggiornare la U. Qua faccio un sacco di copie inutili
		//this->Kprior = std::move(Knew);
		accepted = 1;
	}
	else{
		std::cout<<"Ho rifiutato"<<std::endl;
		accepted = 0;
	}
	std::cout<<"Daje che si ritorna "<<std::endl;
	return std::make_tuple(utils::rgwish(Gold.completeview(), this->Kprior.get_shape(), this->Kprior.get_inv_scale() + data , seed ), accepted );
	//It is easier to use the free function becuase the inv_scale matrix has to be factorized for sure.
	//If the move is accepted, Gold is the new graph
	
}

*/

#endif

