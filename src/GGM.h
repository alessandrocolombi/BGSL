#ifndef __GGM_HPP__
#define __GGM_HPP__

#include "include_headers.h"
#include "include_graphs.h"
#include "include_helpers.h"
#include "GraphPrior.h"

template<template <typename> class GraphStructure = GraphType, typename T = unsigned int>
struct GGMTraits{

	using IdxType  	  		= std::size_t;
	using MatRow      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; 
	using VecRow      		= Eigen::RowVectorXd;
	using VecCol      		= Eigen::VectorXd;
	using CholTypeRow 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	using GroupsPtr   		= std::shared_ptr<const Groups>;
	using PriorPtr			= std::unique_ptr<GraphPrior<GraphStructure, T>>;
	using Graph 			= GraphStructure<T>;
	//Watch out here, using template aliasing for template parameters
	template<typename S>
	using CompleteSkeleton = typename internal_type_traits::Complete_skeleton<GraphStructure>::template CompleteSkeleton<S>;
	template<typename SS>
	using CompleteSkeleton2 = typename internal_type_traits::Complete_type<GraphStructure, T>::template CompleteSkeleton<SS>;
	using CompleteType 		= typename internal_type_traits::Complete_type<GraphStructure, T>::CompleteType;
	using PrecisionType     = GWishart ;
	using ReturnType 		= std::tuple<MatRow, int>; //The last int stands for move accepted (1) or refused (0)	
};


enum class MoveType{
	Add, Remove
};

template<template <typename> class GraphStructure = GraphType, typename T = unsigned int>
class GGM : public GGMTraits<GraphStructure, T> {
	
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
	using Graph				= typename GGMTraits<GraphStructure, T>::Graph;
	using CompleteType		= typename GGMTraits<GraphStructure, T>::CompleteType;
	using ReturnType		= typename GGMTraits<GraphStructure, T>::ReturnType;
	using PrecisionType     = typename GGMTraits<GraphStructure, T>::PrecisionType;
	
	
	public:
		//Constructors
		GGM(PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _trGwishSampler):
				ptr_prior(std::move(_ptr_prior)),  Kprior( _b, _D ), trGwishSampler(_trGwishSampler) {}
		GGM(PriorPtr& _ptr_prior,unsigned int const & _p, double const & _trGwishSampler): 
				ptr_prior(std::move(_ptr_prior)), Kprior(_p), trGwishSampler(_trGwishSampler) {}
		GGM(GGM & _ggm):
				ptr_prior(_ggm.ptr_prior->clone()), Kprior(_ggm.Kprior), trGwishSampler(_ggm.trGwishSampler){}
		GGM(GGM &&) = default;

		//Operators
		GGM & operator=(const GGM & _ggm){
			if(&_ggm != this){
				ptr_prior.reset(nullptr);
				ptr_prior = _ggm.ptr->clone();
				Kprior = _ggm.Kprior;
			}
			return *this;
		}
		GGM & operator=(GGM &&)=default;

		//Getters
		double get_shape()const{
			return Kprior.get_shape();
		}
		MatCol get_inv_scale()const{
			return Kprior.get_inv_scale();
		}
		//Initialize precision matrix
		inline void init_precision(Graph & G, MatRow const & mat){
			Kprior.set_matrix(G.completeview(), mat);
			Kprior.compute_Chol();
			if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
				positions.resize(G.get_possible_links());
			}
			else{
				positions.resize(G.get_possible_block_links());
			}
			std::iota(positions.begin(), positions.end(), 0); //fill with increasing values
		}

		//This method takes the current graph (both form are accepted) and the probability of selecting an addition and return a tuple,
		//with the proposed new graph, a double that is the (log) proposal ratio and the type of selected move
		std::tuple<Graph, double, MoveType>  propose_new_graph(Graph & Gold, double alpha, sample::GSL_RNG const & engine = sample::GSL_RNG()); 
		
		virtual ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, sample::GSL_RNG const & engine = sample::GSL_RNG()) = 0;
		virtual ~GGM() = default;

		bool data_factorized = false;
	protected:
		PriorPtr ptr_prior; 
		PrecisionType Kprior;
		double trGwishSampler;
		std::vector<unsigned int> positions; //vector storing all possible positions. --> ones union zeros = positions
		std::pair<unsigned int, unsigned int> selected_link; 
		MatCol D_plus_U;
		MatCol chol_inv_DplusU;
		
};


template<template <typename> class GraphStructure, typename T>
std::tuple< typename GGMTraits<GraphStructure, T>::Graph, double, MoveType> 
GGM<GraphStructure, T>::propose_new_graph(typename GGMTraits<GraphStructure, T>::Graph & Gold, double alpha, sample::GSL_RNG const & engine ){
	
	using Graph = GraphStructure<T>;
	//Initialize useful quantities
	if(alpha >= 1.0 || alpha <= 0.0){
	  alpha = 0.5;
	}

	sample::runif rand;
	sample::runif_int rand_int;

	double log_proposal_Graph;
	MoveType Move;
	//Select move type
	if(Gold.get_n_links() == 0){
				//Empty graph, add
		Move = MoveType::Add;
	}
	else if(Gold.get_n_links() == Gold.get_possible_links()){
				//Full graph, remove
		Move = MoveType::Remove;
	}
	else{
				//Not empty nor full, choose by change
		(rand(engine) < alpha) ? (Move = MoveType::Add) : (Move = MoveType::Remove);
	}

	//Select link/block to be added or removed

	if(Move == MoveType::Add){
			//Compute log_proposal_ratio
			log_proposal_Graph = std::log(1-alpha)-std::log(alpha);
			if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
				log_proposal_Graph += std::log(Gold.get_possible_links() - Gold.get_n_links()) - std::log(1 + Gold.get_n_links());
			}
			else{
				log_proposal_Graph += std::log(Gold.get_possible_block_links() - Gold.get_n_block_links()) - std::log(1 + Gold.get_n_block_links());
			}
			//Select the edge to add
			std::vector<unsigned int> adj_list(Gold.get_adj_list());
			std::vector<unsigned int> zeros;
			if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
				zeros.resize(Gold.get_possible_links()-Gold.get_n_links());
			}
			else{
				zeros.resize(Gold.get_possible_block_links() - Gold.get_n_block_links());
			}

			std::copy_if(positions.cbegin(), positions.cend(), zeros.begin(), [&adj_list](unsigned int const & pos){return (bool)adj_list[pos]==false;}); //fill vector zeros with the positions where zeros are stored
			
			unsigned int selected{zeros[rand_int(engine, zeros.size())]};
			adj_list[selected] = 1; //set value in ad_list to true, i.e create a link
			selected_link = Gold.pos_to_ij(selected);
			if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
				return std::make_tuple( Graph (adj_list),  log_proposal_Graph, Move );
			}
			else{
				return std::make_tuple( Graph (adj_list, Gold.get_ptr_groups()), log_proposal_Graph, Move );
			}
			
	}
	else{
		//Compute log proposal ratio
		log_proposal_Graph = std::log(alpha)-std::log(1 - alpha);
		if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
			log_proposal_Graph += std::log(Gold.get_n_links()) - std::log(1 + Gold.get_possible_links() - Gold.get_n_links());
		}
		else{
			log_proposal_Graph += std::log(Gold.get_n_block_links()) - std::log(1 + Gold.get_possible_block_links() - Gold.get_n_block_links());
		}
		std::vector<unsigned int> adj_list(Gold.get_adj_list());
		std::vector<unsigned int> ones;
		if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
			ones.resize(Gold.get_n_links());
		}
		else{
			ones.resize(Gold.get_n_block_links());
		}
		std::copy_if(positions.cbegin(), positions.cend(), ones.begin(), [&adj_list](unsigned int const & pos){return (bool)adj_list[pos]==true;});

		unsigned int selected{ones[rand_int(engine, ones.size())]};
		adj_list[selected] = 0; //set value in ad_list to true, i.e create a link
		selected_link = Gold.pos_to_ij(selected);
		if constexpr(internal_type_traits::isCompleteGraph<GraphStructure,T>::value){
			return std::make_tuple( Graph (adj_list), log_proposal_Graph, Move );
		}
		else{
			return std::make_tuple( Graph (adj_list, Gold.get_ptr_groups()), log_proposal_Graph, Move );
		}
	}
}




#endif