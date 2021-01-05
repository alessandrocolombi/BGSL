#ifndef __GGMFACTORY_HPP__
#define __GGMFACTORY_HPP__

#include "GGM.h"
#include "AddRemoveMH.h"
#include "RJMH.h"
#include "DRJMH.h"


enum class GGMAlgorithm{
	MH,
	RJ,
	DRJ
};

template< GGMAlgorithm algo = GGMAlgorithm::MH, 
		  template <typename> class GraphStructure = GraphType, typename T = unsigned int , typename... Args  >
std::unique_ptr< GGM<GraphStructure, T> > Create_GGM(Args&&... args){

	static_assert(algo == GGMAlgorithm::MH || algo == GGMAlgorithm::RJ || algo == GGMAlgorithm::DRJ,
			      "Error, only possible algorithms are MH, RJ and DRJ");
	static_assert(std::is_same_v<bool, T> || std::is_same_v<int, T>  || std::is_same_v<unsigned int, T>,
				  "Error, the only type i can work with are bool, unsigned int and int .");	

	/*
		static_assert( std::is_same_v< BlockGraphAdj<T>, GraphStructure<T> > || 
					   std::is_same_v< BlockGraph   <T>, GraphStructure<T> > || 
					   std::is_same_v< GraphType    <T>, GraphStructure<T> >  ,
					   "Error, the only graphs that i can manage are BlockGraphAdj, BlockGraph, GraphType");
		// std::is_same_v<BlockGraphAdj, GraphStructure> --> error, they are skeletons of types, not type.	
	*/
	if constexpr(algo == GGMAlgorithm::MH)
		return std::make_unique< AddRemoveMH<GraphStructure, T> >(std::forward<Args>(args)...); 
	else if constexpr(algo == GGMAlgorithm::RJ)
		return std::make_unique< ReversibleJumpsMH<GraphStructure, T> >(std::forward<Args>(args)...);
	else if constexpr(algo == GGMAlgorithm::DRJ)
		return std::make_unique< DoubleReversibleJumpsMH<GraphStructure, T> > (std::forward<Args>(args)...);

}


// ----------------------------------------------------------------------------------------------------------------------------------------------------


#endif