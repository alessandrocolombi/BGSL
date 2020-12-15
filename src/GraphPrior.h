#ifndef __GRAPHPRIORS_HPP__
#define __GRAPHPRIORS_HPP__

#include "include_headers.h"
#include "include_graphs.h"
#include "include_helpers.h"


//Graph may be: GraphType / BlockGraph / BlockGraphAdj
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class GraphPrior
{
		using Graph = GraphStructure<T>;
	public:
		//mettere uno static_assert per il tipo inserito?
		//GraphPrior();
		virtual double Prob(Graph const & G) const = 0;
		virtual double ratio(Graph const & G_num, Graph const & G_den) const = 0;
		virtual double log_ratio(Graph const & G_num, Graph const & G_den) const  = 0;
		virtual std::unique_ptr< GraphPrior<GraphStructure, T> > clone() const = 0;
		virtual ~GraphPrior() = default;	
};

//Note: get_complete_size() returns an unsigned int. This means that G1.get_complete_size() - G2.get_complete_size() is seen as a difference between unsigned int
//if  G1.get_complete_size() is greater no problem, if it lower the result is a big positive number
//Solution -> cast the diffenrence
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int>
class UniformPrior : public GraphPrior<GraphStructure, T>
{
		using Graph = GraphStructure<T>;
	public:
		//UniformPrior();
		double Prob(Graph const & G) const override{
			unsigned int esp( 0.5*G.get_complete_size()*(G.get_complete_size() - 1)  );
			return 1/utils::power(2.0, esp);
		}
		double ratio(Graph const & G_num, Graph const & G_den) const override{
			return 1.0;
		}
		double log_ratio(Graph const & G_num, Graph const & G_den) const override{
			return 0.0;
		}
		std::unique_ptr< GraphPrior<GraphStructure, T> > clone() const override{
			return std::make_unique< UniformPrior<GraphStructure, T> >(*this);
		}
};

//Non è una specializzazione, ha diretto ad un nome suo perché è indipendente da UniformPrior
//Graph may be: BlockGraph / BlockGraphAdj
template<template <typename> class BlockGraphStructure = BlockGraph, typename T = unsigned int>
class TruncatedUniformPrior : public GraphPrior< BlockGraphStructure, T >{
	using BlockGraphType = BlockGraphStructure<T>;
	//Questa static assert mi puzza
	static_assert(std::is_same_v< BlockGraphAdj<unsigned int>, BlockGraphType > || std::is_same_v<BlockGraphAdj<bool>, BlockGraphType > ||
				  std::is_same_v< BlockGraph   <unsigned int>, BlockGraphType > || std::is_same_v<BlockGraph   <bool>, BlockGraphType > ,
				  "Error, only graphs in block form are allowed in a truncated prior");
		
	public:
		//TruncatedUniformPrior();
		double Prob(BlockGraphType const & G) const override{
			unsigned int esp( 0.5*G.get_size()*(G.get_size() - 1) + G.get_size() - G.get_number_singleton() );
			return 1/utils::power(2.0, esp);
		}
		double ratio(BlockGraphType const & G_num, BlockGraphType const & G_den) const override{
			return 1.0;
		}
		double log_ratio(BlockGraphType const & G_num, BlockGraphType const & G_den) const override{
			return 0.0;
		}
		std::unique_ptr< GraphPrior<BlockGraphStructure, T> > clone() const override{
			return std::make_unique<TruncatedUniformPrior<BlockGraphStructure, T> >(*this);
		}
};


//Graph may be: GraphType / BlockGraph / BlockGraphAdj
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class BernoulliPrior : public GraphPrior<GraphStructure, T>
{
		using Graph = GraphStructure<T>;
	public:
		BernoulliPrior(double const & _theta):theta(_theta){
			if(theta <= 0)
				throw std::runtime_error("The parameter of BernoulliPrior has to be strictly positive");
			if(theta >= 1)
				throw std::runtime_error("The parameter of BernoulliPrior has to be strictly less than one");
		}
		double Prob(Graph const & G) const override{
			return utils::power(theta, G.get_n_links())*utils::power(1.0-theta, G.get_possible_links() - G.get_n_links());
		}
		double ratio(Graph const & G_num, Graph const & G_den) const override{
			if(G_num.get_size() != G_den.get_size())
				throw std::runtime_error("Dimension of numerator and denominator graphs are not compatible");
			return ( utils::power(theta,       static_cast<int>(G_num.get_n_links() - G_den.get_n_links()) ) * 
					 utils::power(1.0 - theta, static_cast<int>(G_den.get_n_links() - G_num.get_n_links()) )  );
		}
		double log_ratio(Graph const & G_num, Graph const & G_den) const override{
			return ( static_cast<int>(G_num.get_n_links() - G_den.get_n_links() )*std::log(theta) + 
					 static_cast<int>(G_den.get_n_links() - G_num.get_n_links() )*std::log(1 - theta));
		}
		std::unique_ptr< GraphPrior<GraphStructure, T> > clone() const override{
			return std::make_unique<BernoulliPrior<GraphStructure, T>>(*this);
		}
	private:
		double theta;	
};

//Graph may be: BlockGraph / BlockGraphAdj
template<template <typename> class BlockGraphStructure = BlockGraph, typename T = unsigned int >
class TruncatedBernoulliPrior : public GraphPrior< BlockGraphStructure, T>
{
		using BlockGraphType = BlockGraphStructure<T>;
	public:
		TruncatedBernoulliPrior(double const & _theta):theta(_theta){
			if(theta <= 0)
				throw std::runtime_error("The parameter of Bernoulli Prior has to be strictly positive");
			if(theta >= 1)
				throw std::runtime_error("The parameter of Bernoulli Prior has to be strictly less than one");
		}
		double Prob(BlockGraphType const & G) const override{
			return ( utils::power(theta, G.get_n_block_links()) * 
				     utils::power(1.0 - theta, G.get_possible_block_links() - G.get_n_block_links()) );
		}
		double ratio(BlockGraphType const & G_num, BlockGraphType const & G_den) const override{
			return ( utils::power(theta,       static_cast<int>(G_num.get_n_block_links() - G_den.get_n_block_links()) ) * 
					 utils::power(1.0 - theta, static_cast<int>(G_den.get_n_block_links() - G_num.get_n_block_links()) )  );
		}
		double log_ratio(BlockGraphType const & G_num, BlockGraphType const & G_den) const override{
			return ( static_cast<int>(G_num.get_n_block_links() - G_den.get_n_block_links() )*std::log(theta) + 
					 static_cast<int>(G_den.get_n_block_links() - G_num.get_n_block_links() )*std::log(1 - theta));
		}
		std::unique_ptr< GraphPrior<BlockGraphStructure, T> > clone() const override{
			return std::make_unique<TruncatedBernoulliPrior<BlockGraphStructure, T> >(*this);
		}
	private:
		double theta;	
};

// ----------------------------------------------------------------------------------------------------------------------------------------

enum class PriorType{
	Complete,
	Truncated
};
enum class PriorCategory{
	Uniform,
	Bernoulli
};

template< PriorType Type=PriorType::Complete, PriorCategory Category=PriorCategory::Uniform, 
		  template <typename> class GraphStructure = GraphType, typename T = unsigned int , typename... Args  >
std::unique_ptr< GraphPrior<GraphStructure, T> > Create_GraphPrior(Args&&... args){

	static_assert(Type == PriorType::Complete || Type == PriorType::Truncated,
				  "Error, only possible types are Complete and Truncated");
	static_assert(Category == PriorCategory::Uniform || Category == PriorCategory::Bernoulli,
			      "Error, only possible categories are Uniform and Bernoulli");
	static_assert(std::is_same_v<bool, T> || std::is_same_v<unsigned int, T>,
				  "Error, the only type i can work with are bool and unsigned int.");		
	
	static_assert( std::is_same_v< BlockGraphAdj<unsigned int>, GraphStructure<T> > || std::is_same_v<BlockGraphAdj<bool>, GraphStructure<T> > ||
				   std::is_same_v< BlockGraph   <unsigned int>, GraphStructure<T> > || std::is_same_v<BlockGraph   <bool>, GraphStructure<T> > ||
				   std::is_same_v< GraphType    <unsigned int>, GraphStructure<T> > || std::is_same_v<GraphType    <bool>, GraphStructure<T> > ,
				   "Error, the only graphs that i can manage are BlockGraphAdj, BlockGraph, GraphType, with type bool or unsigned int");
	// std::is_same_v<BlockGraphAdj, GraphStructure> --> error, they are skeletons of types, not type.	
		  		   
	if constexpr(Type == PriorType::Complete){
		if constexpr(Category == PriorCategory::Uniform)
			return std::make_unique< UniformPrior<GraphStructure, T> >(std::forward<Args>(args)...); 
		else
			return std::make_unique< BernoulliPrior<GraphStructure, T> >(std::forward<Args>(args)...);
	}
	else{
		if constexpr(Category == PriorCategory::Uniform)
			return std::make_unique< TruncatedUniformPrior<GraphStructure, T> >(std::forward<Args>(args)...);
		else
			return std::make_unique< TruncatedBernoulliPrior<GraphStructure, T> >(std::forward<Args>(args)...);
	}

}

// ----------------------------------------------------------------------------------------------------------------------------------------






#endif

