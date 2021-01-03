#ifndef __GGMSAMPLER_HPP__
#define __GGMSAMPLER_HPP__

#include "SamplerOptions.h"

//Template possibilities for storing graphs are:
// std::unordered_map< std::vector<bool>, int> / std::map< std::vector<bool>, int> / std::vector< std::pair< std::vector<bool>, int> >
template<template <typename> class GraphStructure = GraphType, typename T = unsigned int, class RetGraph = std::unordered_map< std::vector<bool>, int>>
class GGMsampler : public SamplerTraits
{
	public:
	using Graph 	    = GraphStructure<T>;
	using CompleteType  = typename GGMTraits<GraphStructure, T>::CompleteType;
	using PrecisionType = typename GGMTraits<GraphStructure, T>::PrecisionType;
	using GGMType 	    = std::unique_ptr<GGM<GraphStructure, T>>;
	using RetType 	    = std::tuple<RetK, RetGraph, int, int>;
	//Right now i'm passing way to many parameters and hyperparameters
	GGMsampler( MatCol const & _data, unsigned int const & _n, Parameters const & _params, Hyperparameters const & _hy_params, 
			    Init<GraphStructure, T> const & _init, GGMType & _GGM_method, unsigned int _seed = 0, bool _print_bp = true):
			    data(_data), params(_params), hy_params(_hy_params), ptr_GGM_method(std::move(_GGM_method)) ,init(_init),
				p(_data.rows()), n(_n), grid_pts(_params.Basemat.rows()), seed(_seed), print_bp(_print_bp)
	{
	 	this->check();
	 	if(seed == 0)
	 		seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();	
	} 
	RetType run();

	private:
	void check();
	MatCol data; //p x p
	Parameters params;
	Hyperparameters hy_params;
	GGMType ptr_GGM_method;
	Init<GraphStructure, T> init;
	unsigned int p;
	unsigned int n;
	unsigned int grid_pts;
	unsigned int seed;
	int total_accepted{0};
	int visited{0};
	bool print_bp;

};

template<template <typename> class GraphStructure, typename T, class RetGraph>
void GGMsampler<GraphStructure, T, RetGraph>::check(){
	if( hy_params.p_addrm <= 0 || hy_params.p_addrm >= 1)
		hy_params.p_addrm = 0.5;
	if( ptr_GGM_method->get_inv_scale().cols() != p || data.rows() != p ||
		p != init.K0.cols() || init.K0.cols() != init.K0.rows() || 
		p != init.G0.get_complete_size()) //check for p
			throw std::runtime_error("Error, incoerent number of basis");
}



template<template <typename> class GraphStructure, typename T, class RetGraph >
typename GGMsampler<GraphStructure, T, RetGraph>::RetType 
GGMsampler<GraphStructure, T, RetGraph>::run()
{
	//Typedefs
	GGM<GraphStructure, T> & GGM_method= *ptr_GGM_method;

	// Declare all parameters (makes use of C++17 structured bindings)
	const unsigned int & r = grid_pts;
	const double&  p_addrm   = this->hy_params.p_addrm; 
	const auto &[niter, nburn, thin, thinG, MCiterPrior, MCiterPost, Basemat, ptr_groups, iter_to_store, iter_to_storeG, threshold] = this->params;
	MatRow K = init.K0; 
	Graph  G = init.G0;
	GGM_method.init_precision(G,K); //non serve per ARMH
	total_accepted = 0;
	visited = 0;

	//Random engine and distributions
	sample::GSL_RNG engine(seed);
	sample::rmvnorm rmv; //Covariance parametrization
	sample::rgamma  rGamma;
	//Define all those quantities that can be compute once
	const MatRow Irow(MatRow::Identity(p,p));
	//Structure for return
	RetK	  SaveK; 	 
	RetGraph  SaveG;
	SaveK.reserve(iter_to_storeG);

	//Setup for progress bar, need to specify the total number of iterations
	pBar bar(niter);
	 
	//Start MCMC loop
	for(int iter = 0; iter < niter; iter++){
		
		//Show progress bar
		bar.update(1);
		if(print_bp){
			bar.print(); 
		}
		

		int accepted_mv{0};

		//Graphical Step
		std::tie(K, accepted_mv) = GGM_method(data, n, G, p_addrm, 0); //G is modified inside the function.
				//std::cout<<"SampledK:"<<std::endl<<K<<std::endl;
		total_accepted += accepted_mv;
		//Save
		if(iter >= nburn){
			if((iter - nburn)%thinG == 0){ 
				SaveK.emplace_back(utils::get_upper_part(K));
				std::vector<bool> adj;
				if constexpr( ! std::is_same_v<T, bool>){
					std::vector<T> adj_nobool(G.get_adj_list());
					adj.resize(adj_nobool.size());
					std::transform(adj_nobool.begin(), adj_nobool.end(), adj.begin(), [](T x) { return (bool)x;});
				}
				else{
					adj = G.get_adj_list();
				}

				if constexpr( std::is_same_v< RetGraph, std::vector< std::pair< std::vector<bool>, int> > >){
					auto it = std::find_if(SaveG.begin(), SaveG.end(), [&adj](std::pair< std::vector<bool>, int> const & sg)
														 {return sg.first == adj;});
					if(it == SaveG.end()){
						SaveG.emplace_back(std::make_pair(adj, 1)); 
						visited++;
					}
					else
						it->second++;
				}
				else if constexpr (std::is_same_v< RetGraph, std::map< std::vector<bool>, int> > || 
								   std::is_same_v< RetGraph, std::unordered_map< std::vector<bool>, int> >) 
				{
					auto it = SaveG.find(adj);
					if(it == SaveG.end()){
						SaveG.insert(std::make_pair(adj, 1));
						visited++;
					}
					else
						it->second++;
					//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					// In constexpr if clause the false part is not instanziated only if the argument of the if  
					// statement is a template parameter
					//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				}		
				
			}
		}
	}
	//std::cout<<std::endl<<"FGM sampler has finished"<<std::endl;
	//std::cout<<"Accepted moves = "<<total_accepted<<std::endl;
	//std::cout<<"visited graphs = "<<visited<<std::endl;
	return std::make_tuple(SaveK, SaveG, total_accepted, visited);
}




#endif
