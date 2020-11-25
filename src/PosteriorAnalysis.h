#ifndef __POSTERIORANALYSIS_HPP__	
#define __POSTERIORANALYSIS_HPP__	

#include "SamplerOptions.h"

namespace analysis{

	using IdxType 	 = SamplerTraits::IdxType;
	using MatRow  	 = SamplerTraits::MatRow;
	using MatCol  	 = SamplerTraits::MatCol;     		
	using VecRow  	 = SamplerTraits::VecRow;	
	using VecCol     = SamplerTraits::VecCol;  		
	using GroupsPtr  = SamplerTraits::GroupsPtr;
	using RetBeta	 = SamplerTraits::RetBeta;
	using RetMu		 = SamplerTraits::RetMu;
	using RetK	 	 = SamplerTraits::RetK;
	using RetTaueps	 = SamplerTraits::RetTaueps;
	using RetGraph   = SamplerTraits::RetGraph;
	using RetType	 = SamplerTraits::RetType;

	//It works both for all types of RetGraphs (vector / map / unordered_map)
	MatRow Compute_plinks(RetGraph const & SampledGraphs, const unsigned int & iter_saved, GroupsPtr const & groups = nullptr){
		// Need to have the same size!
		auto sum_adj = [](std::pair< std::vector<bool>, int> const & a1, std::pair< std::vector<bool>, int> const & a2){
			std::vector<int> res(a1.first.size());
			const std::vector<bool> &v1(a1.first);
			const std::vector<bool> &v2(a2.first);
			const int & freq1 = a1.second;
			const int & freq2 = a2.second;
			std::transform(v1.cbegin(), v1.cend(), v2.cbegin(), res.begin(), 
				[&freq1, &freq2](bool e1, bool e2){
					return (int)e1*freq1 + (int)e2*freq2;
				}  
			);
			return res;
		}; //For weighted sum of adj_list 
		auto sum = [](std::vector<double> const & plinks_vet, std::pair< std::vector<bool>, int> const & a2){
			std::vector<double> res(plinks_vet.size());
			const std::vector<bool> &v2(a2.first);
			const int & freq = a2.second;
			std::transform(plinks_vet.cbegin(), plinks_vet.cend(), v2.cbegin(), res.begin(), 
				[&freq](int e1, bool e2){
					return e1 + (double)e2*freq;
				}  
			);
			return res;
		};
		int n_elem(SampledGraphs.cbegin()->first.size());
		std::vector<double> plinks_adj( std::accumulate(SampledGraphs.cbegin(), SampledGraphs.cend(), std::vector<double> (n_elem, 0.0), sum) );
		
		if( groups == nullptr){ //Assume it is a complete Graph
			GraphType<double> G(plinks_adj);
			MatRow plinks( G.get_graph() );
			plinks /= iter_saved;
			plinks.diagonal().array() = 1;
			return plinks;
		}
		else{ //Assume it is a Block Graph
			BlockGraph<double> G(plinks_adj, groups);
			MatRow plinks( G.get_graph());
			plinks /= iter_saved;
			std::vector<unsigned int> singleton(G.get_pos_singleton());
			if(singleton.size() != 0)
				std::for_each(singleton.cbegin(), singleton.cend(),[&plinks](unsigned int const & pos){plinks(pos, pos)=1.0;});
			return plinks;
		}
	}

	std::tuple<MatCol, VecCol, MatRow, MatRow, double> 
	PointwiseEstimate(RetType const & ret, const int& iter_to_store, const int& iter_to_storeG, GroupsPtr const & groups = nullptr){
		const auto&[SaveBeta, SaveMu, SaveK, SaveG, SaveTaueps] = ret;

		//SaveBeta is a vector of size iter_to_store  with (p x n)-dimensional matrices
		//SaveMu   is a vector of size iter_to_store  with (p)-dimensional eigen vectors
		//SaveK    is a vector of size iter_to_storeG with (p x p)-dimensional matrices
		//SaveTaueps is a std::vector<double> of size iter_to_store    
		//SaveG    is a vector/map/unsigned_map of size iter_to_storeG with std::vector<bool> representing the adjacendy matrix and its frequency

		//It would be better to use std::reduce with execution::par but it is not supported in R and it is difficult to interface with Eigen

		//Beta and Mu
		MatCol MeanBeta(MatCol::Zero(SaveBeta[0].rows(), SaveBeta[0].cols()));
		VecCol MeanMu( VecCol::Zero(SaveMu[0].size()) );
		for(int i = 0; i < SaveBeta.size(); ++i){
			MeanBeta += SaveBeta[i];
			MeanMu 	 += SaveMu[i]; 
		}
		MeanBeta /= iter_to_store;	
		MeanMu /= iter_to_store;
		//Tau_eps
		double MeanTaueps( std::accumulate(  SaveTaueps.begin(), SaveTaueps.end(), 0.0  ) ); 
		MeanTaueps /= iter_to_store;
		//Precision
		MatRow MeanK( MatRow::Zero(SaveK[0].rows(), SaveK[0].cols())  );
		for(int i = 0; i < SaveK.size(); ++i){
			MeanK += SaveK[i];
		}
		MeanK /= iter_to_store;
		//plinks
		MatRow plinks(Compute_plinks(SaveG, iter_to_storeG, groups));
		return std::make_tuple(MeanBeta, MeanMu, MeanK, plinks, MeanTaueps);

	}

}

#endif