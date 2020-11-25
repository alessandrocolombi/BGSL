#ifndef __SAMPLEROPTIONS_HPP__
#define __SAMPLEROPTIONS_HPP__

#include "include_headers.h"
#include "include_graphs.h"
#include "include_helpers.h"
#include "GraphPrior.h"
#include "GGMFactory.h"
#include "ProgressBar.h"


struct SamplerTraits{
	
	using IdxType  	  		= std::size_t;
	using MatRow      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; 
	using VecRow      		= Eigen::RowVectorXd;
	using VecCol      		= Eigen::VectorXd;
	using CholTypeRow 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	using GroupsPtr   		= std::shared_ptr<const Groups>;
	using RetBeta	  		= std::vector<MatCol>;
	using RetMu		  		= std::vector<VecCol>; //Sarebbe meglio salvarli in una matrice pxiter_to_store cosi poi posso fare operazioni rowwise
	using RetK	 	  		= std::vector<MatRow>; //Non memory friendly perché sto salvando anche la lower part di una matrice simmetrica
	using RetTaueps	  		= std::vector<double>;
	//using RetGraph 			= std::unordered_map< std::vector<bool>, int>;
	//using RetGraph 			= std::map< std::vector<bool>, int>;
	using RetGraph       	= std::vector< std::pair< std::vector<bool>, int> >;
	using RetType	  		= std::tuple<RetBeta, RetMu, RetK, RetGraph, RetTaueps>;
	using IteratorRetBeta	= std::vector<MatCol>::iterator;
	using IteratorRetMu		= std::vector<VecCol>::iterator;
	using IteratorRetK	 	= std::vector<MatRow>::iterator; //Non memory friendly perché sto salvando anche la lower part di una matrice simmetrica
	using IteratorRetTaueps	= std::vector<double>::iterator;

};


class Hyperparameters : public SamplerTraits{
public:
	Hyperparameters()=default;
	Hyperparameters(unsigned int const & _p):a_tau_eps(2.0*10), b_tau_eps(2.0*0.001), sigma_mu(100.0), p_addrm(0.5), 
											 b_K(3.0), D_K(MatCol::Identity(_p,_p)), sigmaG(0.1), Gprior(0.5){}
	Hyperparameters(double const & _a, double const & _b, double const & _sigmamu, double const & _bk, MatCol const & _D, double const & _p, 
					double const & _sigmaG, double const & _Gprior):
					a_tau_eps(_a), b_tau_eps(_b), sigma_mu(_sigmamu), b_K(_bk), D_K(_D), p_addrm(_p), sigmaG(_sigmaG), Gprior(_Gprior){}

	double a_tau_eps;
    double b_tau_eps;
    double sigma_mu;
	double b_K;
	MatCol D_K; //p x p
    double p_addrm; 
    double sigmaG;
    double Gprior;

    friend std::ostream & operator<<(std::ostream &str, Hyperparameters & hp){
    	str<<"a_tau_eps = "<<hp.a_tau_eps<<std::endl;
    	str<<"b_tau_eps = "<<hp.b_tau_eps<<std::endl;
    	str<<"sigma_mu  = "<<hp.sigma_mu<<std::endl;
    	str<<"p_addrm   = "<<hp.p_addrm<<std::endl;
    	str<<"b_K = "<<hp.b_K<<std::endl;
    	str<<"D_K:"<<std::endl<<hp.D_K<<std::endl;
    	str<<"sigmaG = "<<hp.sigmaG<<std::endl;
    	str<<"Gprior = "<<hp.Gprior<<std::endl;
    	return str;
    } 
};


class Parameters : public SamplerTraits{
public:
	Parameters()=default;
	Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, 
				unsigned int const & _MCiterPr, unsigned int const & _MCiterPost , MatCol const & _PHI, GroupsPtr const & _ptrGr = nullptr):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), MCiterPrior(_MCiterPr), MCiterPost(_MCiterPost) ,Basemat(_PHI), ptr_groups(_ptrGr)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}
	Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, 
				unsigned int const & _MCiterPr, MatCol const & _PHI, GroupsPtr const & _ptrGr):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), MCiterPrior(_MCiterPr),Basemat(_PHI), ptr_groups(_ptrGr)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}		
	Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, MatCol const & _PHI, GroupsPtr const & _ptrGr):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), Basemat(_PHI), ptr_groups(_ptrGr)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}							
	Parameters(MatCol const & _PHI):niter(3), nburn(1), thin(1), thinG(1), MCiterPrior(100), MCiterPost(100), Basemat(_PHI), ptr_groups(nullptr)
	{
	 	iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
	 	iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	} 
	Parameters(MatCol const & _PHI, GroupsPtr const & _ptrGr):niter(3), nburn(1), thin(1), thinG(1), MCiterPrior(100), MCiterPost(100), Basemat(_PHI), ptr_groups(_ptrGr)
	{
	 	iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
	 	iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	} 
	//Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, MatCol const & _PHI):
				//niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), Basemat(_PHI)
				//{
					//TBasemat = Basemat.transpose();
				//}			
	int niter;
	int nburn;
	int thin;
	int thinG;
	unsigned int MCiterPrior; 
	unsigned int MCiterPost; 
	MatCol Basemat; //grid_pts x p
	GroupsPtr ptr_groups;
	unsigned int iter_to_store;
	unsigned int iter_to_storeG;
	friend std::ostream & operator<<(std::ostream &str, Parameters & pm){
		str<<"niter = "<<pm.niter<<std::endl;
		str<<"nburn = "<<pm.nburn<<std::endl;
		str<<"thin  = "<<pm.thin<<std::endl;
		str<<"thinG = "<<pm.thinG<<std::endl;
		str<<"Basemat: "<<std::endl<<pm.Basemat<<std::endl;
		str<<"MCiterPrior  = "<<pm.MCiterPrior<<std::endl;
		str<<"MCiterPost  = "<<pm.MCiterPost<<std::endl;
		str<<"iter_to_store  = "<<pm.iter_to_store<<std::endl;
		str<<"iter_to_storeG = "<<pm.iter_to_storeG<<std::endl;
		if(pm.ptr_groups == nullptr)
			str<<"groups = "<<"Not defined"<<std::endl;
		return str;
	} 
};


template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class Init : public SamplerTraits{
	using Graph = GraphStructure<T>;
public:
	//Those constructors may lead to errors if uncorrectly set. Every block graph has to be constructed with its groups	

		//This is for complete graphs
	//Init()=default;	
	Init(unsigned int const & _n, unsigned int const & _p):
			Beta0(MatCol::Zero(_p,_n)), mu0(VecCol::Zero(_p)), tau_eps0(0.0), K0(MatRow::Identity(_p,_p)), G0(_p)
	{
		   	G0.set_empty_graph();
	};
		//This is for block graphs															   
	Init(unsigned int const & _n, unsigned int const & _p, GroupsPtr const & _ptrGr)
		:Beta0(MatCol::Zero(_p,_n)), mu0(VecCol::Zero(_p)), tau_eps0(0.0), K0(MatRow::Identity(_p,_p)), G0(_ptrGr){
			G0.set_empty_graph();
		};
		
	void set_init(MatCol const & _Beta0, VecCol const & _mu0, double const & _tau_eps0, MatRow const & _K0, Graph const & _G0){
		Beta0 = _Beta0;
		mu0 = _mu0;
		tau_eps0 = _tau_eps0;
		G0 = _G0; 
	}	
	
	MatCol Beta0; //p x n
	VecCol mu0; // p
	double tau_eps0; //scalar
	MatRow K0; // p x p
	Graph  G0;
};

// -----------------------------------------------------------------------------------------------------------------------------------------------

//For Complete Graphs (GraphType)
std::unique_ptr< GGM<GraphType, unsigned int> > 
SelectMethod_GraphType(std::string const & namePr, std::string const & nameGGM, Hyperparameters const & hy, Parameters const & param){
	//1) Select prior
	std::unique_ptr< GraphPrior<GraphType, unsigned int> > prior = nullptr;
	if(namePr == "Uniform")
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Uniform,   GraphType, unsigned int >() );
	else if (namePr == "Bernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Bernoulli, GraphType, unsigned int >(hy.Gprior) );
	else
		throw std::runtime_error("Error, the only possible priors right now are: Uniform and Bernoulli");
	//2) Select algorithm
	if(nameGGM == "MH")
		return Create_GGM<GGMAlgorithm::MH, GraphType, unsigned int >(prior, hy.b_K, hy.D_K , param.MCiterPrior, param.MCiterPost);
	else if(nameGGM == "RJ")
		return Create_GGM<GGMAlgorithm::RJ, GraphType, unsigned int >(prior, hy.b_K, hy.D_K, hy.sigmaG, param.MCiterPrior);
	else if(nameGGM == "DRJ")
		return Create_GGM<GGMAlgorithm::DRJ,GraphType, unsigned int >(prior, hy.b_K, hy.D_K, hy.sigmaG);
	else
		throw std::runtime_error("Error, the only possible GGM algorithm right now are: MH, RJ, DRJ");
}

//For Block Graphs (BlockGraph)
std::unique_ptr< GGM<BlockGraph, unsigned int> > 
SelectMethod_BlockGraph(std::string const & namePr, std::string const & nameGGM, Hyperparameters const & hy, Parameters const & param){
	//1) Select prior
	std::unique_ptr< GraphPrior<BlockGraph, unsigned int> > prior = nullptr;
	if( namePr == "Uniform" )
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Uniform,   BlockGraph, unsigned int >() );
	else if( namePr == "Bernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Bernoulli, BlockGraph, unsigned int >(hy.Gprior) );
	else if( namePr == "TruncatedUniform")
		prior = std::move( Create_GraphPrior<PriorType::Truncated,PriorCategory::Uniform,   BlockGraph, unsigned int >() );
	else if( namePr == "TruncatedBernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Truncated,PriorCategory::Bernoulli, BlockGraph, unsigned int >(hy.Gprior) );
	else
		throw std::runtime_error("Error, the only possible priors right now are: Uniform, TruncatedUniform, Bernoulli, TruncatedBernoulli ");
	//2) Select algorithm
	if(nameGGM == "MH")
		return Create_GGM<GGMAlgorithm::MH, BlockGraph, unsigned int >(prior, hy.b_K, hy.D_K , param.MCiterPrior, param.MCiterPost);
	else if(nameGGM == "RJ")
		return Create_GGM<GGMAlgorithm::RJ, BlockGraph, unsigned int >(prior, hy.b_K, hy.D_K, hy.sigmaG, param.MCiterPrior);
	else if(nameGGM == "DRJ")
		return Create_GGM<GGMAlgorithm::DRJ,BlockGraph, unsigned int >(prior, hy.b_K, hy.D_K, hy.sigmaG);
	else
		throw std::runtime_error("Error, the only possible GGM algorithm right now are: MH, RJ, DRJ");
}

//For Block Graphs (BlockGraphAdj)
std::unique_ptr< GGM<BlockGraphAdj, unsigned int> > 
SelectMethod_BlockGraphAdj(std::string const & namePr, std::string const & nameGGM, Hyperparameters const & hy, Parameters const & param){
	//1) Select prior
	std::unique_ptr< GraphPrior<BlockGraphAdj, unsigned int> > prior = nullptr;
	if( namePr == "Uniform" )
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Uniform,   BlockGraphAdj, unsigned int >() );
	else if( namePr == "Bernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Bernoulli, BlockGraphAdj, unsigned int >(hy.Gprior) );
	else if( namePr == "TruncatedUniform")
		prior = std::move( Create_GraphPrior<PriorType::Truncated,PriorCategory::Uniform,   BlockGraphAdj, unsigned int >() );
	else if( namePr == "TruncatedBernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Truncated,PriorCategory::Bernoulli, BlockGraphAdj, unsigned int >(hy.Gprior) );
	else
		throw std::runtime_error("Error, the only possible priors right now are: Uniform, TruncatedUniform, Bernoulli, TruncatedBernoulli ");
	//2) Select algorithm
	if(nameGGM == "MH")
		return Create_GGM<GGMAlgorithm::MH, BlockGraphAdj, unsigned int >(prior, hy.b_K, hy.D_K , param.MCiterPrior, param.MCiterPost);
	else if(nameGGM == "RJ")
		return Create_GGM<GGMAlgorithm::RJ, BlockGraphAdj, unsigned int >(prior, hy.b_K, hy.D_K, hy.sigmaG, param.MCiterPrior);
	else if(nameGGM == "DRJ")
		return Create_GGM<GGMAlgorithm::DRJ,BlockGraphAdj, unsigned int >(prior, hy.b_K, hy.D_K, hy.sigmaG);
	else
		throw std::runtime_error("Error, the only possible GGM algorithm right now are: MH, RJ, DRJ");
}
#endif