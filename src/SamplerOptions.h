#ifndef __SAMPLEROPTIONS_HPP__
#define __SAMPLEROPTIONS_HPP__

#include "include_headers.h"
#include "include_graphs.h"
#include "include_helpers.h"
#include "include_GGM.h"

struct SamplerTraits{
	// RetK is a vector containing the upper triangular part of the precision matrix. It is important to remember that this choice implies that 
	// elements are saved row by row.
	using IdxType  	  		= std::size_t;
	using MatRow      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; 
	using VecRow      		= Eigen::RowVectorXd;
	using VecCol      		= Eigen::VectorXd;
	using CholTypeRow 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	using GroupsPtr   		= std::shared_ptr<const Groups>;
	
	//Used to save sampled values in memory and not on file
	using RetBeta	  		= std::vector<MatCol>;
	using RetMu		  		= std::vector<VecCol>; 
	using RetK	 	  		= std::vector<VecCol>; 
	using RetTaueps	  		= std::vector<double>;
	//using RetGraph 		= std::unordered_map< std::vector<bool>, int>;
	//using RetGraph 		= std::map< std::vector<bool>, int>;
	//using RetGraph       	= std::vector< std::pair< std::vector<bool>, int> >;
	//using RetType	  		= std::tuple<RetBeta, RetMu, RetK, RetGraph, RetTaueps>;
	using IteratorRetBeta	= std::vector<MatCol>::iterator;
	using IteratorRetMu		= std::vector<VecCol>::iterator;
	using IteratorRetK	 	= std::vector<VecCol>::iterator; 
	using IteratorRetTaueps	= std::vector<double>::iterator;
};



class Hyperparameters : public SamplerTraits{
	public:
	Hyperparameters()=default;
	Hyperparameters(unsigned int const & _p):a_tau_eps(2.0*10), b_tau_eps(2.0*0.001), sigma_mu(100.0), p_addrm(0.5), 
											 b_K(3.0), D_K(MatCol::Identity(_p,_p)), sigmaG(0.1), Gprior(0.5){}
	Hyperparameters(double const & _bK, MatCol const & _D, double const & _paddrm, double const & _sigmaG, double const & _Gprior)
					:a_tau_eps(0.0), b_tau_eps(0.0), sigma_mu(1.0), p_addrm(_paddrm), b_K(_bK), D_K(_D), sigmaG(_sigmaG), Gprior(_Gprior){}										 
	Hyperparameters(double const & _a, double const & _b, double const & _sigmamu, double const & _bk, MatCol const & _D, double const & _paddrm, 
					double const & _sigmaG, double const & _Gprior):
					a_tau_eps(_a), b_tau_eps(_b), sigma_mu(_sigmamu), b_K(_bk), D_K(_D), p_addrm(_paddrm), sigmaG(_sigmaG), Gprior(_Gprior){}

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
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), MCiterPrior(_MCiterPr), MCiterPost(_MCiterPost) ,Basemat(_PHI), ptr_groups(_ptrGr), trGwishSampler(1e-8)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}
	Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, 
				unsigned int const & _MCiterPr, unsigned int const & _MCiterPost , MatCol const & _PHI, double const & _trGwishSampler, GroupsPtr const & _ptrGr = nullptr):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), MCiterPrior(_MCiterPr), MCiterPost(_MCiterPost) ,Basemat(_PHI), ptr_groups(_ptrGr), trGwishSampler(_trGwishSampler)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}
	Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, 
				unsigned int const & _MCiterPr, MatCol const & _PHI, GroupsPtr const & _ptrGr):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), MCiterPrior(_MCiterPr), MCiterPost(_MCiterPr), Basemat(_PHI), ptr_groups(_ptrGr), trGwishSampler(1e-8)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}		
	Parameters(int const & _niter, int const & _nburn, double const & _thin, unsigned int const & _MCiterPr, unsigned int const & _MCiterPost, double const & _trGwishSampler):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thin), MCiterPrior(_MCiterPr), MCiterPost(_MCiterPost), ptr_groups(nullptr), trGwishSampler(_trGwishSampler)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}		
	Parameters(int const & _niter, int const & _nburn, double const & _thin, double const & _thinG, MatCol const & _PHI, GroupsPtr const & _ptrGr):
				niter(_niter), nburn(_nburn), thin(_thin), thinG(_thinG), Basemat(_PHI), ptr_groups(_ptrGr), trGwishSampler(1e-8)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
					iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	}							
	Parameters(MatCol const & _PHI):niter(3), nburn(1), thin(1), thinG(1), MCiterPrior(100), MCiterPost(100), Basemat(_PHI), ptr_groups(nullptr), trGwishSampler(1e-8)
	{
	 	iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
	 	iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	} 
	Parameters(MatCol const & _PHI, GroupsPtr const & _ptrGr):niter(3), nburn(1), thin(1), thinG(1), MCiterPrior(100), MCiterPost(100), Basemat(_PHI), ptr_groups(_ptrGr), trGwishSampler(1e-8)
	{
	 	iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
	 	iter_to_storeG = static_cast<unsigned int>((niter - nburn)/thinG);
	} 
		
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
	double trGwishSampler;
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
		str<<"trGwishSampler = "<<pm.trGwishSampler<<std::endl;
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
	Init(unsigned int const & _n, unsigned int const & _p):
			Beta0(MatCol::Zero(_p,_n)), mu0(VecCol::Zero(_p)), tau_eps0(1.0), K0(MatRow::Identity(_p,_p)), G0(_p)
	{
		   	G0.set_empty_graph();
	};

	//This is for block graphs															   
	Init(unsigned int const & _n, unsigned int const & _p, GroupsPtr const & _ptrGr):
			Beta0(MatCol::Zero(_p,_n)), mu0(VecCol::Zero(_p)), tau_eps0(1.0), K0(MatRow::Identity(_p,_p)), G0(_ptrGr)
	{
		G0.set_empty_graph();
	};
	//Explicit constructor. Only for Complete
	Init(MatCol const & _Beta0, VecCol const & _mu0, double const & _tau_eps0, MatRow const & _K0, Graph const & _G0):
			Beta0(_Beta0), mu0(_mu0), tau_eps0(_tau_eps0), K0(_K0) , G0(_G0)
	{
		   if(tau_eps0 == 0)
		   	throw("Initial value for tau_eps0 cannot be 0");

	};

	void set_init(MatCol const & _Beta0, VecCol const & _mu0, double const & _tau_eps0, MatRow const & _K0, Graph const & _G0){
		Beta0 = _Beta0;
		mu0 = _mu0;
		tau_eps0 = _tau_eps0;
		G0 = _G0; 
		K0 = _K0;
	}
	void set_init(MatRow const & _K0, Graph const & _G0){
		G0 = _G0; 
		K0 = _K0;
	}	
	
	MatCol Beta0; //p x n
	VecCol mu0; // p
	double tau_eps0; //scalar
	MatRow K0; // p x p
	Graph  G0;
};


// -----------------------------------------------------------------------------------------------------------------------------------------------

//Generic
template < template <typename> class GraphStructure = GraphType, typename T = unsigned int > 
std::unique_ptr< GGM<GraphStructure, T> > 
SelectMethod_Generic(std::string const & namePr, std::string const & nameGGM, Hyperparameters const & hy, Parameters const & param){
	using Graph = GraphStructure<T>;
	if( !(namePr == "Uniform" || namePr == "Bernoulli" || namePr == "TruncatedUniform" || namePr == "TruncatedBernoulli") )
		throw std::runtime_error("Error, the only possible priors right now are: Uniform, TruncatedUniform, Bernoulli, TruncatedBernoulli");
	//1) Select prior
	std::unique_ptr< GraphPrior<GraphStructure, T> > prior = nullptr;
	if( namePr == "Uniform" )
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Uniform, GraphStructure, T >() );
	else if( namePr == "Bernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Complete, PriorCategory::Bernoulli, GraphStructure, T >(hy.Gprior) );
	if constexpr( internal_type_traits::isBlockGraph<GraphStructure, T>::value ){
		if( namePr == "TruncatedUniform")
		prior = std::move( Create_GraphPrior<PriorType::Truncated,PriorCategory::Uniform,   GraphStructure, T >() );
		else if( namePr == "TruncatedBernoulli")
		prior = std::move( Create_GraphPrior<PriorType::Truncated,PriorCategory::Bernoulli, GraphStructure, T >(hy.Gprior) );
	}
	if(prior == nullptr)
		throw std::runtime_error("Error, the type of selected graph is not compatible with the requested prior. Complete graphs cannot use Truncated priors ");	
	//2) Select algorithm
	if(nameGGM == "MH")
		return Create_GGM<GGMAlgorithm::MH, GraphStructure, T >(prior, hy.b_K, hy.D_K, param.trGwishSampler , param.MCiterPrior, param.MCiterPost);
	else if(nameGGM == "RJ")
		return Create_GGM<GGMAlgorithm::RJ, GraphStructure, T >(prior, hy.b_K, hy.D_K, param.trGwishSampler, hy.sigmaG, param.MCiterPrior);
	else if(nameGGM == "DRJ")
		return Create_GGM<GGMAlgorithm::DRJ,GraphStructure, T >(prior, hy.b_K, hy.D_K, param.trGwishSampler, hy.sigmaG);
	else
		throw std::runtime_error("Error, the only possible GGM algorithm right now are: MH, RJ, DRJ");
}



#endif




