#ifndef __FLMSAMPLEROPTIONS_HPP__
#define __FLMSAMPLEROPTIONS_HPP__

#include "include_headers.h"
#include "include_helpers.h"
#include "include_graphs.h"
#include "ProgressBar.h"


struct FLMsamplerTraits{
	
	using IdxType  	  		= std::size_t;
	using MatRow      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using MatCol      		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; 
	using VecRow      		= Eigen::RowVectorXd;
	using VecCol      		= Eigen::VectorXd;
	using CholTypeRow 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Upper>;
	using CholTypeCol 		= Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Lower>;
	using RetBeta	  		= std::vector<MatCol>; 
	using RetMu		  		= std::vector<VecCol>; //Sarebbe meglio salvarli in una matrice pxiter_to_store cosi poi posso fare operazioni rowwise
	using RetTauK	 	  	= std::vector<VecCol>; 
	using RetK	 	  		= std::vector<MatRow>; 
	using RetTaueps	  		= std::vector<double>;
	//using RetType 			= std::tuple<RetBeta, RetMu, RetTauK, RetTaueps>;
	using IteratorRetBeta	= std::vector<MatCol>::iterator;
	using IteratorRetMu		= std::vector<VecCol>::iterator;
	using IteratorRetTauK	= std::vector<VecCol>::iterator; 
	using IteratorRetTaueps	= std::vector<double>::iterator;
};

class FLMHyperparameters : public FLMsamplerTraits{
	public:
	FLMHyperparameters()=default;
	FLMHyperparameters(unsigned int const & _p):
					a_tau_eps(2.0*10), b_tau_eps(2.0*0.001), sigma_mu(100.0), a_tauK(a_tau_eps), b_tauK(b_tau_eps), bK(3.0), DK(MatCol::Identity(_p,_p)){}									 
	FLMHyperparameters(double const & _a, double const & _b, double const & _sigmamu, double const & _aK, double const & _bk ):
					a_tau_eps(_a), b_tau_eps(_b), sigma_mu(_sigmamu), a_tauK(_aK), b_tauK(_bk){}
	FLMHyperparameters(double const & _a, double const & _b, double const & _sigmamu, double const & _bK, MatCol const & _DK ):
					a_tau_eps(_a), b_tau_eps(_b), sigma_mu(_sigmamu), a_tauK(1.0), b_tauK(1.0), bK(_bK), DK(_DK){}
	double a_tau_eps;
    double b_tau_eps;
    double sigma_mu;
	double a_tauK;
	double b_tauK;
	double bK;
	MatCol DK;

    friend std::ostream & operator<<(std::ostream &str, FLMHyperparameters & hp){
    	str<<"a_tau_eps = "<<hp.a_tau_eps<<std::endl;
    	str<<"b_tau_eps = "<<hp.b_tau_eps<<std::endl;
    	str<<"sigma_mu  = "<<hp.sigma_mu<<std::endl;
    	str<<"a_tauK   = "<<hp.a_tauK<<std::endl;
    	str<<"b_tauK = "<<hp.b_tauK<<std::endl;
    	str<<"bK   = "<<hp.bK<<std::endl;
    	str<<"DK = "<<hp.DK<<std::endl;
    	return str;
    } 
};


class FLMParameters : public FLMsamplerTraits{
	public:
	FLMParameters()=default;
	FLMParameters(int const & _niter, int const & _nburn, double const & _thin, MatCol const & _PHI, double const & _trGwishSampler = 1e-8):
				niter(_niter), nburn(_nburn), thin(_thin), Basemat(_PHI), trGwishSampler(_trGwishSampler)
	{
					iter_to_store  = static_cast<unsigned int>((niter - nburn)/thin );
	}
	int niter;
	int nburn;
	int thin;
	MatCol Basemat; //grid_pts x p
	unsigned int iter_to_store;
	double trGwishSampler;
	friend std::ostream & operator<<(std::ostream &str, FLMParameters & pm){
		str<<"niter = "<<pm.niter<<std::endl;
		str<<"nburn = "<<pm.nburn<<std::endl;
		str<<"thin  = "<<pm.thin<<std::endl;
		str<<"Basemat: "<<std::endl<<pm.Basemat<<std::endl;
		str<<"iter_to_store  = "<<pm.iter_to_store<<std::endl;
		return str;
	} 
};

class InitFLM : public FLMsamplerTraits{
	public:
	InitFLM()=default;
	InitFLM(unsigned int const & _n, unsigned int const & _p):
			Beta0(MatCol::Zero(_p,_n)), mu0(VecCol::Zero(_p)), tau_eps0(1.0), tauK0(VecCol::Ones(_p)), K0(MatCol::Identity(_p,_p)), G(_p)
			{
				G.set_empty_graph();
			}
	InitFLM(unsigned int const & _n, unsigned int const & _p, GraphType<unsigned int> const & _G):
			Beta0(MatCol::Zero(_p,_n)), mu0(VecCol::Zero(_p)), tau_eps0(1.0), tauK0(VecCol::Ones(_p)), K0(MatCol::Identity(_p,_p)), G(_G) {}				
	void set_init(MatCol const & _Beta0, VecCol const & _mu0, double const & _tau_eps0, VecCol const & _tauK0){
		Beta0 = _Beta0;
		mu0 = _mu0;
		tau_eps0 = _tau_eps0;
		tauK0 = _tauK0; 
		G.set_empty_graph();
	}
	void set_init(MatCol const & _Beta0, VecCol const & _mu0, double const & _tau_eps0, MatCol const & _K0, GraphType<unsigned int> const & _G){
		Beta0 = _Beta0;
		mu0 = _mu0;
		tau_eps0 = _tau_eps0;
		K0 = _K0; 
		G  = _G; 
	}			
	MatCol Beta0; //p x n
	VecCol mu0; // p
	double tau_eps0; //scalar
	VecCol tauK0; // p 
	MatCol K0; // p x p
	GraphType<unsigned int> G;
};

// -----------------------------------------------------------------------------------------------------------------------------------------------

#endif