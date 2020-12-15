#ifndef __FLMSAMPLER_HPP__
#define __FLMSAMPLER_HPP__

#include "FLMSamplerOptions.h"

enum class GraphForm{
	Diagonal, Fix
};

//Functional Linear Model sampler.
//It is not a graphical model, the precision matrix on the regression coefficients is forced to be diagonal
template< GraphForm Graph = GraphForm::Diagonal>
class FLMsampler : public FLMsamplerTraits
{
	using RetType = std::conditional_t< Graph == GraphForm::Diagonal, 
										std::tuple<RetBeta, RetMu, RetTauK, RetTaueps> , 
										std::tuple<RetBeta, RetMu, RetK, RetTaueps>    > ;
	public:
	FLMsampler( MatCol const & _data, FLMParameters const & _params, FLMHyperparameters const & _hy_params, 
			    InitFLM const & _init, unsigned int _seed = 0):
			    data(_data), params(_params), hy_params(_hy_params) ,init(_init),
				p(_init.Beta0.rows()), n(_init.Beta0.cols()), grid_pts(_params.Basemat.rows()), seed(_seed)
	{
	 	this->check();
	 	if(seed==0){
	 	  std::random_device rd;
	 	  seed=rd();
	 	}	
	} 

	RetType run();

	private:
	void check() const;
	MatCol data; //grid_pts x n
	FLMParameters params;
	FLMHyperparameters hy_params;
	InitFLM init;
	unsigned int p;
	unsigned int n;
	unsigned int grid_pts;
	unsigned int seed;
};

template< GraphForm Graph >
void FLMsampler<Graph>::check() const{
	if(data.rows() != grid_pts) //check for grid_pts
		throw std::runtime_error("Error, incoerent number of grid points");
	if(data.cols() != n) //check for n
		throw std::runtime_error("Error, incoerent number of data");
	if( params.Basemat.cols() != p || p != init.mu0.size() ) //check for p
			throw std::runtime_error("Error, incoerent number of basis");
}

template< GraphForm Graph >
typename FLMsampler<Graph>::RetType FLMsampler<Graph>::run()
{	
	static_assert(Graph == GraphForm::Diagonal || Graph == GraphForm::Fix,
			      "Error, this sampler is not a graphical model, the graph has to be given. The possibilities for Graph parameter are GraphForm::Diagonal or GraphForm::Fix");
	std::cout<<"FLM sampler started"<<std::endl;
	// Declare all parameters (makes use of C++17 structured bindings)
	const unsigned int & r = grid_pts;
	const double&  a_tau_eps = this->hy_params.a_tau_eps;
	const double&  b_tau_eps = this->hy_params.b_tau_eps;
	const double&  sigma_mu  = this->hy_params.sigma_mu;
	const double&  a_tauK    = this->hy_params.a_tauK; 
	const double&  b_tauK    = this->hy_params.b_tauK; 
	const double&  bK    	 = this->hy_params.bK; 
	const MatCol&  DK    	 = this->hy_params.DK; 
	const auto &[niter, nburn, thin, Basemat, iter_to_store, threshold] = this->params;
	MatCol Beta = init.Beta0; //p x n
	VecCol mu = init.mu0; // p
	double tau_eps = init.tau_eps0; //scalar
	VecCol tauK = init.tauK0; 
	MatCol K = init.K0;
	const GraphType<unsigned int> &G = init.G; 
	//Random engine and distributions
	sample::GSL_RNG engine(seed);
	sample::rnorm rnorm;
	sample::rmvnorm rmv; //Covariance parametrization
	sample::rgamma  rgamma;
	//Define all those quantities that can be compute once
	const MatRow tbase_base = Basemat.transpose()*Basemat; // p x p
	const MatCol tbase_data = Basemat.transpose()*data;	 //  p x n
	const double Sdata(data.cwiseProduct(data).sum()); // Sum_i(<yi, yi>) sum of the inner products of each data 
	const double Sdata_btaueps(Sdata+b_tau_eps);
	const double a_tau_eps_post = (n*r + a_tau_eps)*0.5;	
	const double a_tauK_post = (n + a_tauK)*0.5;	
	const MatRow Irow(MatRow::Identity(p,p));
	const VecCol one_over_sigma_mu_vec(VecCol::Constant(p,1/sigma_mu));
	const MatRow one_over_sigma_mu_mat((1/sigma_mu)*Irow);
	//Structure for return
	RetBeta	  SaveBeta;
	RetMu	  SaveMu;	 
	RetTauK	  SaveTauK;
	RetK 	  SaveK; 	 
	RetTaueps SaveTaueps;

	SaveBeta.reserve(iter_to_store);
	SaveMu.reserve(iter_to_store);
	if constexpr(Graph == GraphForm::Diagonal){
		SaveTauK.reserve(iter_to_store);
	}
	else{
		SaveK.reserve(iter_to_store);
	}
	
	SaveTaueps.reserve(iter_to_store);

	//Setup for progress bar, need to specify the total number of iterations
	pBar bar(niter);
	 
	//Start MCMC loop
	for(int iter = 0; iter < niter; iter++){
		//Show progress bar
		bar.update(1);
		bar.print(); 

		if constexpr(Graph == GraphForm::Diagonal){
			//mu
			VecCol S_beta(Beta.rowwise().sum());
			VecCol A(one_over_sigma_mu_vec + n*tauK);
					//std::cout<<"A:"<<std::endl<<A<<std::endl;
					//std::cout<<"S_beta:"<<std::endl<<S_beta<<std::endl;
			for(unsigned int i = 0; i < p; ++i){ //paralellizzabile ma secondo me non ne vale la pena
						//std::cout<<"(1/A(i))*tauK(i)*S_beta(i) = "<<(1/A(i))*tauK(i)*S_beta(i)<<std::endl;
				mu(i) = rnorm(engine, (1/A(i))*tauK(i)*S_beta(i), std::sqrt(1/A(i)));
			}
			//Beta
			CholTypeRow chol_invBn(tau_eps*tbase_base + MatRow (tauK.asDiagonal())); 
			MatRow Bn(chol_invBn.solve(Irow));
			VecCol Kmu = tauK.cwiseProduct(mu); 
					//std::cout<<"Bn:"<<std::endl<<Bn<<std::endl;
					//std::cout<<"Kmu:"<<std::endl<<Kmu<<std::endl;
			//Quantities needed down the road
			VecCol U(VecCol::Zero(p)); 
			double b_tau_eps_post(Sdata_btaueps);
			#pragma omp parallel for shared(Beta, U), reduction(+ : b_tau_eps_post)
			for(unsigned int i = 0; i < n; ++i){
				VecCol bn_i = Bn*(tau_eps*tbase_data.col(i) + Kmu); 
				VecCol beta_i = rmv(engine, bn_i, Bn); //Since it has to be used in some calculations, save it so that it won't be necessary to find it later
				#pragma omp critical 
				{
					Beta.col(i) = beta_i;
					U += (beta_i - mu).cwiseProduct(beta_i - mu);
				}
				b_tau_eps_post += beta_i.dot(tbase_base*beta_i) - 2*beta_i.dot(tbase_data.col(i));  
			}
					//std::cout<<"U:"<<std::endl<<U<<std::endl;
			//Precision tauK
			for(unsigned int i = 0; i < p; ++i){ //paralellizzabile ma secondo me non ne vale la pena
				tauK(i) = rgamma(engine, a_tauK_post, 2/(U(i) + b_tauK) );
			}
					//std::cout<<"tauK:"<<std::endl<<tauK<<std::endl;
			//Precision tau_eps
			b_tau_eps_post /= 2.0;
			tau_eps = rgamma(engine, a_tau_eps_post, 1/b_tau_eps_post);
					//std::cout<<"tau_eps:"<<std::endl<<tau_eps<<std::endl;
			//Save
			if(iter >= nburn){
				if((iter - nburn)%thin == 0){
					SaveBeta.emplace_back(Beta);
					SaveMu.emplace_back(mu);
					SaveTaueps.emplace_back(tau_eps);
					SaveTauK.emplace_back(tauK);
				}
			}
		}
		else{
			//mu
			VecCol S_beta(Beta.rowwise().sum());
			CholTypeRow chol_invA(one_over_sigma_mu_mat + n*K); 
			MatRow A(chol_invA.solve(Irow));
			VecCol a(A*(K*S_beta));
			mu = rmv(engine, a, A);
			//Beta
			CholTypeRow chol_invBn(tau_eps*tbase_base + K); 
			MatRow Bn(chol_invBn.solve(Irow));
			VecCol Kmu(K*mu);
			//Quantities needed down the road
			MatCol U(MatCol::Zero(p,p)); //Has to be ColMajor
			double b_tau_eps_post(Sdata_btaueps);
			#pragma omp parallel for shared(Beta, U), reduction(+ : b_tau_eps_post)
			for(unsigned int i = 0; i < n; ++i){
				VecCol bn_i = Bn*(tau_eps*tbase_data.col(i) + Kmu); 
				VecCol beta_i = rmv(engine, bn_i, Bn); //Since it has to be used in some calculations, save it so that it won't be necessary to find it later
				#pragma omp critical 
				{
					Beta.col(i) = beta_i;
					U += (beta_i - mu)*(beta_i - mu).transpose();
				}
				b_tau_eps_post += beta_i.dot(tbase_base*beta_i) - 2*beta_i.dot(tbase_data.col(i));  
			}
			//Precision K
			K = std::move( utils::rgwish(G,bK+n,DK+U,threshold) );
			//Precision tau
			b_tau_eps_post /= 2.0;
			tau_eps = rgamma(engine, a_tau_eps_post, 1/b_tau_eps_post);
			//Save
			if(iter >= nburn){
				if((iter - nburn)%thin == 0){
					SaveBeta.emplace_back(Beta);
					SaveMu.emplace_back(mu);
					SaveTaueps.emplace_back(tau_eps);
					SaveK.emplace_back(K);
				}
			}
		}
	}
	std::cout<<std::endl<<"FLM sampler has finished"<<std::endl;
	if constexpr(Graph == GraphForm::Diagonal){
		return std::make_tuple(SaveBeta, SaveMu, SaveTauK, SaveTaueps);
	}
	else{
		return std::make_tuple(SaveBeta, SaveMu, SaveK, SaveTaueps);
	}
	
}








#endif