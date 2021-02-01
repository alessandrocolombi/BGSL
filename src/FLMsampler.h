#ifndef __FLMSAMPLER_HPP__
#define __FLMSAMPLER_HPP__

#include "FLMSamplerOptions.h"

enum class GraphForm{
	Diagonal, Fix
};

//Functional Linear Model sampler.
//It is not a graphical model, the graph has to be fixed, diagonal or generic forms are both allowed
template< GraphForm Graph = GraphForm::Diagonal>
class FLMsampler : public FLMsamplerTraits
{
	//For saving in memory, not on file
	//using RetType = std::conditional_t< Graph == GraphForm::Diagonal, 
										//std::tuple<RetBeta, RetMu, RetTauK, RetTaueps> , 
										//std::tuple<RetBeta, RetMu, RetK, RetTaueps>    > ;
	public:
	FLMsampler( MatCol const & _data, FLMParameters const & _params, FLMHyperparameters const & _hy_params, 
			    InitFLM const & _init,  std::string const & _file_name = "FLMresult", unsigned int _seed = 0, bool _print_pb = true):
			    data(_data), params(_params), hy_params(_hy_params) ,init(_init),
				p(_init.Beta0.rows()), n(_init.Beta0.cols()), grid_pts(_params.Basemat.rows()), engine(_seed), print_pb(_print_pb), file_name(_file_name)
	{
	 	this->check();
	 	file_name += ".h5";
	} 

	int run();

	private:
	void check() const;
	MatCol data; //grid_pts x n
	FLMParameters params;
	FLMHyperparameters hy_params;
	InitFLM init;
	unsigned int p;
	unsigned int n;
	unsigned int grid_pts;
	sample::GSL_RNG engine;
	bool print_pb;
	std::string file_name;
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
int FLMsampler<Graph>::run() //typename FLMsampler<Graph>::RetType 
{	
	static_assert(Graph == GraphForm::Diagonal || Graph == GraphForm::Fix,
			      "Error, this sampler is not a graphical model, the graph has to be given. The possibilities for Graph parameter are GraphForm::Diagonal or GraphForm::Fix");
	
	// Declare all parameters (makes use of C++17 structured bindings)
	const unsigned int & r = grid_pts;
	const unsigned int n_elem_mat = 0.5*p*(p+1); //Number of elements in the upper part of precision matrix (diagonal inclused). It is what is saved if K is a matrix
	const double&  a_tau_eps = this->hy_params.a_tau_eps;
	const double&  b_tau_eps = this->hy_params.b_tau_eps;
	const double&  sigma_mu  = this->hy_params.sigma_mu;
	const double&  a_tauK    = this->hy_params.a_tauK; 
	const double&  b_tauK    = this->hy_params.b_tauK; 
	const double&  bK    	 = this->hy_params.bK; 
	const MatCol&  DK    	 = this->hy_params.DK; 
	const auto &[niter, nburn, thin, Basemat, iter_to_store, threshold] = this->params;
	unsigned int prec_elem{0}; //What is the number of elemets in the precision matrix to be saved? It depends on the template parameter. 
	std::string sampler_version = "FLMsampler_";
	if constexpr(Graph == GraphForm::Diagonal){ 
		prec_elem = p;
		sampler_version += "diagonal";
	}
	else{
		prec_elem = n_elem_mat;
		sampler_version += "fixed";
	}
	//Initial values
	MatCol Beta = init.Beta0; //p x n
	VecCol mu = init.mu0; // p
	double tau_eps = init.tau_eps0; //scalar
	VecCol tauK = init.tauK0; 
	MatRow K = init.K0;
	const GraphType<unsigned int> &G = init.G; 

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
	
	//Open file
	HDF5conversion::FileType file;
	file = H5Fcreate(file_name.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); 
	if(file < 0)
		throw std::runtime_error("Cannot create the file. The most probable reason is that the execution was stopped before closing a file having the same name of the one that was asked to be generated. Delete the old file or change the name of the new one");

	int bi_dim_rank = 2; //for 2-dim datasets. Beta are matrices
	int one_dim_rank = 1;//for 1-dim datasets. All other quantities
	//Print file info
	HDF5conversion::DataspaceType dataspace_info;
	HDF5conversion::ScalarType Dim_info = 4;
	dataspace_info = H5Screate_simple(one_dim_rank, &Dim_info, NULL);
	HDF5conversion::DatasetType  dataset_info;
	dataset_info  = H5Dcreate(file,"/Info", H5T_NATIVE_UINT, dataspace_info, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_info < 0)
		throw std::runtime_error("Error can not create dataset for Info");
	{
		std::vector< unsigned int > info{p,n,iter_to_store,iter_to_store};
		unsigned int * buffer_info = info.data();		
		HDF5conversion::StatusType status = H5Dwrite(dataset_info, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_info);
	}
	//Print what sampler has been used
	HDF5conversion::DatasetType  dataset_version;
	HDF5conversion::DataspaceType dataspace_version = H5Screate(H5S_NULL);
	dataset_version = H5Dcreate(file, "/Sampler", H5T_STD_I32LE, dataspace_version, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_version < 0)
		throw std::runtime_error("Error, can not create dataset for Sampler");
	HDF5conversion::WriteString(dataset_version, sampler_version);
	//Create dataspaces
	HDF5conversion::DataspaceType dataspace_Beta, dataspace_Mu, dataspace_Prec, dataspace_TauEps;
	HDF5conversion::ScalarType Dim_beta_ds[2] = {p, n*iter_to_store}; 
	HDF5conversion::ScalarType Dim_mu_ds = p*iter_to_store;
	HDF5conversion::ScalarType Dim_K_ds = prec_elem*iter_to_store;
	HDF5conversion::ScalarType Dim_taueps_ds = iter_to_store;

	dataspace_Beta = H5Screate_simple(bi_dim_rank,   Dim_beta_ds,     NULL);
	dataspace_Mu = H5Screate_simple(one_dim_rank, &Dim_mu_ds,         NULL);
	dataspace_TauEps = H5Screate_simple(one_dim_rank, &Dim_taueps_ds, NULL);
	dataspace_Prec = H5Screate_simple(one_dim_rank, &Dim_K_ds, NULL);

	//Create dataset
	HDF5conversion::DatasetType  dataset_Beta, dataset_Mu, dataset_Prec, dataset_TauEps;
	dataset_Beta  = H5Dcreate(file,"/Beta", H5T_NATIVE_DOUBLE, dataspace_Beta, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_Beta < 0)
		throw std::runtime_error("Error can not create dataset for Beta ");
	dataset_Mu = H5Dcreate(file,"/Mu", H5T_NATIVE_DOUBLE, dataspace_Mu, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_Mu < 0)
		throw std::runtime_error("Error can not create dataset for Mu");
	//SURE_ASSERT(dataset_Mu>=0,"Cannot create dataset for Mu");
	dataset_TauEps  = H5Dcreate(file,"/TauEps", H5T_NATIVE_DOUBLE, dataspace_TauEps, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_TauEps < 0)
		throw std::runtime_error("Error can not create dataset for TauEps");
	dataset_Prec = H5Dcreate(file,"/Precision", H5T_NATIVE_DOUBLE, dataspace_Prec, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataspace_Prec < 0)
		throw std::runtime_error("Error can not create dataset for Precision");



	//Setup for progress bar, need to specify the total number of iterations
	pBar bar(niter);
	 
	//Start MCMC loop
	unsigned int it_saved{0};
	for(int iter = 0; iter < niter; iter++){
		//Show progress bar
		bar.update(1);
		if(print_pb){
		 bar.print(); 
		}

		if constexpr(Graph == GraphForm::Diagonal){
			//mu
			VecCol S_beta(Beta.rowwise().sum());
			VecCol A(one_over_sigma_mu_vec + n*tauK);

			for(unsigned int i = 0; i < p; ++i){ //For the moment, it is not worth to be parallelized
				mu(i) = rnorm(engine, (1/A(i))*tauK(i)*S_beta(i), std::sqrt(1/A(i)));
			}

			
			//Beta
			CholTypeRow chol_invBn(tau_eps*tbase_base + MatRow (tauK.asDiagonal())); 
			MatRow Bn(chol_invBn.solve(Irow));
			VecCol Kmu = tauK.cwiseProduct(mu); 

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
			//Precision tauK
			for(unsigned int j = 0; j < p; ++j){ //For the moment, it is not worth to be parallelized
				tauK(j) = rgamma(engine, a_tauK_post, 2/(U(j) + b_tauK) );
			}

			//Precision tau_eps
			b_tau_eps_post /= 2.0;
			tau_eps = rgamma(engine, a_tau_eps_post, 1/b_tau_eps_post);



			//Save
			if(iter >= nburn){

				if((iter - nburn)%thin == 0 && it_saved < iter_to_store){

					HDF5conversion::AddMatrix(dataset_Beta,   Beta,    it_saved);	
					HDF5conversion::AddVector(dataset_Mu,     mu,      it_saved);	
					HDF5conversion::AddVector(dataset_Prec,   tauK,    it_saved);	
					HDF5conversion::AddScalar(dataset_TauEps, tau_eps, it_saved);
					it_saved++;

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
			MatCol D_plus_U(DK+U);
			K = std::move( utils::rgwish(G,bK+n,D_plus_U,threshold,engine) );
			
			//Precision tau
			b_tau_eps_post /= 2.0;
			tau_eps = rgamma(engine, a_tau_eps_post, 1/b_tau_eps_post);
			


			//Save
			if(iter >= nburn){
				if((iter - nburn)%thin == 0 && it_saved < iter_to_store){

					VecCol UpperK{utils::get_upper_part(K)};
					HDF5conversion::AddMatrix(dataset_Beta,   Beta,    it_saved);	
					HDF5conversion::AddVector(dataset_Mu,     mu,      it_saved);	
					HDF5conversion::AddVector(dataset_Prec,   UpperK,  it_saved);	
					HDF5conversion::AddScalar(dataset_TauEps, tau_eps, it_saved);
					it_saved++;

				}
			}
		}

		//Check for User interruption
		try{
		  Rcpp::checkUserInterrupt();
		}
		catch(Rcpp::internal::InterruptedException e){
			Rcpp::Rcout<<"Execution stopped during iter "<<iter<<"/"<<niter<<std::endl;
			H5Dclose(dataset_Beta);
			H5Dclose(dataset_TauEps);
			H5Dclose(dataset_Prec);
			H5Dclose(dataset_Mu);
			H5Dclose(dataset_info);
			H5Dclose(dataset_version);
			H5Fclose(file);
			return -1;
		}
	}

	H5Dclose(dataset_Beta);
	H5Dclose(dataset_TauEps);
	H5Dclose(dataset_Prec);
	H5Dclose(dataset_Mu);
	H5Dclose(dataset_info);
	H5Dclose(dataset_version);
	H5Fclose(file);
	return 0;
}








#endif