#ifndef __GGMSAMPLER_HPP__
#define __GGMSAMPLER_HPP__

#include "SamplerOptions.h"

//Old version of this function let the user choose the return container for the graphs. Template possibilities for storing graphs are:
/*< class RetGraph = std::unordered_map< std::vector<bool>, int>*/
// std::unordered_map< std::vector<bool>, int> / std::map< std::vector<bool>, int> / std::vector< std::pair< std::vector<bool>, int> >


template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class GGMsampler : public SamplerTraits
{
	public:
	using Graph 	    = GraphStructure<T>;
	using CompleteType  = typename GGMTraits<GraphStructure, T>::CompleteType;
	using PrecisionType = typename GGMTraits<GraphStructure, T>::PrecisionType;
	using GGMType 	    = std::unique_ptr<GGM<GraphStructure, T>>;
	//using RetType 	    = std::tuple<RetK, RetGraph, int, int>; //Used only if sampled valued were saved in memory
	
	GGMsampler( MatCol const & _data, unsigned int const & _n, Parameters const & _params, Hyperparameters const & _hy_params, 
			    Init<GraphStructure, T> const & _init, GGMType & _GGM_method, std::string const & _file_name = "GGMresult", unsigned int _seed = 0, bool _print_bp = true):
			    data(_data), params(_params), hy_params(_hy_params), ptr_GGM_method(std::move(_GGM_method)) ,init(_init),
				p(_data.rows()), n(_n), grid_pts(_params.Basemat.rows()), engine(_seed), print_bp(_print_bp), file_name(_file_name)
	{
	 	this->check();
	 	file_name += ".h5";
	} 
	int run();

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
	sample::GSL_RNG engine;
	int total_accepted{0};
	int visited{0};
	bool print_bp;
	std::string file_name;

};

template<template <typename> class GraphStructure, typename T >
void GGMsampler<GraphStructure, T >::check(){
	if( hy_params.p_addrm <= 0 || hy_params.p_addrm >= 1)
		hy_params.p_addrm = 0.5;
	if( ptr_GGM_method->get_inv_scale().cols() != p || data.rows() != p ||
		p != init.K0.cols() || init.K0.cols() != init.K0.rows() || 
		p != init.G0.get_complete_size()) //check for p
			throw std::runtime_error("Error, incoerent number of basis");
}



template<template <typename> class GraphStructure, typename T  >
int GGMsampler<GraphStructure, T >::run()
{
	//Typedefs
	GGM<GraphStructure, T> & GGM_method= *ptr_GGM_method;

	// Declare all parameters (makes use of C++17 structured bindings)
	const unsigned int & r = grid_pts;
	const double&  p_addrm   = this->hy_params.p_addrm; 
	const auto &[niter, nburn, thin, thinG, MCiterPrior, MCiterPost, Basemat, ptr_groups, iter_to_store, iter_to_storeG, threshold] = this->params;
	MatRow K = init.K0; 
	Graph  G = init.G0;
	GGM_method.init_precision(G,K); 
	total_accepted = 0;
	visited = 0;
	const unsigned int prec_elem = 0.5*p*(p+1); //Number of elements in the upper part of precision matrix (diagonal inclused). It is what is saved of the Precision matrix
	unsigned int n_graph_elem{0};
	if constexpr( internal_type_traits::isBlockGraph<GraphStructure, T>::value ){
		n_graph_elem = G.get_possible_block_links();
	}
	else{
		n_graph_elem = G.get_possible_links();
	}
	unsigned int it_saved{0};

	//Random engine and distributions
	sample::rmvnorm rmv; //Covariance parametrization
	sample::rgamma  rGamma;

	//Define all those quantities that can be compute once
	const MatRow Irow(MatRow::Identity(p,p));
						
	//Open file
	HDF5conversion::FileType file;
	file = H5Fcreate(file_name.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if(file < 0)
		throw std::runtime_error("Cannot create the file. The most probable reason is that the execution was stopped before closing a file having the same name of the one that was asked to be generated. Delete the old file or change the name of the new one");
	int one_dim_rank = 1;//for 1-dim datasets. All other quantities
	
	//Print info file
	HDF5conversion::DataspaceType dataspace_info;
	HDF5conversion::ScalarType Dim_info = 4;
	dataspace_info = H5Screate_simple(one_dim_rank, &Dim_info, NULL);
	HDF5conversion::DatasetType  dataset_info;
	dataset_info  = H5Dcreate(file,"/Info", H5T_NATIVE_UINT, dataspace_info, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_info < 0)
		throw std::runtime_error("Error, can not create dataset for Info ");
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
	HDF5conversion::WriteString(dataset_version, "GGMsampler");
	//Create dataspaces
	HDF5conversion::DataspaceType dataspace_Prec, dataspace_Graph;
	HDF5conversion::ScalarType Dim_K_ds = prec_elem*iter_to_store;
	HDF5conversion::ScalarType Dim_G_ds = n_graph_elem*iter_to_store;

	dataspace_Prec  = H5Screate_simple(one_dim_rank, &Dim_K_ds, NULL);
	dataspace_Graph = H5Screate_simple(one_dim_rank, &Dim_G_ds, NULL);

	//Create dataset
	HDF5conversion::DatasetType  dataset_Prec, dataset_Graph;
	dataset_Prec = H5Dcreate(file,"/Precision", H5T_NATIVE_DOUBLE, dataspace_Prec, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_Prec < 0)
		throw std::runtime_error("Error, can not create dataset for Precision");
	dataset_Graph = H5Dcreate(file,"/Graphs", H5T_NATIVE_UINT, dataspace_Graph, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if(dataset_Graph < 0)
		throw std::runtime_error("Error, can not create dataset for Graphs");
	

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
		std::tie(K, accepted_mv) = GGM_method(data, n, G, p_addrm, engine); //G is modified inside the function.
		total_accepted += accepted_mv;

		//Check for user interruption
		if(iter%1000 == 0){
			Rcpp::checkUserInterrupt(); //files are not closed 
		}

		//Save
		if(iter >= nburn){
			if((iter - nburn)%thinG == 0 && it_saved < iter_to_store ){ 

			//Save on file
			std::vector<unsigned int> adj_file;
			if constexpr( ! std::is_same_v<T, unsigned int>){
				std::vector<T> adj_nouint(G.get_adj_list());
				adj_file.resize(adj_nouint.size());
				std::transform(adj_nouint.begin(), adj_nouint.end(), adj_file.begin(), [](T x) { return (unsigned int)x;});
			}
			else{
				adj_file = G.get_adj_list();
			}
			VecCol UpperK{utils::get_upper_part(K)};
			HDF5conversion::AddVector(dataset_Prec, UpperK, it_saved);	
			HDF5conversion::AddUintVector(dataset_Graph, adj_file, it_saved);
			it_saved++;
			}
		}
	}
	H5Dclose(dataset_Graph);
	H5Dclose(dataset_Prec);
	H5Dclose(dataset_info);
	H5Dclose(dataset_version);
	H5Fclose(file);


	return total_accepted;
}


#endif
