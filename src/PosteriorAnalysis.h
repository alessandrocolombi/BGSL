#ifndef __POSTERIORANALYSIS_HPP__	
#define __POSTERIORANALYSIS_HPP__	

#include "SamplerOptions.h"
#include "FLMSamplerOptions.h"
#include <gsl/gsl_statistics_double.h> //used to compute quantiles

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
	using RetTauK	 = FLMsamplerTraits::RetTauK;

	/*Compute posterior probability of inclusion of each possible link*/
	template< class RetGraph, typename T = bool  >
	MatRow Compute_plinks(RetGraph const & SampledGraphs, const unsigned int & iter_saved, GroupsPtr const & groups = nullptr)
	{
		static_assert(  std::is_same_v< RetGraph, std::unordered_map<std::vector<T>, int> > || std::is_same_v< RetGraph, std::map<std::vector<T>, int> > || 
						std::is_same_v< RetGraph, std::vector< std::pair< std::vector<T>, int> > >,
						"Error, incompatible container inserted");
		
		//Sum two adjacency lists
		auto sum_adj = [](std::pair< std::vector<T>, int> const & a1, std::pair< std::vector<T>, int> const & a2){
			std::vector<int> res(a1.first.size());
			const std::vector<T> &v1(a1.first);
			const std::vector<T> &v2(a2.first);
			const int & freq1 = a1.second;
			const int & freq2 = a2.second;
			std::transform(v1.cbegin(), v1.cend(), v2.cbegin(), res.begin(), 
				[&freq1, &freq2](T e1, T e2){
					return (int)e1*freq1 + (int)e2*freq2;
				}  
			);
			return res;
		}; 
		//Weighted update of adj_list, weights are given by the frequence of visits
		auto sum = [](std::vector<double> const & plinks_vet, std::pair< std::vector<T>, int> const & a2){
			std::vector<double> res(plinks_vet.size());
			const std::vector<T> &v2(a2.first);
			const int & freq = a2.second;
			std::transform(plinks_vet.cbegin(), plinks_vet.cend(), v2.cbegin(), res.begin(), 
				[&freq](int e1, T e2){
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
			plinks.diagonal().array() = 1; //Set the diagoal equal to one
			return plinks;
		}
		else{ //Assume it is a Block Graph
			BlockGraph<double> G(plinks_adj, groups);
			MatRow plinks( G.get_graph());
			plinks /= iter_saved;
			std::vector<unsigned int> singleton(G.get_pos_singleton());
			if(singleton.size() != 0) //Set the diagoal equal to one if there is a singleton 
				std::for_each(singleton.cbegin(), singleton.cend(),[&plinks](unsigned int const & pos){plinks(pos, pos)=1.0;});
			return plinks;
		}
	}


	//------------------------------------------------------------------------------------------------------------------------------------------------------
	//Reading values from file


	//This function is thought for Beta matrices. file_name has to containg also the extension (name.h5). The file has to contain a dataset called /Beta.
	//It then returns a (p x n) matrix containing the mean values of every Beta
	MatCol Matrix_PointwiseEstimate( std::string const & file_name, const int& saved_iter, const unsigned int & p, const unsigned int & n )
	{
		
		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_Beta_rd;
		//Open file
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		dataset_Beta_rd  = H5Dopen(file, "/Beta", H5P_DEFAULT);
		if(dataset_Beta_rd < 0)
			throw std::runtime_error("Error, can not open dataset for Beta ");
		//Read beta
		std::vector< MatCol > vett_Beta(saved_iter);
		for(int i = 0; i < saved_iter; ++i){
			vett_Beta[i] = HDF5conversion::ReadMatrix(dataset_Beta_rd, p, n, i);	
		}

		MatCol MeanBeta(MatCol::Zero(p, n));
		for(int i = 0; i < saved_iter; ++i){
			MeanBeta += vett_Beta[i];
		}
		MeanBeta /= saved_iter;	
		H5Dclose(dataset_Beta_rd);
		H5Fclose(file);
		return MeanBeta;
	}
	//This function is thought for Mu and Precision vector. file_name has to containg also the extension (name.h5). The file has to contain a dataset called /Mu or /Precision.
	//Only possibilities for vett_type are indeed Mu or Precision.
	//It then returns a (length)-dimensional vector containing the mean values of every element
	VecCol Vector_PointwiseEstimate( std::string const & file_name, const int& saved_iter, const unsigned int & length, std::string const & vett_type )
	{
		if(vett_type != "Mu" && vett_type != "Precision")
			throw std::runtime_error("Error in Vector_PointwiseEstimate(). vett_type can only be Mu or Precision");

		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_rd;
		//Open file
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		if(vett_type == "Mu"){
			dataset_rd  = H5Dopen(file, "/Mu", H5P_DEFAULT);
			if(dataset_rd < 0)
				throw std::runtime_error("Error, can not open dataset for Mu");
		}
		else
		{
			dataset_rd = H5Dopen(file, "/Precision", H5P_DEFAULT);
			if(dataset_rd < 0)
				throw std::runtime_error("Error, can not open dataset for Precision");
		}
		
		//Read 
		std::vector< VecCol > vect(saved_iter);
		for(int i = 0; i < saved_iter; ++i){
			vect[i] = HDF5conversion::ReadVector(dataset_rd, length, i);	
		}

		//Compute mean
		VecCol Mean(VecCol::Zero(length));
		for(int i = 0; i < saved_iter; ++i){
			Mean += vect[i];
		}
		Mean /= saved_iter;	

		H5Dclose(dataset_rd);
		H5Fclose(file);
		return Mean;
	}
	//This function is thought for TauEps variables. file_name has to containg also the extension (name.h5). The file has to contain a dataset called /TauEps.
	//It then returns a double with the mean values of the sampled values
	double Scalar_PointwiseEstimate( std::string const & file_name, const int& saved_iter )
	{
		
		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_rd;
		//Open file
		file = H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/TauEps", H5P_DEFAULT);
		if(dataset_rd < 0)
			throw std::runtime_error("Error, can not open dataset for TauEps ");
		//Read 
		std::vector< double > vett(saved_iter);
		for(int i = 0; i < saved_iter; ++i){
			vett[i] = HDF5conversion::ReadScalar(dataset_rd, i);	
		}
		//Compute mean
		double mean = std::accumulate(vett.cbegin(), vett.cend(), 0.0);
		mean /= saved_iter;
		H5Dclose(dataset_rd);
		H5Fclose(file);	
		return mean;
	}

	//Computes quantiles for evey Beta. file_name has the same requirement of Matrix_PointwiseEstimate() 
	std::tuple<MatCol,MatCol> Matrix_ComputeQuantiles(	std::string const & file_name, unsigned int const & stored_iter, unsigned int const & p, unsigned int const & n, 
														double const & alpha_lower = 0.05, double const & alpha_upper = 0.95	)
	{
		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_rd;
		//Open file
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/Beta", H5P_DEFAULT);
		if(dataset_rd < 0)
			throw std::runtime_error("Error, can not opem dataset for Beta ");

		//Read chains and compute quantiles
		MatCol LowerBound(MatCol::Zero(p,n));
		MatCol UpperBound(MatCol::Zero(p,n));
		for(int i = 0; i < p; ++i){
			for(int j = 0; j < n; ++j){
				std::vector<double> chain = HDF5conversion::GetChain_from_Matrix(dataset_rd, i, j, stored_iter, n);
				std::sort(chain.begin(), chain.end());
				LowerBound(i,j) = gsl_stats_quantile_from_sorted_data(&chain.data()[0], 1, chain.size(), alpha_lower);
				UpperBound(i,j) = gsl_stats_quantile_from_sorted_data(&chain.data()[0], 1, chain.size(), alpha_upper);
			}
			std::cout<<"Finihed p = "<<i<<std::endl;
		}

		H5Dclose(dataset_rd);
		H5Fclose(file);
		return std::make_tuple(LowerBound, UpperBound);
	}
	//Computes quantiles for evey Mu or Precision element. file_name has the same requirement of Vector_PointwiseEstimate() 
	std::tuple<VecCol,VecCol> Vector_ComputeQuantiles(	std::string const & file_name, unsigned int const & stored_iter, unsigned int const & n_elem, std::string const & vett_type, 
														double const & alpha_lower = 0.05, double const & alpha_upper = 0.95	)
	{
		if(vett_type != "Mu" && vett_type != "Precision")
			throw std::runtime_error("Error in Vector_PointwiseEstimate(). vett_type can only be Mu or Precision");

		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_rd;
		//Open file
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		if(vett_type == "Mu"){
			dataset_rd  = H5Dopen(file, "/Mu", H5P_DEFAULT);
			if(dataset_rd < 0)
				throw std::runtime_error("Error, can not open dataset for Mu ");
		}
		else
		{
			dataset_rd  = H5Dopen(file, "/Precision", H5P_DEFAULT);
			if(dataset_rd < 0)
				throw std::runtime_error("Error, can not open dataset for Precision");
		}
		//Read chains and compute quantiles
		VecCol LowerBound(VecCol::Zero(n_elem));
		VecCol UpperBound(VecCol::Zero(n_elem));
		for(int i = 0; i < n_elem; ++i){
			std::vector<double> chain = HDF5conversion::GetChain_from_Vector(dataset_rd, i, stored_iter, n_elem);
			std::sort(chain.begin(), chain.end());
			LowerBound(i) = gsl_stats_quantile_from_sorted_data(&chain.data()[0], 1, chain.size(), alpha_lower);
			UpperBound(i) = gsl_stats_quantile_from_sorted_data(&chain.data()[0], 1, chain.size(), alpha_upper);
		}

		H5Dclose(dataset_rd);
		H5Fclose(file);
		return std::make_tuple(LowerBound, UpperBound);
	}
	//Computes quantiles for tau_eps. file_name has the same requirement of Scalar_PointwiseEstimate() 
	std::tuple<double,double> Scalar_ComputeQuantiles(	std::string const & file_name, unsigned int const & stored_iter, double const & alpha_lower = 0.05, double const & alpha_upper = 0.95	)
	{
		
		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_rd;
		//Open file
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/TauEps", H5P_DEFAULT);
		if(dataset_rd < 0)
			throw std::runtime_error("Error, can not open dataset for TauEps");
		//Read chains all the chain
		std::vector<double> chain(stored_iter);
		double * ptr_chain = chain.data();
		HDF5conversion::StatusType status;
		status = H5Dread(dataset_rd,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr_chain);

		//Compute Quantiles
		double LowerBound{0};
		double UpperBound{0};
		std::sort(chain.begin(), chain.end());
		LowerBound = gsl_stats_quantile_from_sorted_data(&chain.data()[0], 1, chain.size(), alpha_lower);
		UpperBound = gsl_stats_quantile_from_sorted_data(&chain.data()[0], 1, chain.size(), alpha_upper);
		//Close and return
		H5Dclose(dataset_rd);
		H5Fclose(file);
		return std::make_tuple(LowerBound, UpperBound);
	}


	std::tuple<MatRow, HDF5conversion::SampledGraphs, VecCol, int > //plinks, map with graphs frequence of visit, traceplot, visited graphs
	Summary_Graph(std::string const & file_name, unsigned int const & stored_iter, unsigned int const & p, GroupsPtr const & groups = nullptr)
	{
		unsigned int n_elem;
		if( groups == nullptr){ //Assume it is a complete Graph
			n_elem = 0.5*p*(p-1);
		}
		else{ //Assume it is a block Graph
			n_elem = 0.5*groups->get_n_groups()*(groups->get_n_groups() + 1) - groups->get_n_singleton();
		}

		HDF5conversion::SampledGraphs Glist;
		VecCol traceplot_size;
		int visited;
		//Open file
		HDF5conversion::FileType file;
		HDF5conversion::DatasetType dataset_rd;
		//Open file
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		if(file < 0)
			throw std::runtime_error("Cannot open the file in read-only mode. The most probable reason is that it was not closed correctly");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/Graphs", H5P_DEFAULT);
		if(dataset_rd < 0)
			throw std::runtime_error("Error, can not open the dataset for Graphs");

		std::tie(Glist, traceplot_size, visited) = HDF5conversion::GetGraphsChain(dataset_rd, n_elem, stored_iter);
		MatRow plinks = Compute_plinks<decltype(Glist), unsigned int>(Glist, stored_iter, groups);
		H5Dclose(dataset_rd);
		H5Fclose(file);
		return std::make_tuple(plinks, Glist, traceplot_size, visited);
	}

}

#endif