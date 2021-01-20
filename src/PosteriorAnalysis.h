#ifndef __POSTERIORANALYSIS_HPP__	
#define __POSTERIORANALYSIS_HPP__	

#include "SamplerOptions.h"
#include "FLMSamplerOptions.h"
#include <gsl/gsl_statistics_double.h>

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
	//using RetGraph   = SamplerTraits::RetGraph;
	//using RetType	 = SamplerTraits::RetType;

	/*
	//It works both for all types of RetGraphs (vector / map / unordered_map)
	template< class RetGraph = std::unordered_map< std::vector<bool>, int> >
	MatRow Compute_plinks(RetGraph const & SampledGraphs, const unsigned int & iter_saved, GroupsPtr const & groups = nullptr)
	{

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
	*/

	//It works both for all types of RetGraphs (vector / map / unordered_map)
	template< class RetGraph, typename T = bool  >
	MatRow Compute_plinks(RetGraph const & SampledGraphs, const unsigned int & iter_saved, GroupsPtr const & groups = nullptr)
	{
		static_assert(  std::is_same_v< RetGraph, std::unordered_map<std::vector<T>, int> > || std::is_same_v< RetGraph, std::map<std::vector<T>, int> > || 
						std::is_same_v< RetGraph, std::vector< std::pair< std::vector<T>, int> > >,
						"Error, incompatible container inserted");
		// Need to have the same size!
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
		}; //For weighted sum of adj_list 
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

	//Computes mean values for output of FGMsampler
	template< class RetGraph = std::unordered_map< std::vector<bool>, int> > //template parameters
	std::tuple<MatCol, VecCol, VecCol, MatRow, double>  //Return type
	PointwiseEstimate(	std::tuple<RetBeta, RetMu, RetK, RetGraph, RetTaueps> const & ret, 
						const int& iter_to_store, const int& iter_to_storeG, GroupsPtr const & groups = nullptr)
	{
		
		//Unpacking the tuple with reference binding (c++17)
		const auto&[SaveBeta, SaveMu, SaveK, SaveG, SaveTaueps] = ret;

		//SaveBeta is a vector of size iter_to_store  with (p x n)-dimensional matrices
		//SaveMu   is a vector of size iter_to_store  with (p)-dimensional eigen vectors
		//SaveK    is a vector of size iter_to_store with (0.5*p*(p+1))-dimensional eigen vectors
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
		//MatRow MeanK( MatRow::Zero(SaveK[0].rows(), SaveK[0].cols())  );
		VecCol MeanK( VecCol::Zero(SaveK[0].size())  );
		for(int i = 0; i < SaveK.size(); ++i){
			MeanK += SaveK[i];
		}
		MeanK /= iter_to_storeG;
		//plinks
		MatRow plinks(Compute_plinks(SaveG, iter_to_storeG, groups));
		return std::make_tuple(MeanBeta, MeanMu, MeanK, plinks, MeanTaueps);
	}

	//Computes mean values for output of FLMsampler. 
	std::tuple<MatCol, VecCol, VecCol, double>  //Return type
	PointwiseEstimate(	std::tuple<RetBeta, RetMu, RetTauK, RetTaueps> const & ret, const int& iter_to_store )
	{
		
		//Unpacking the tuple with reference binding (c++17)
		const auto&[SaveBeta, SaveMu, SaveTauK, SaveTaueps] = ret;

		//SaveBeta is a vector of size iter_to_store  with (p x n)-dimensional matrices
		//SaveMu   is a vector of size iter_to_store  with (p)-dimensional eigen vectors
		//SaveTauK   is a vector of size iter_to_store  with (p)-dimensional eigen vectors
		//SaveTaueps is a std::vector<double> of size iter_to_store    

		//Beta, Mu, TauK
		MatCol MeanBeta(MatCol::Zero(SaveBeta[0].rows(), SaveBeta[0].cols()));
		VecCol MeanMu  ( VecCol::Zero(SaveMu[0].size()) );
		VecCol MeanTauK( VecCol::Zero(SaveTauK[0].size()) );
		for(int i = 0; i < SaveBeta.size(); ++i){
			MeanBeta += SaveBeta[i];
			MeanMu 	 += SaveMu[i]; 
			MeanTauK += SaveTauK[i];
		}
		MeanBeta /= iter_to_store;	
		MeanMu /= iter_to_store;
		MeanTauK /= iter_to_store;
		//Tau_eps
		double MeanTaueps( std::accumulate(  SaveTaueps.begin(), SaveTaueps.end(), 0.0  ) ); 
		MeanTaueps /= iter_to_store;
		return std::make_tuple(MeanBeta, MeanMu, MeanTauK, MeanTaueps);
	}
	
	//Computes mean values for output of GGMsampler
	//Usage: need to be explicit when passing the templete parameter because often cannot be deduced
	//analysis::PointwiseEstimate<decltype(SampledG)>(std::tie(SampledK, SampledG),param.iter_to_storeG);
	template< class RetGraph = std::unordered_map< std::vector<bool>, int> > //template parameters
	std::tuple<VecCol, MatRow>  //Return type
	PointwiseEstimate(	std::tuple<RetK, RetGraph> const & ret, 
						const int& iter_to_store, GroupsPtr const & groups = nullptr)
	{
		
		//Unpacking the tuple with reference binding (c++17)
		const auto&[SaveK, SaveG] = ret;
		//SaveK    is a vector of size iter_to_store with (0.5*p*(p+1))-dimensional eigen vectors
		//SaveG    is a vector/map/unsigned_map of size iter_to_store with std::vector<bool> representing the adjacendy matrix and its frequency

		//It would be better to use std::reduce with execution::par but it is not supported in R and it is difficult to interface with Eigen
		VecCol MeanK( VecCol::Zero(SaveK[0].size())  );
		for(int i = 0; i < SaveK.size(); ++i){
			MeanK += SaveK[i];
		}
		MeanK /= iter_to_store;
		//plinks
		MatRow plinks(Compute_plinks(SaveG, iter_to_store, groups));
		return std::make_tuple(MeanK, plinks);
	}
	
	std::tuple<MatCol,MatCol> QuantileBeta(RetBeta const & SaveBeta, double const & alpha_lower = 0.05, double const & alpha_upper = 0.95)
	{
		using MyMat = std::vector< std::vector< std::vector<double> > >;
		// n x p x iter
		const int n = SaveBeta[0].cols();
		const int p = SaveBeta[0].rows();
		const int iter = SaveBeta.size();

		MyMat ordered(n);
		for(int nn = 0; nn < n; ++nn){
			ordered[nn].resize(p);
			for(int pp = 0; pp < p; ++pp)
				ordered[nn][pp].resize(iter);
		}

		//length(SaveBeta) = iter_to_store
		//dim(SaveBeta[k]) = (p x n)
		for(int nrep = 0; nrep < SaveBeta.size(); ++nrep){
			for(unsigned int j = 0; j < SaveBeta[nrep].cols(); ++j){
				ordered[j].resize(p);
				for(unsigned int i = 0; i < SaveBeta[nrep].rows(); ++i){
					ordered[j][i][nrep] = SaveBeta[nrep](i,j);
				}
			}
		}
		MatCol LowerBound(MatRow::Zero(p,n));
		MatCol UpperBound(MatRow::Zero(p,n));
		for(int nn = 0; nn < n; ++nn){
			for(int pp = 0; pp < p; ++pp){
				std::sort(ordered[nn][pp].begin(), ordered[nn][pp].end());
				LowerBound(pp,nn) = gsl_stats_quantile_from_sorted_data(&ordered[nn][pp].data()[0], 1, ordered[nn][pp].size(), alpha_lower);
				UpperBound(pp,nn) = gsl_stats_quantile_from_sorted_data(&ordered[nn][pp].data()[0], 1, ordered[nn][pp].size(), alpha_upper);
			}
		}
		return std::make_tuple(LowerBound, UpperBound);
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
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		dataset_Beta_rd  = H5Dopen(file, "/Beta", H5P_DEFAULT);
		SURE_ASSERT(dataset_Beta_rd>=0,"Cannot create dataset for Beta");
		//Read beta
		std::vector< MatCol > vett_Beta(saved_iter);
		for(int i = 0; i < saved_iter; ++i){
			vett_Beta[i] = HDF5conversion::ReadMatrix(dataset_Beta_rd, p, n, i);	
			//std::cout<<"#rows = "<<vett_Beta[i].rows()<<"; #cols = "<<vett_Beta[i].cols()<<std::endl;
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
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		if(vett_type == "Mu"){
			dataset_rd  = H5Dopen(file, "/Mu", H5P_DEFAULT);
			SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for Mu");
		}
		else
		{
			dataset_rd  = H5Dopen(file, "/Precision", H5P_DEFAULT);
			SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for Precision");
		}
		
		//Read 
		std::vector< VecCol > vect(saved_iter);
		for(int i = 0; i < saved_iter; ++i){
			vect[i] = HDF5conversion::ReadVector(dataset_rd, length, i);	
			//std::cout<<"#size = "<<vect[i].size()<<std::endl;
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
		file=H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT); //it is read only
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/TauEps", H5P_DEFAULT);
		SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for TauEps");
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
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/Beta", H5P_DEFAULT);
		SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for Beta");

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
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		if(vett_type == "Mu"){
			dataset_rd  = H5Dopen(file, "/Mu", H5P_DEFAULT);
			SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for Mu");
		}
		else
		{
			dataset_rd  = H5Dopen(file, "/Precision", H5P_DEFAULT);
			SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for Precision");
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
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/TauEps", H5P_DEFAULT);
		SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for TauEps");
		//Read chains all the chain
		std::vector<double> chain(stored_iter);
		double * ptr_chain = chain.data();
		HDF5conversion::StatusType status;
		status = H5Dread(dataset_rd,H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr_chain);
		SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
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
		SURE_ASSERT(file>0,"Cannot open file, read only mode selected");
		//Open datasets
		dataset_rd  = H5Dopen(file, "/Graphs", H5P_DEFAULT);
		SURE_ASSERT(dataset_rd>=0,"Cannot create dataset for Graphs");

		std::tie(Glist, traceplot_size, visited) = HDF5conversion::GetGraphsChain(dataset_rd, n_elem, stored_iter);
		MatRow plinks = Compute_plinks<decltype(Glist), unsigned int>(Glist, stored_iter, groups);
		return std::make_tuple(plinks, Glist, traceplot_size, visited);
	}

}

#endif