#ifndef __HDF5CONVERSION__
#define __HDF5CONVERSION__

#undef H5_USE_16_API
#include "hdf5.h"
#include "extendedAssert.h"
#include "include_headers.h"

namespace HDF5conversion{
	
	using FileType 		= hid_t;
	using DatasetType 	= hid_t;
	using DataspaceType = hid_t;
	using MemspaceType 	= hid_t;
	using StatusType 	= herr_t;
	using ScalarType 	= hsize_t;
	using MatCol 		= Eigen::MatrixXd;
	using VecCol 		= Eigen::VectorXd;
	using SampledGraphs = std::map< std::vector<unsigned int>, int>;

	//Takes a bi-dimensional dataset of dimension (p x iter_to_store*n) and adds a column matrix (p x n) starting from position iter
	void AddMatrix(DatasetType & dataset, MatCol & Mat, unsigned int const & iter); //for Matrices of double (thought for Beta matrices)
	
	//Takes a bi-dimensional dataset of dimension (p x iter_to_store*n) and reads a column matrix (p x n) starting from position iter
	MatCol ReadMatrix(DatasetType & dataset, unsigned int const & p, unsigned int const & n ,unsigned int const & iter); //for Matrices of double (thought for Beta matrices)

	//Takes a linear dataset of dimension (p*iter_to_store) and adds a p-dimensional vector starting from position iter
	void AddVector(DatasetType & dataset, VecCol & vect, unsigned int const & iter);

	//Takes a linear dataset of dimension (p*iter_to_store) and read a p-dimensional vector starting from position iter
	VecCol ReadVector(DatasetType & dataset, unsigned int const & p, unsigned int const & iter);

	//Takes a linear dataset of dimension (iter_to_store) and adds a scalar in position iter
	void AddScalar(DatasetType & dataset, double & val, unsigned int const & iter);

	//Takes a linear dataset of dimension (iter_to_store) and read the value in position iter
	double ReadScalar(DatasetType & dataset, unsigned int const & iter);

	//Takes a linear dataset of dimension (p*iter_to_store) and adds a p-dimensional vector starting from position iter
	void AddUintVector(DatasetType & dataset, std::vector<unsigned int> & vect, unsigned int const & iter);

	//Takes a linear dataset of dimension (p*iter_to_store) and read a p-dimensional vector starting from position iter
	std::vector<unsigned int> ReadUintVector(DatasetType & dataset, unsigned int const & p, unsigned int const & iter);

	// Takes a 2-dimnesional dataset (p x stored_iter*n) and extracts the samples values for the spline_index-th coefficients of the curve_index-th curve
	// In more abstract terms, gets the element in position (spline_index, curve_index) in all the stored_iter matrices that were saved.
	// spline_index and curve_index are zero-based defined. The first spline is defined by 0 index. The first curve is defined by 0 index
	VecCol GetChain_from_Matrix(DataspaceType & dataset, unsigned int const & spline_index, unsigned int const & curve_index, unsigned int const & stored_iter, unsigned int const & n_curves);

	// Takes a 1-dimnesional dataset (p*stored_iter) and extracts the samples values elements in index-th position
	// In other terms, it gets the element in position (index) in all the stored_iter vectors that were saved.
	// index is zero-based defined. The first element is defined by 0 index
	VecCol GetChain_from_Vector(DataspaceType & dataset, unsigned int const & index, unsigned int const & stored_iter, unsigned int const & p);

	//Gets a 1-dimension dataset with the stored graphs and returns a tuple with:
	//map with visited graphs and how many times where visited; Vector with the dimension of the visited graph; number of visited graphs;
	std::tuple<SampledGraphs, VecCol, int>
	GetGraphsChain(DatasetType & dataset, unsigned int const & n_elem, unsigned int const & stored_iter);
	
}





#endif

