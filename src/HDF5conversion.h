#ifndef __HDF5CONVERSION__
#define __HDF5CONVERSION__

#undef H5_USE_16_API
#include "hdf5.h"
#include "extendedAssert.h"
#include "include_headers.h"

/* Functions for writing and reading on HDF5 datasets
*
* HDF5 datasets are defined by dataspaces, Each dataspace defines a rank, i.e the number of dimensions for the dataset, and a C-style array definining the size of each dimension.
* In this file, we refer to "Matrix" for a bi-dimensional dataset and "Vector" for one dimensional dataset. 
* In BGSL's samplers, at each iteration a new matrix/vector/scalar has to be added to the dataset. The idea is to create a (p x n*iter_to_store) bi-dimensional dataset and fill it with a new
* (p x n) matrix every time. This is done with AddMatrix() function. For what concern vectors, a lineat dataset of length (p * iter_to_store) is created and filled every time with p-dimensional 
* vectors via AddVector() function. Finally, scalars are stored in a one-dimensional dataset of length iter_to_store by means of AddScalar() function.
* Each add method has its Read version that performs the inverse operation.
* In order to store graphs, use AddUintVector() functions, which stores the upper triangular part in a linear dataset. All graphs are saved, once the sampling is done, use GetGraphsChain() function 
* to create an stl container with all the sampled graphs and the number of times they were visited.
* Do not get tricked by notation, AddVector() function works for vector of every size, not only for vector of size p where p is the number of basis.
*
* WARNING: This file is not general at all. It only works the specified types, which are indeed the one used in the sampling. For example, it is possible to add only ColumnMajor, dynamic, 
* matrices of double and ColumnVector of double. Moreover the graph has to store unsigned integers as values. 
* This is for sure not a fully satisfactory results, still there are a lot of troubles in templatazing this class, because it works directly with the buffer of data and for setting the type of storage
* hdf5-native types are used. It was not the only possibility but for sure it was the simplest one.
*/

namespace HDF5conversion{
	
	using FileType 		= hid_t;
	using DatasetType 	= hid_t;
	using DataspaceType = hid_t;
	using MemspaceType 	= hid_t;
	using StatusType 	= herr_t;
	using ScalarType 	= hsize_t;
	using MatCol 		= Eigen::MatrixXd;
	using MatRow 		= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using VecCol 		= Eigen::VectorXd;
	using VecRow   		= Eigen::RowVectorXd;
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
	std::vector<double> GetChain_from_Matrix(DataspaceType & dataset, unsigned int const & spline_index, unsigned int const & curve_index, unsigned int const & stored_iter, unsigned int const & n_curves);

	// Takes a 1-dimnesional dataset (p*stored_iter) and extracts the samples values elements in index-th position
	// In other terms, it gets the element in position (index) in all the stored_iter vectors that were saved.
	// index is zero-based defined. The first element is defined by 0 index
	std::vector<double> GetChain_from_Vector(DataspaceType & dataset, unsigned int const & index, unsigned int const & stored_iter, unsigned int const & p);

	//Gets a 1-dimension dataset with the stored graphs and returns a tuple with:
	//map with visited graphs and how many times where visited; Vector with the dimension of the visited graph; number of visited graphs;
	std::tuple<SampledGraphs, VecCol, int>
	GetGraphsChain(DatasetType & dataset, unsigned int const & n_elem, unsigned int const & stored_iter);
	
}





#endif

