#include "HDF5conversion.h"

namespace HDF5conversion{
	
	//Takes a bi-dimensional dataset of dimension (p x iter_to_store*n) and adds a column matrix (p x n) starting from position iter
	void AddMatrix(DatasetType & dataset, MatCol & Mat, unsigned int const & iter) //for Matrices of double (thought for Beta matrices)
	{
		MatRow Mat2(Mat); //hdf5 stores matrices in C-style that is row-wise. Has to respect that ordering.
		//dataset has dimension p x iter_to_store*n
		ScalarType p = static_cast<ScalarType>(Mat.rows());
		ScalarType n = static_cast<ScalarType>(Mat.cols());
		double * buffer = Mat2.data(); //Mat cannot be const because need to access to its buffer
		ScalarType offset[2] = {0,iter*n}; //initial point in complete matrix
		ScalarType count [2] = {1,1}; //leave 1,1 for basic usage. Means that only one block is added.
		ScalarType stride[2] = {1,1}; //leave 1,1 for basic usage. Means that only one block is added.
		ScalarType block [2] = {p,n}; //sub-matrix
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 2 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to add a matrix only to a bi-dimensional dataset");
		ScalarType Dspace_dims[2];
		H5Sget_simple_extent_dims(dataspace_sub, Dspace_dims, NULL);
		if(static_cast<ScalarType>(Dspace_dims[0]) != p)
			throw std::runtime_error("In order to add a (p x n) matrix, the inserted dataset has to have p rows");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, offset, stride, count, block); //define the memory space for the sub matrix (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of Mat that has to be saved. It has to have the same number of elements specified in dataspace_sub. Experience suggests to use it linear (i.e rank = 1 and not 2).
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = p*n; //having the same number of elements of the matrix
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    
	    //Write
	    status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace_sub, H5P_DEFAULT, buffer); //H5Dwrite(dataset, type, memspace, dataspace_sub, H5P_DEFAULT, data_buffer);
	    SURE_ASSERT(status>=0,"Cannot write file. Status"<< status);
	    //std::cout<<"Write! "<<std::endl;
	}
	//Takes a bi-dimensional dataset of dimension (p x iter_to_store*n) and reads a column matrix (p x n) starting from position iter
	MatCol ReadMatrix(DatasetType & dataset, unsigned int const & p, unsigned int const & n ,unsigned int const & iter) //for Matrices of double (thought for Beta matrices)
	{
		//dataset has dimension p x iter_to_store*n
		//MatCol result(MatCol::Zero(p,n)); //Instantiate space for result matrix //<-----------------
		MatRow result(MatRow::Zero(p,n)); //Instantiate space for result matrix //<-----------------
		double * buffer = result.data();
		ScalarType offset[2] = {0,iter*n}; //initial point in complete matrix
		ScalarType count [2] = {1,1}; //leave 1,1 for basic usage. Means that only one block is read.
		ScalarType stride[2] = {1,1}; //leave 1,1 for basic usage. Means that only one block is read.
		ScalarType block [2] = {p,n}; //sub-matrix
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset.
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 2 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to read a matrix only in a bi-dimensional dataset");
		ScalarType Dspace_dims[2];
		H5Sget_simple_extent_dims(dataspace_sub, Dspace_dims, NULL);
		if(static_cast<ScalarType>(Dspace_dims[0]) != p)
			throw std::runtime_error("In order to read a (p x n) matrix, the inserted dataset has to have p rows");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, offset, stride, count, block); //define the memory space for the sub matrix (in dataset)
		MemspaceType memspace; //Used to define the part of Matrix that has to be saved. It has to have the same number of elements specified in dataspace_sub. Experience suggests to use it linear (i.e rank = 1 and not 2).
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = p*n; //having the same number of elements of the matrix
		memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
		//Read
		status=H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace_sub, H5P_DEFAULT, buffer);
		SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
		//std::cout<<"result:"<<std::endl<<result<<std::endl;
		return result;
	}

	//Takes a linear dataset of dimension (p*iter_to_store) and adds a p-dimensional vector starting from position iter
	void AddVector(DatasetType & dataset, VecCol & vect, unsigned int const & iter)
	{
		//dataset has dimension p*iter_to_store
		ScalarType p = static_cast<ScalarType>(vect.size());
		double * buffer = vect.data(); //vect cannot be const because need to access to its buffer
		ScalarType offset = iter*p; //initial point in complete vector dataset
		ScalarType count  = 1; //leave 1 for basic usage. Means that only one block is added.
		ScalarType stride = 1; //leave 1 for basic usage. Means that only one block is added.
		ScalarType block  = p; //dimension of inserted vector
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 1 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to add a vector only in a one-dimensional dataset");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, &offset, &stride, &count, &block); //define the memory space for the sub vector (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of vect that has to be saved. It has to have the same number of elements specified in dataspace_sub. 
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = p; //having the same number of elements of the inserted vector
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    
	    //Write
	    status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace_sub, H5P_DEFAULT, buffer); //H5Dwrite(dataset, type, memspace, dataspace_sub, H5P_DEFAULT, data_buffer);
	    SURE_ASSERT(status>=0,"Cannot write file. Status"<< status);
	    //std::cout<<"Write! "<<std::endl;
	}

	//Takes a linear dataset of dimension (p*iter_to_store) and read a p-dimensional vector starting from position iter
	VecCol ReadVector(DatasetType & dataset, unsigned int const & p, unsigned int const & iter)
	{
		//dataset has dimension p*iter_to_store
		VecCol result(VecCol::Zero(p)); //Instantiate space for result vector
		double * buffer = result.data(); 
		ScalarType offset = iter*p; //initial point in complete vector dataset
		ScalarType count  = 1; //leave 1 for basic usage. Means that only one block is read.
		ScalarType stride = 1; //leave 1 for basic usage. Means that only one block is read.
		ScalarType block  = p; //dimension of inserted vector
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 1 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to read a vector only in a one-dimensional dataset");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, &offset, &stride, &count, &block); //define the memory space for the sub vector (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of vector that has to be saved. It has to have the same number of elements specified in dataspace_sub. 
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = p; //having the same number of elements of the inserted vector
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    //Read
	    status=H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace_sub, H5P_DEFAULT, buffer);
	    SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
	    //std::cout<<"result:"<<std::endl<<result<<std::endl;
	    return result;
	}
	
	//Takes a linear dataset of dimension (iter_to_store) and adds a scalar in position iter
	void AddScalar(DatasetType & dataset, double & val, unsigned int const & iter)
	{
		//dataset has dimension iter_to_store
		double * buffer = &val;
		ScalarType offset = iter; //initial point in complete vector dataset
		ScalarType count  = 1; //leave 1 for basic usage. Means that only one block is added.
		ScalarType stride = 1; //leave 1 for basic usage. Means that only one block is added.
		ScalarType block  = 1; //adding just a scalar
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 1 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to add a scalar only in a one-dimensional dataset");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, &offset, &stride, &count, &block); //define the memory space for the sub vector (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of mu that has to be saved. It has to have the same number of elements specified in dataspace_sub. 
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = 1; //having just one element because it is adding a scalr
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    
	    //Write
	    status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace_sub, H5P_DEFAULT, buffer); //H5Dwrite(dataset, type, memspace, dataspace_sub, H5P_DEFAULT, data_buffer);
	 	SURE_ASSERT(status>=0,"Cannot write file. Status"<< status);
	    //std::cout<<"Write! "<<std::endl;
	}

	//Takes a linear dataset of dimension (iter_to_store) and read the value in position iter
	double ReadScalar(DatasetType & dataset, unsigned int const & iter)
	{
		//dataset has dimension iter_to_store
		double result{0}; //Instantiate space for result 
		double * buffer = &result; 
		ScalarType offset = iter; 	//initial point in complete vector dataset
		ScalarType count  = 1; 		//leave 1 for basic usage. Means that only one block is read.
		ScalarType stride = 1; 		//leave 1 for basic usage. Means that only one block is read.
		ScalarType block  = 1;		//reading just a scalar
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 1 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to read a scalar only in a one-dimensional dataset");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, &offset, &stride, &count, &block); //define the memory space for the sub vector (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of mu that has to be saved. It has to have the same number of elements specified in dataspace_sub. 
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = 1; //having the same number of elements of the inserted vector
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    //Read
	    status=H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace_sub, H5P_DEFAULT, buffer);
	    SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
	    //std::cout<<"result:"<<std::endl<<result<<std::endl;
	    return result;
	}

	//Takes a linear dataset of dimension (p*iter_to_store) and adds a p-dimensional vector starting from position iter
	void AddUintVector(DatasetType & dataset, std::vector<unsigned int> & vect, unsigned int const & iter)
	{
		//dataset has dimension p*iter_to_store
		ScalarType p = static_cast<ScalarType>(vect.size());
		unsigned int * buffer = vect.data(); //vect cannot be const because need to access to its buffer
		ScalarType offset = iter*p; //initial point in complete vector dataset
		ScalarType count  = 1; //leave 1 for basic usage. Means that only one block is added.
		ScalarType stride = 1; //leave 1 for basic usage. Means that only one block is added.
		ScalarType block  = p; //dimension of inserted vector
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 1 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to add a vector only in a one-dimensional dataset");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, &offset, &stride, &count, &block); //define the memory space for the sub vector (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of vect that has to be saved. It has to have the same number of elements specified in dataspace_sub. 
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = p; //having the same number of elements of the inserted vector
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    
	    //Write
	    status = H5Dwrite(dataset, H5T_NATIVE_UINT, memspace, dataspace_sub, H5P_DEFAULT, buffer); //H5Dwrite(dataset, type, memspace, dataspace_sub, H5P_DEFAULT, data_buffer);
	    SURE_ASSERT(status>=0,"Cannot write file. Status"<< status);
	    //std::cout<<"Write! "<<std::endl;
	}

	//Takes a linear dataset of dimension (p*iter_to_store) and read a p-dimensional vector starting from position iter
	std::vector<unsigned int> ReadUintVector(DatasetType & dataset, unsigned int const & p, unsigned int const & iter)
	{
		//dataset has dimension p*iter_to_store
		std::vector<unsigned int> result(p); //Instantiate space for result vector
		unsigned int * buffer = result.data(); 
		ScalarType offset = iter*p; //initial point in complete vector dataset
		ScalarType count  = 1; //leave 1 for basic usage. Means that only one block is read.
		ScalarType stride = 1; //leave 1 for basic usage. Means that only one block is read.
		ScalarType block  = p; //dimension of inserted vector
		
		DataspaceType dataspace_sub; //Used to define the part of dataset that has to be modified.
		dataspace_sub =  H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Check dimension
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace_sub));
		if( rank != 1 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("It is possible to read a vector only in a one-dimensional dataset");
		//If fine, go on describing the "selection" for that dataspace
		StatusType status = H5Sselect_hyperslab(dataspace_sub, H5S_SELECT_SET, &offset, &stride, &count, &block); //define the memory space for the sub vector (in dataset)
		//hssize_t numel =  H5Sget_select_npoints (dataspace_sub); //just a check
		//std::cout<<"numel = "<<numel<<std::endl;

		MemspaceType memspace; //Used to define the part of vector that has to be saved. It has to have the same number of elements specified in dataspace_sub. 
		int rank_mem = 1; //linear buffer
		ScalarType dims_sub = p; //having the same number of elements of the inserted vector
	    memspace = H5Screate_simple(rank_mem, &dims_sub, NULL);  //create dataspace for memspace
	    //numel =  H5Sget_select_npoints (memspace); //just a check
	    //std::cout<<"numel = "<<numel<<std::endl;
	    //Read
	    status=H5Dread(dataset, H5T_NATIVE_UINT, memspace, dataspace_sub, H5P_DEFAULT, buffer);
	    SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
	    return result;
	}

	// Takes a 2-dimnesional dataset (p x stored_iter*n) and extracts the samples values for the spline_index-th coefficients of the curve_index-th curve
	// In more abstract terms, gets the element in position (spline_index, curve_index) in all the stored_iter matrices that were saved.
	// spline_index and curve_index are zero-based defined. The first spline is defined by 0 index. The first curve is defined by 0 index
	std::vector<double> GetChain_from_Matrix(DataspaceType & dataset, unsigned int const & spline_index, unsigned int const & curve_index, unsigned int const & stored_iter, unsigned int const & n_curves)
	{
		std::vector<double> chain(stored_iter);
		double * pchain = chain.data();

		DataspaceType dataspace, mid2; 
		dataspace = H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Checks
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace));
		if( rank != 2 ) //need to check that the inserted dataset is bi-dimensional (its rank is 2)
			throw std::runtime_error("The GetChain_from_Matrix() function requires in input a 2-dimension dataset. Use GetChain_from_Vector() for 1-dimnesional datasets");
		ScalarType Dspace_dims[2];
		H5Sget_simple_extent_dims(dataspace, Dspace_dims, NULL);
		if(static_cast<ScalarType>(Dspace_dims[0]) <= static_cast<ScalarType>(spline_index))
			throw std::runtime_error("Invalid spline_index request. It exceeds dataset dimension");
		if(static_cast<ScalarType>(Dspace_dims[1]) != static_cast<ScalarType>(stored_iter*n_curves))
			throw std::runtime_error("Dataset dimension is not compatible with stored_iter and n_curves parameters. dataset has to be (p x stored_iter*n_curves)");
		if(curve_index >= n_curves)
			throw std::runtime_error("Invalid curve_index request. It exceeds dataset dimension");
		//Need to define a linear space with dimension of the number of selected points	
		ScalarType PointsToRead = stored_iter;
		mid2 = H5Screate_simple(1, &PointsToRead, NULL);
		//Specity coordinates of the points to be selected. Has to respect HDF5 notation (which sucks btw)
		//ScalarType coord[stored_iter][2]; /* first dimension is the number of points that will be selected, second argument is the Dataset rank */
		auto coord = new ScalarType [stored_iter][2]; /* first dimension is the number of points that will be selected, second argument is the Dataset rank */
		//std::cout<<"coordinate:"<<std::endl;
		for(int i = 0; i < stored_iter; ++i){ //fill the array 
			//coord[i][0] = spline_index; coord[i][1] = curve_index + n_curves*i; //coefficients are the defined with the following notation:
			coord[i][0] = spline_index; coord[i][1] = curve_index + n_curves*i; //coefficients are the defined with the following notation:
			//the i-th (first index) of the stored_iter points has spline_index (the assigned value) as first coordinate (the second index)
			//the i-th (first index) of the stored_iter points has curve_index + n_curves*i (the assigned value) as second coordinate (the second index)
			//std::cout<<"("<<spline_index<<", "<<curve_index + n_curves*i<<")"<<std::endl;
		}				
		StatusType status = H5Sselect_elements(dataspace, H5S_SELECT_SET, stored_iter, *coord);
		status = H5Dread(dataset, H5T_NATIVE_DOUBLE, mid2, dataspace, H5P_DEFAULT, pchain);
		SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
		//std::cout<<"chain:"<<std::endl<<chain<<std::endl;
		delete [] coord; //check with valgrind, no leak are possible
		return chain;
	}

	// Takes a 1-dimnesional dataset (p*stored_iter) and extracts the samples values elements in index-th position
	// In other terms, it gets the element in position (index) in all the stored_iter vectors that were saved.
	// index is zero-based defined. The first element is defined by 0 index
	std::vector<double> GetChain_from_Vector(DataspaceType & dataset, unsigned int const & index, unsigned int const & stored_iter, unsigned int const & p)
	{
		std::vector<double> chain(stored_iter);
		double * pchain = chain.data();

		DataspaceType dataspace, mid2; 
		dataspace = H5Dget_space(dataset); // gets dataspace that is defining dataset
		//Checks
		int rank = static_cast<int>(H5Sget_simple_extent_ndims(dataspace));
		if( rank != 1 ) //need to check that the inserted dataset is one-dimensional (its rank is 1)
			throw std::runtime_error("It is possible to add a matrix only to a bi-dimensional dataset");
		ScalarType Dspace_dims[1];
		H5Sget_simple_extent_dims(dataspace, Dspace_dims, NULL);
		if(static_cast<ScalarType>(Dspace_dims[0]) != static_cast<ScalarType>(stored_iter*p))
			throw std::runtime_error("Dataset dimension is not compatible with stored_iter and p parameters. dataset has to be of size (p*stored_iter)");
		if(index >= p)
			throw std::runtime_error("Invalid index request. It exceeds dataset dimension");
		//Need to define a linear space with dimension of the number of selected points	
		ScalarType PointsToRead = stored_iter;
		mid2 = H5Screate_simple(1, &PointsToRead, NULL);
		//Specity coordinates of the points to be selected. Has to respect HDF5 notation (which sucks btw)
		auto coord = new ScalarType [stored_iter][1]; /* first dimension is the number of points that will be selected, second argument is the Dataset rank */
		//std::cout<<"coordinate:"<<std::endl;
		for(int i = 0; i < stored_iter; ++i){ //fill the array 
			coord[i][0] = index + i*p;  //coefficients are the defined with the following notation:
			//the i-th (first index) of the stored_iter points has index + i*p (the assigned value) as first coordinate (the second index)
			//std::cout<<"("<<spline_index<<", "<<curve_index + n_curves*i<<")"<<std::endl;
		}				
		StatusType status = H5Sselect_elements(dataspace, H5S_SELECT_SET, stored_iter, *coord);
		status = H5Dread(dataset, H5T_NATIVE_DOUBLE, mid2, dataspace, H5P_DEFAULT, pchain);
		SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
		//std::cout<<"chain:"<<std::endl<<chain<<std::endl;
		delete [] coord; //check with valgrind, no leak are possible

		SampledGraphs s;
		return chain;
	}
	
	//Gets a 1-dimension dataset with the stored graphs and returns a tuple with:
	//map with visited graphs and how many times where visited; Vector with the dimension of the visited graph; number of visited graphs;
	std::tuple<SampledGraphs, VecCol, int>
	GetGraphsChain(DatasetType & dataset, unsigned int const & n_elem, unsigned int const & stored_iter)
	{
		SampledGraphs SaveG;
		VecCol GraphSize(VecCol::Zero(stored_iter));
		int visited{0};
		for(int i = 0; i < stored_iter; ++i){
			std::vector<unsigned int> sampledGraph_adj = HDF5conversion::ReadUintVector(dataset, n_elem, i); //Reads the graph from file
			GraphSize(i) = std::accumulate(sampledGraph_adj.cbegin(), sampledGraph_adj.cend(), 0);
			auto it = SaveG.find(sampledGraph_adj); //seach if it was already visited
			if(it == SaveG.end()){ //if not, add
				SaveG.insert(std::make_pair(sampledGraph_adj, 1));
				visited++;
			}
			else{
				it->second++;
			}
		}
		return std::make_tuple(SaveG, GraphSize, visited);
	}

}