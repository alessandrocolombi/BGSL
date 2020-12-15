#include "GSLwrappers.h"


namespace spline{

	std::tuple<MatType, std::vector<double> > 
	generate_design_matrix(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
	  					   std::vector<double> const & grid_points)
	{			
		unsigned int r = grid_points.size();
		//Checks
		if(a >= b)
			throw std::runtime_error("The domain in malformed, the first argument has to be stricly less then the second one");
		if(r <= 2 || grid_points[0] < a || *grid_points.rbegin() > b)
			throw std::runtime_error("The vector of grid points is malformed or incompatible with the provided interval");			
		//GSL B-Spline workspace
		gsl_bspline_workspace *bw;
		int nbreak = n_basis + 2 - order;  
	  	bw = gsl_bspline_alloc(order, nbreak);
	  	//Generate knots
	  	gsl_bspline_knots_uniform(a, b, bw);
	  	std::vector<double> knots(nbreak-2);
	  	//For same strange reason, bw->knots internally repeats order-times the first and the last value.
	  	for(int i = 0; i < nbreak-2; ++i){
	  		knots[i] = gsl_vector_get(bw->knots, i+order);
	  	}
	  	// Construct the fit matrix Basemat 
	  	gsl_vector *B;
	  	B = gsl_vector_alloc(n_basis);		  	
	  	MatType Basemat(MatType::Zero(r,n_basis));
	  	 
	  	// This loop cannot be inverted, it is possible only to compute all the sline in a given point and not one spline in all the points
	  	// This implies that if the Basemat il ColMajor, this operation won't be cache friendly.
	  	for(unsigned int i = 0; i < r; ++i)//for every grid point
	  	{
	  	    // compute B_j(xi) for all j 
	  	    gsl_bspline_eval(grid_points[i], B, bw);

	  	    // fill in row i of Basemat 
	  	    for(unsigned int j = 0; j < n_basis; ++j) //for every basis
	  	    {
	  	        Basemat(i,j) = gsl_vector_get(B, j);
	  	    }
	  	}

		gsl_bspline_free(bw); //free workspace
		gsl_vector_free(B);  //free vector B
	  	return std::make_tuple(Basemat, knots);
	}

	std::tuple<MatType, std::vector<double> > 
	generate_design_matrix(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
							unsigned int const & r)
	{
		if(a >= b)
			throw std::runtime_error("The domain in malformed, the first argument has to be stricly less then the second one");
		if(r <= 2)
			throw std::runtime_error("The number of grid points is too small");
		double h = (double)(b-a)/(double)(r-1);
		std::vector<double> grid_points(r);
		grid_points[0] 	 = a;
		grid_points[r-1] = b;
		for(int i = 1; i <= r-2; ++i){
			grid_points[i] = grid_points[i-1] + h;
		}
		return generate_design_matrix(order, n_basis, a, b, grid_points);
	}

	std::vector<MatType> evaluate_spline_derivative(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  					std::vector<double> const & grid_points, unsigned int const & nderiv)
	{
		unsigned int r = grid_points.size();
		//Checks
		if(a >= b)
			throw std::runtime_error("The domain in malformed, the first argument has to be stricly less then the second one");
		if(r <= 2 || grid_points[0] < a || *grid_points.rbegin() > b)
			throw std::runtime_error("The vector of grid points is malformed or incompatible with the provided interval");	
		if(nderiv == 0){
			auto[res,knots] = generate_design_matrix(order, n_basis, a, b, grid_points);
			return {res};
		}
		else{
			//GSL B-Spline workspace
			gsl_bspline_workspace *bw;
			int nbreak = n_basis + 2 - order;  
			bw = gsl_bspline_alloc(order, nbreak);
			//Generate knots
			gsl_bspline_knots_uniform(a, b, bw);

			//Create GSL matrix for storing spline derivatives
			gsl_matrix *dB;
			dB = gsl_matrix_alloc(n_basis, nderiv + 1);
			std::vector<MatType> Basemat_derivatives(nderiv+1); 
			for(MatType& Mat : Basemat_derivatives)
				Mat = MatType::Zero(r,n_basis);

			for(unsigned int i = 0; i < r; ++i)
			{
			    gsl_bspline_deriv_eval(grid_points[i], nderiv, dB, bw); //dB is RowMajor, can't change that
			    Eigen::Map<MatCol> dB_eigen(&(dB->data[0]),nderiv+1,n_basis);
			    for(unsigned int der = 0; der < Basemat_derivatives.size(); ++der){
			    	Basemat_derivatives[der].row(i) = dB_eigen.row(der);
			    }

			}
			//for(auto __v : Basemat_derivatives)
				//std::cout<<__v<<std::endl;
			gsl_bspline_free(bw); //free workspace
			gsl_matrix_free(dB); //free matrix dB
			return Basemat_derivatives;
		}
	}
	std::vector<MatType> evaluate_spline_derivative(unsigned int const & order, unsigned int const & n_basis, double const & a, double const & b, 
								  					unsigned int const & r, unsigned int const & nderiv)
	{
		if(r <= 2)
			throw std::runtime_error("The number of grid points is too small");
		double h = (double)(b-a)/(double)(r-1);
		std::vector<double> grid_points(r);
		grid_points[0] 	 = a;
		grid_points[r-1] = b;
		for(int i = 1; i <= r-2; ++i){
			grid_points[i] = grid_points[i-1] + h;
		}
		return evaluate_spline_derivative(order, n_basis, a, b, grid_points, nderiv);
	}
}