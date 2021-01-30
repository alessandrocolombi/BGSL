#ifndef __RJMH_HPP__
#define __RjMH_HPP__

#include "GGM.h"


/*
	New version using enable_if. 
	Only one class is instantiated, no specialization. RJ methods are overloaded using enable if. What are the benefits? 
	Now the specialization for complete graphs is available for ALL complete graphs type, it is much easier to generalize. If a new implementation of
	GraphType is given, if it is a complete graph, this "specialization" is immediatelty available without the need of writing an entire new class.
	Code is also safer for BlockGraphs. First it was simply called if GraphStructure was not GraphType. Now, it is called only if GraphStructure represents
	a BlockGraph.
	How is it done? 
	enable_if relies on SFINAE. The first two arguments (GG and TT) are needed to enforce template parameter substitution. It would not work otherwise, SFINAE
	only kicks in with template parameters. The third argument is the enable_if condition. Checks if isBlockGraph or isCompleteGraph are true. If so, a new type
	if defined but it is defaulted to 0 since it won't be needed. It is admissible since unused template parameter may or may not have a name. It is a sort of
	"dummy" parameter, necessary only to activate the mechanism of enable if.
	REMARK: note the space between ">" and "=0".

	This solution is a sort of function overload of at compile time.
	If neither of the static if is true, then enable_if declares no type and a compiler error like " error: no type named 'type' in 'struct std::enable_if<false, unsigned int>'" is raised.

	We implemented two type of reversible jumps. RJ is our solution, the comlpetion operation for those elements that depends on the changed link is done according to all new elements.
	RJ_new is the solution proposed by Lenkoski, the same operation is done with respect to old values of precision matrix. The reason is not explained, we implemented both, we are still 
	testing the differences.
*/


template<template <typename> class GraphStructure = GraphType, typename T = unsigned int >
class ReversibleJumpsMH : public GGM<GraphStructure, T> {
	public:
		//Typedefs
		using IdxType  	  		= typename GGMTraits<GraphStructure, T>::IdxType;
		using MatRow      		= typename GGMTraits<GraphStructure, T>::MatRow;
		using MatCol      		= typename GGMTraits<GraphStructure, T>::MatCol;
		using VecRow      		= typename GGMTraits<GraphStructure, T>::VecRow;
		using VecCol      		= typename GGMTraits<GraphStructure, T>::VecCol;
		using CholTypeRow 		= typename GGMTraits<GraphStructure, T>::CholTypeRow;
		using CholTypeCol 		= typename GGMTraits<GraphStructure, T>::CholTypeCol;
		using GroupsPtr   		= typename GGMTraits<GraphStructure, T>::GroupsPtr;
		using PriorPtr			= typename GGMTraits<GraphStructure, T>::PriorPtr;
		using Graph 			= typename GGMTraits<GraphStructure, T>::Graph;
		template <typename S>
		using CompleteSkeleton  = typename GGMTraits<GraphStructure, T>::template CompleteSkeleton<S>;
		using CompleteType 		= typename GGMTraits<GraphStructure, T>::CompleteType;
		using PrecisionType     = typename GGMTraits<GraphStructure, T>::PrecisionType;
		using ReturnType 		= typename GGMTraits<GraphStructure, T>::ReturnType;
		//Constructors
		ReversibleJumpsMH(	PriorPtr& _ptr_prior,double const & _b , MatCol const & _D, double const & _trGwishSampler, double const & _sigma, 
						  			unsigned int const & _MCiterPrior = 100):
						  			GGM<GraphStructure, T>(_ptr_prior, _b, _D, _trGwishSampler), sigma(_sigma), MCiterPrior(_MCiterPrior) {}
		ReversibleJumpsMH(	PriorPtr& _ptr_prior,unsigned int const & _p, double const & _trGwishSampler, double const & _sigma, 
									unsigned int const & _MCiterPrior = 100 ):
						  			GGM<GraphStructure, T>(_ptr_prior, _p, _trGwishSampler), sigma(_sigma), MCiterPrior(_MCiterPrior) {}
		
		//Our proposed RJ move
		template< template <typename> class GG = GraphStructure, typename TT = T,
					std::enable_if_t< internal_type_traits::isBlockGraph<GG,TT>::value , TT> =0  > //BlockGraph case						  			
		std::tuple<PrecisionType, double, double> RJ(CompleteType const & Gnew_CompleteView, PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine = sample::GSL_RNG());
		
		template< template <typename> class GG = GraphStructure, typename TT = T,
					std::enable_if_t< internal_type_traits::isCompleteGraph<GG,TT>::value , TT> =0  > //CompleteGraphs case
		std::tuple<PrecisionType, double, double> RJ(CompleteType const & Gnew, PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine = sample::GSL_RNG());
		

		//Type of RJ proposed by Lenkoski.
		template< template <typename> class GG = GraphStructure, typename TT = T,
					std::enable_if_t< internal_type_traits::isBlockGraph<GG,TT>::value , TT> =0  > //BlockGraph case
		std::tuple<PrecisionType, double, double> RJ_new(CompleteType const & Gnew_CompleteView, PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine = sample::GSL_RNG());

		template< template <typename> class GG = GraphStructure, typename TT = T,
					std::enable_if_t< internal_type_traits::isCompleteGraph<GG,TT>::value , TT> =0  > //CompleteGraphs case
		std::tuple<PrecisionType, double, double> RJ_new(CompleteType const & Gnew, PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine = sample::GSL_RNG() );
		
		//Call operator
		ReturnType operator()(MatCol const & data, unsigned int const & n, Graph & Gold, double alpha, sample::GSL_RNG const & engine = sample::GSL_RNG());
	protected:
		double const sigma; //it is a standard deviation
		unsigned int MCiterPrior;
};


template<template <typename> class GraphStructure, typename T>
template< template <typename> class GG, typename TT, std::enable_if_t< internal_type_traits::isBlockGraph<GG,TT>::value , TT> > //BlockGraphs
std::tuple<typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType, double, double>
ReversibleJumpsMH<GraphStructure, T>::RJ( typename ReversibleJumpsMH<GraphStructure, T>::CompleteType const & Gnew_CompleteView,
										  typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine)
{
		using Graph 		= GraphStructure<T>;
		using Container		= std::vector< std::pair<unsigned int, unsigned int> >;
		using Citerator 	= Container::const_iterator;

		unsigned int p(Kold_prior.get_matrix().rows());
		//2) Find all the links that are changing in Complete form 
		const std::pair<unsigned int, unsigned int>& changed_link = this->selected_link; //For lighter notation
		const std::vector<unsigned int> A_l0(Gnew_CompleteView.get_group(changed_link.first));
		const unsigned int sizeAl0{static_cast<unsigned int>(A_l0.size())};
		unsigned int pos_Al0{0};
		Container L(Gnew_CompleteView.map_to_complete(changed_link.first, changed_link.second)); //Follow the notation of the manual
		Citerator it_L = L.cbegin();
		//Not the best possibile choice in term of efficiency but i prefer to create a function for sake of clarity and possibile generalizations.
		//In this way, if something has to be changed it is enough to change it here and not in the loop			
		auto build_jacobian_esponent = [&changed_link, &A_l0, &Gnew_CompleteView](unsigned int const & pos){
			if(changed_link.first != changed_link.second)
				return Gnew_CompleteView.get_group_size(changed_link.second);
			else
				return (unsigned int)(A_l0.size() - 1 - pos);
		};
		//a) Fill new Phi
		MatRow Phi_new(MatRow::Zero(p,p));
		if(!Kold_prior.isFactorized){
			Kold_prior.compute_Chol();
		}
		MatRow Phi_old(Kold_prior.get_upper_Chol()); 

		double log_element_proposal{0};
		double log_jacobian{0};

		//i = 0 case)
		Phi_new(0,0) = Phi_old(0,0);
		for(unsigned int j = 1; j < p; ++j){
			if(Gnew_CompleteView(0,j) == false){ //There is no link. Is it one of the removed one?
				Phi_new(0,j) = 0; 	//Even if it is, it is not a free element and has to be computed by completion operation
				if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair((unsigned int)0,j) == *it_L){
					log_element_proposal +=  (Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j)) ;
					it_L++;
				}
			}
			else{ //There is a link. Is it the new one?
				if(Move == MoveType::Add && it_L != L.cend() && std::make_pair((unsigned int)0,j) == *it_L){
					Phi_new(0,j) = sample::rnorm()(engine, Phi_old(0,j), this->sigma); //The new element is a free element
					log_element_proposal += (Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j));
					//log_jacobian += Phi_old(0,0);
					it_L++;
				}
				else
					Phi_new(0,j) = Phi_old(0,j);
			}
		}	
		if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == 0){
			log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(0,0));
			pos_Al0++;
		}
		//i = 1 case)
		Phi_new(1,1) = Phi_old(1,1);
		for(unsigned int j = 2; j < p; ++j){

			if(Gnew_CompleteView(1,j) == false){ //There is no link. Is it the removed one?
				Phi_new(1,j) = - ( Phi_new(0,1)*Phi_new(0,j) )/Phi_new(1,1); //Even if it is, it is not a free element and has to be computed by completion operation
				if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair((unsigned int)1,j) == *it_L){
					log_element_proposal += (Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j));
					it_L++;
				}

			}
			else{ //There is a link. Is it the new one?

				if(Move == MoveType::Add && it_L != L.cend() && std::make_pair((unsigned int)1,j) == *it_L){
					Phi_new(1,j) = sample::rnorm()(engine, Phi_old(1,j), this->sigma); //The new element is a free element
					log_element_proposal +=	(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j));
					it_L++;
				}
				else
					Phi_new(1,j) = Phi_old(1,j);
			}
		}
		if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == 1){
			log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(1,1));
			pos_Al0++;
		}
		//i > 1 case)
		for(unsigned int i = 2; i < p-1; ++i){

			Phi_new(i,i) = Phi_old(i,i);
			VecCol Psi_i(Phi_new.block(0,i,i,1));
			for(unsigned int j = i+1; j < p; ++j){
				if(Gnew_CompleteView(i,j) == false){ //There is no link. Is it the removed one?
					//Even if it is, it is not a free element and has to be computed by completion operation
					Phi_new(i,j) = - ( Psi_i.dot(VecCol (Phi_new.block(0,j,i,1))) )/Phi_new(i,i);
					if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair(i,j) == *it_L){
						log_element_proposal += (Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j));
						it_L++;
					}

				}
				else{ //There is a link. Is it the new one?

					if(Move == MoveType::Add && it_L != L.cend() && std::make_pair(i,j) == *it_L){ 
						Phi_new(i,j) = sample::rnorm()(engine, Phi_old(i,j), this->sigma); //The new element is a free element
						log_element_proposal += (Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j));
						it_L++;
					}
					else
						Phi_new(i,j) = Phi_old(i,j);
				}
			}

			if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == i){
				log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(i,i));
				pos_Al0++;
			}
		}
		Phi_new(p-1,p-1) = Phi_old(p-1,p-1); //Last element
		if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == p-1){
			log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(p-1,p-1));
			pos_Al0++;
		}
		log_element_proposal /= 2*this->sigma*this->sigma;
		log_element_proposal += static_cast<double>(L.size())*(0.5*utils::log_2pi + std::log(std::abs(this->sigma)));
		return std::make_tuple( 
				PrecisionType (Phi_new, Kold_prior.get_shape(), Kold_prior.get_inv_scale(), Kold_prior.get_chol_invD() ),
				log_element_proposal, log_jacobian );
}

template<template <typename> class GraphStructure, typename T>
template< template <typename> class GG, typename TT, std::enable_if_t< internal_type_traits::isCompleteGraph<GG,TT>::value , TT> > //CompleteGraphs
std::tuple<typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType, double, double>
ReversibleJumpsMH<GraphStructure, T>::RJ( typename ReversibleJumpsMH<GraphStructure, T>::CompleteType const & Gnew,
										  typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine)
{
	using Graph = GraphStructure<T>;

	const unsigned int p(Kold_prior.get_matrix().rows());
	const std::pair<unsigned int, unsigned int>& changed_link = this->selected_link; //For lighter notation
	double log_element_proposal{0};
	//a) Fill new Phi
	MatRow Phi_new(MatRow::Zero(p,p));

	if(!Kold_prior.isFactorized){
		Kold_prior.compute_Chol();
	}

	MatRow Phi_old(Kold_prior.get_upper_Chol()); 

	bool found=false;

	//i = 0 case)
	Phi_new(0,0) = Phi_old(0,0);
	for(unsigned int j = 1; j < p; ++j){

		if(Gnew(0,j) == false){ //There is no link. Is it the removed one?
			Phi_new(0,j) = 0; //Even if it is, it is not a free element and has to be computed by completion operation
			if(Move == MoveType::Remove && !found && std::make_pair((unsigned int)0,j) == changed_link){
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j))/(2*this->sigma*this->sigma);
			}

		}
		else{ //There is a link. Is it the new one?
			if(Move == MoveType::Add && !found && std::make_pair((unsigned int)0,j) == changed_link){
				Phi_new(0,j) = sample::rnorm()(engine, Phi_old(0,j), this->sigma); //The new element is a free element
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j))/(2*this->sigma*this->sigma);
			}
			else
				Phi_new(0,j) = Phi_old(0,j);
		}
	}
	//i = 1 case)
	Phi_new(1,1) = Phi_old(1,1);
	for(unsigned int j = 2; j < p; ++j){

		if(Gnew(1,j) == false){ //There is no link. Is it the removed one?
			Phi_new(1,j) = - ( Phi_new(0,1)*Phi_new(0,j) )/Phi_new(1,1); //Even if it is, it is not a free element and has to be computed by completion operation
			if(Move == MoveType::Remove && !found && std::make_pair((unsigned int)1,j) == changed_link){
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j))/(2*this->sigma*this->sigma);
			}

		}
		else{ //There is a link. Is it the new one?

			if(Move == MoveType::Add && !found && std::make_pair((unsigned int)1,j) == changed_link){
				found = true;
				Phi_new(1,j) = sample::rnorm()(engine, Phi_old(1,j), this->sigma); //The new element is a free element
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j))/(2*this->sigma*this->sigma);
			}
			else
				Phi_new(1,j) = Phi_old(1,j);
		}//changed_link è davvero una pair
	}
	//i > 1 case)
	for(unsigned int i = 2; i < p-1; ++i){

		Phi_new(i,i) = Phi_old(i,i);
		VecCol Psi_i(Phi_new.block(0,i,i,1));
		for(unsigned int j = i+1; j < p; ++j){
			if(Gnew(i,j) == false){ //There is no link. Is it the removed one?
				Phi_new(i,j) = - ( Psi_i.dot(VecCol (Phi_new.block(0,j,i,1))) )/Phi_new(i,i);
				if(Move == MoveType::Remove && !found && std::make_pair(i,j) == changed_link){
					found = true;
					log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
											(Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j))/(2*this->sigma*this->sigma);
				}

			}
			else{ //There is a link. Is it the new one?

				if(Move == MoveType::Add && !found && std::make_pair(i,j) == changed_link){ 
					found = true;
					Phi_new(i,j) = sample::rnorm()(engine, Phi_old(i,j), this->sigma); //The new element is a free element
					log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
											(Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j))/(2*this->sigma*this->sigma);
				}
				else
					Phi_new(i,j) = Phi_old(i,j);
			}//changed_link è davvero una pair
		}
	}
	Phi_new(p-1,p-1) = Phi_old(p-1,p-1); //Last element
	//return
	return std::make_tuple( PrecisionType (Phi_new, Kold_prior.get_shape(), Kold_prior.get_inv_scale(), Kold_prior.get_chol_invD() ),
			log_element_proposal, std::log(Phi_old(changed_link.first, changed_link.first)) );	
}

template<template <typename> class GraphStructure, typename T>
template< template <typename> class GG, typename TT, std::enable_if_t< internal_type_traits::isBlockGraph<GG,TT>::value , TT> > //BlockGraphs
std::tuple<typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType, double, double>
ReversibleJumpsMH<GraphStructure, T>::RJ_new( typename ReversibleJumpsMH<GraphStructure, T>::CompleteType const & Gnew_CompleteView,
										 	  typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine)
{
		using Graph 		= GraphStructure<T>;
		using Container		= std::vector< std::pair<unsigned int, unsigned int> >;
		using Citerator 	= Container::const_iterator;

		unsigned int p(Kold_prior.get_matrix().rows());
		//2) Find all the links that are changing in Complete form 
		const std::pair<unsigned int, unsigned int>& changed_link = this->selected_link; //For lighter notation
		const std::vector<unsigned int> A_l0 = Gnew_CompleteView.get_group(changed_link.first);
		const unsigned int sizeAl0{static_cast<unsigned int>(A_l0.size())};
		unsigned int pos_Al0{0};
		Container L(Gnew_CompleteView.map_to_complete(changed_link.first, changed_link.second));
		Citerator it_L = L.cbegin();
		//Not the best possibile choice in term of efficiency but i prefer to create a function for sake of clarity and possibile generalizations.
		//In this way, if something has to be changed it is enough to change it here and not in the loop			
		auto build_jacobian_esponent = [&changed_link, &A_l0, &Gnew_CompleteView](unsigned int const & pos){
			if(changed_link.first != changed_link.second)
				return Gnew_CompleteView.get_group_size(changed_link.second);
			else
				return (unsigned int)(A_l0.size() - 1 - pos);
		};
		//a) Fill new Phi
		MatRow Phi_new(MatRow::Zero(p,p));
		if(!Kold_prior.isFactorized){
			Kold_prior.compute_Chol();
		}
		MatRow Phi_old(Kold_prior.get_upper_Chol()); 

		double log_element_proposal{0};
		double log_jacobian{0};
		//i = 0 case)
		Phi_new(0,0) = Phi_old(0,0);
		for(unsigned int j = 1; j < p; ++j){
			if(Gnew_CompleteView(0,j) == false){ //There is no link. Is it one of the removed one?
				Phi_new(0,j) = 0; 	//Even if it is, it is not a free element and has to be computed by completion operation
				if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair((unsigned int)0,j) == *it_L){
					log_element_proposal +=  (Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j)) ;
					it_L++;
				}
			}
			else{ //There is a link. Is it the new one?
				if(Move == MoveType::Add && it_L != L.cend() && std::make_pair((unsigned int)0,j) == *it_L){
					Phi_new(0,j) = sample::rnorm()(engine, Phi_old(0,j), this->sigma); //The new element is a free element
					log_element_proposal += (Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j));
					it_L++;
				}
				else
					Phi_new(0,j) = Phi_old(0,j);
			}
		}	
		if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == 0){
			log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(0,0));
			pos_Al0++;
		}
		//i = 1 case)
		Phi_new(1,1) = Phi_old(1,1);
		for(unsigned int j = 2; j < p; ++j){

			if(Gnew_CompleteView(1,j) == false){ //There is no link. Is it the removed one?
				if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair((unsigned int)1,j) == *it_L){
					Phi_new(1,j) = - ( Phi_old(0,1)*Phi_old(0,j) )/Phi_old(1,1); 
					log_element_proposal += (Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j));
					it_L++;
				}
				else{
					Phi_new(1,j) = - ( Phi_new(0,1)*Phi_new(0,j) )/Phi_new(1,1); 
				}

			}
			else{ //There is a link. Is it the new one?

				if(Move == MoveType::Add && it_L != L.cend() && std::make_pair((unsigned int)1,j) == *it_L){
					Phi_new(1,j) = sample::rnorm()(engine, Phi_old(1,j), this->sigma); //The new element is a free element
					log_element_proposal +=	(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j));
					it_L++;
				}
				else
					Phi_new(1,j) = Phi_old(1,j);
			}
		}
		if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == 1){
			log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(1,1));
			pos_Al0++;
		}
		//i > 1 case)
		for(unsigned int i = 2; i < p-1; ++i){

			Phi_new(i,i) = Phi_old(i,i);
			VecCol Psi_i(Phi_new.block(0,i,i,1));
			for(unsigned int j = i+1; j < p; ++j){
				if(Gnew_CompleteView(i,j) == false){ //There is no link. Is it the removed one?
					if(Move == MoveType::Remove && it_L != L.cend() && std::make_pair(i,j) == *it_L){
						VecCol Psi_old_i(Phi_old.block(0,i,i,1));
						Phi_new(i,j) = - ( Psi_old_i.dot(VecCol (Phi_old.block(0,j,i,1))) )/Phi_old(i,i);
						log_element_proposal += (Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j));
						it_L++;
					}
					else{
						Phi_new(i,j) = - ( Psi_i.dot(VecCol (Phi_new.block(0,j,i,1))) )/Phi_new(i,i);
					}

				}
				else{ //There is a link. Is it the new one?

					if(Move == MoveType::Add && it_L != L.cend() && std::make_pair(i,j) == *it_L){ 
						Phi_new(i,j) = sample::rnorm()(engine, Phi_old(i,j), this->sigma); //The new element is a free element
						log_element_proposal += (Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j));
						it_L++;
					}
					else
						Phi_new(i,j) = Phi_old(i,j);
				}
			}
			if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == i){
				log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(i,i));
				pos_Al0++;
			}
		}
		Phi_new(p-1,p-1) = Phi_old(p-1,p-1); //Last element
		if(pos_Al0 < sizeAl0 && A_l0[pos_Al0] == p-1){
			log_jacobian += build_jacobian_esponent(pos_Al0) * std::log(Phi_new(p-1,p-1));
			pos_Al0++;
		}
		log_element_proposal /= 2*this->sigma*this->sigma;
		log_element_proposal += static_cast<double>(L.size())*(0.5*utils::log_2pi + std::log(std::abs(this->sigma)));
		return std::make_tuple( 
				PrecisionType ( Phi_new, Kold_prior.get_shape(), Kold_prior.get_inv_scale(), Kold_prior.get_chol_invD() ),
				log_element_proposal, log_jacobian );
}

template<template <typename> class GraphStructure, typename T>
template< template <typename> class GG, typename TT, std::enable_if_t< internal_type_traits::isCompleteGraph<GG,TT>::value , TT> > //CompleteGraphs
std::tuple<typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType, double, double>
ReversibleJumpsMH<GraphStructure, T>::RJ_new( typename ReversibleJumpsMH<GraphStructure, T>::CompleteType const & Gnew,
										 	  typename ReversibleJumpsMH<GraphStructure, T>::PrecisionType& Kold_prior, MoveType Move, sample::GSL_RNG const & engine)
{
	using Graph = GraphStructure<T>;

	const unsigned int p(Kold_prior.get_matrix().rows());
	const std::pair<unsigned int, unsigned int>& changed_link = this->selected_link; //For lighter notation
	double log_element_proposal{0};
	//a) Fill new Phi
	MatRow Phi_new(MatRow::Zero(p,p));

	if(!Kold_prior.isFactorized){
		Kold_prior.compute_Chol();
	}

	MatRow Phi_old(Kold_prior.get_upper_Chol()); 

	bool found=false;

	//i = 0 case)
	Phi_new(0,0) = Phi_old(0,0);
	for(unsigned int j = 1; j < p; ++j){

		if(Gnew(0,j) == false){ //There is no link. Is it the removed one?
			Phi_new(0,j) = 0; 
			if(Move == MoveType::Remove && !found && std::make_pair((unsigned int)0,j) == changed_link){
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j))/(2*this->sigma*this->sigma);
			}

		}
		else{ //There is a link. Is it the new one?
			if(Move == MoveType::Add && !found && std::make_pair((unsigned int)0,j) == changed_link){
				Phi_new(0,j) = sample::rnorm()(engine, Phi_old(0,j), this->sigma); //The new element is a free element
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(0,j) - Phi_old(0,j))*(Phi_new(0,j) - Phi_old(0,j))/(2*this->sigma*this->sigma);
			}
			else
				Phi_new(0,j) = Phi_old(0,j);
		}
	}
	//i = 1 case)
	Phi_new(1,1) = Phi_old(1,1);
	for(unsigned int j = 2; j < p; ++j){

		if(Gnew(1,j) == false){ //There is no link. Is it the removed one?
			
			if(Move == MoveType::Remove && !found && std::make_pair((unsigned int)1,j) == changed_link){
				Phi_new(1,j) = - ( Phi_old(0,1)*Phi_old(0,j) )/Phi_old(1,1); 
				found = true;
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j))/(2*this->sigma*this->sigma);
			}
			else{
				Phi_new(1,j) = - ( Phi_new(0,1)*Phi_new(0,j) )/Phi_new(1,1); 
			}

		}
		else{ //There is a link. Is it the new one?

			if(Move == MoveType::Add && !found && std::make_pair((unsigned int)1,j) == changed_link){
				found = true;
				Phi_new(1,j) = sample::rnorm()(engine, Phi_old(1,j), this->sigma); //The new element is a free element
				log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
										(Phi_new(1,j) - Phi_old(1,j))*(Phi_new(1,j) - Phi_old(1,j))/(2*this->sigma*this->sigma);
			}
			else
				Phi_new(1,j) = Phi_old(1,j);
		}//changed_link è davvero una pair
	}
	//i > 1 case)
	for(unsigned int i = 2; i < p-1; ++i){

		Phi_new(i,i) = Phi_old(i,i);
		VecCol Psi_i(Phi_new.block(0,i,i,1));
		for(unsigned int j = i+1; j < p; ++j){
			if(Gnew(i,j) == false){ //There is no link. Is it the removed one?
				
				if(Move == MoveType::Remove && !found && std::make_pair(i,j) == changed_link){
					VecCol Psi_i_old(Phi_old.block(0,i,i,1));
					Phi_new(i,j) = - ( Psi_i_old.dot(VecCol (Phi_old.block(0,j,i,1))) )/Phi_old(i,i);
					found = true;
					log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
											(Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j))/(2*this->sigma*this->sigma);
				}
				else{
					Phi_new(i,j) = - ( Psi_i.dot(VecCol (Phi_new.block(0,j,i,1))) )/Phi_new(i,i);
				}

			}
			else{ //There is a link. Is it the new one?

				if(Move == MoveType::Add && !found && std::make_pair(i,j) == changed_link){ 
					found = true;
					Phi_new(i,j) = sample::rnorm()(engine, Phi_old(i,j), this->sigma); //The new element is a free element
					log_element_proposal = 	0.5*utils::log_2pi + std::log(std::abs(this->sigma)) + 
											(Phi_new(i,j) - Phi_old(i,j))*(Phi_new(i,j) - Phi_old(i,j))/(2*this->sigma*this->sigma);
				}
				else
					Phi_new(i,j) = Phi_old(i,j);
			}//changed_link è davvero una pair
		}
	}
	Phi_new(p-1,p-1) = Phi_old(p-1,p-1); //Last element

	//Return
	return std::make_tuple( 
			PrecisionType ( Phi_new, Kold_prior.get_shape(), Kold_prior.get_inv_scale(), Kold_prior.get_chol_invD() ),
			log_element_proposal, std::log(Phi_old(changed_link.first, changed_link.first)) );
}



template< template <typename> class GraphStructure , typename T >
typename ReversibleJumpsMH<GraphStructure, T>::ReturnType
ReversibleJumpsMH<GraphStructure, T>::operator()(MatCol const & data, unsigned int const & n, typename GGMTraits<GraphStructure, T>::Graph & Gold,  
										         double alpha, sample::GSL_RNG const & engine)
{

	using Graph  = GraphStructure<T>;
	using CompleteType = typename GGMTraits<GraphStructure, T>::CompleteType;

	sample::runif rand;
	double log_acceptance_ratio{0}; 
	bool isInf_new{false};
	bool isInf_old{false};

	//1) Propose new Graph
	auto [Gnew, log_GraphMove_proposal, mv_type] = this->propose_new_graph(Gold, alpha, engine) ;
	//2) Perform RJ according to the proposed move and graph
	CompleteType Gnew_complete(Gnew.completeview());
	CompleteType Gold_complete(Gold.completeview());
	auto [Knew_prior, log_rj_proposal, log_jacobian_mv] = this->RJ_new(Gnew_complete, this->Kprior, mv_type, engine) ;	
				//auto [Knew_prior, log_rj_proposal, log_jacobian_mv ] = this->RJ(Gnew_complete, this->Kprior, mv_type) ;	
	//3) Compute acceptance probability ratio
	PrecisionType& Kold_prior = this->Kprior; //lighter notation to avoid this-> every time
	double log_GraphPr_ratio(this->ptr_prior->log_ratio(Gnew, Gold));
	double const_old = 	Kold_prior.log_normalizing_constat(Gold_complete,MCiterPrior, engine);						  
	double const_new = 	Knew_prior.log_normalizing_constat(Gnew_complete,MCiterPrior, engine);	
	if(const_new < std::numeric_limits<double>::min()){
		isInf_new = true;
		//std::cout<<std::endl<<"Inf in GWish const prior new"<<std::endl;
	}
	if(const_old < std::numeric_limits<double>::min()){
		isInf_old = true;
		//std::cout<<std::endl<<"Inf in GWish const prior old"<<std::endl;
	}
	double log_GWishPrConst_ratio(const_old - const_new);					  
	auto TraceProd = [](MatRow const & A, MatCol const & B){
		double res{0};
		#pragma omp parallel for reduction(+:res)
		for(unsigned int i = 0; i < A.rows(); ++i)
			res += A.row(i)*B.col(i);
		return( -0.5*res );
	}; //This lambda function computes trace(A*B)

	
	
	//D+U is changing every iteration or not? if not, just factorize it once
	if(!this->data_factorized){
		this->D_plus_U = this->Kprior.get_inv_scale() + data;	
		this->chol_inv_DplusU = this->D_plus_U.llt().solve(MatCol::Identity(data.rows(),data.rows())).llt().matrixU();
		this->data_factorized = true;
	}
	double log_LL_GWishPr_ratio(  TraceProd( Knew_prior.get_matrix() - Kold_prior.get_matrix() , this->D_plus_U)  );
	if(mv_type == MoveType::Add)
		log_acceptance_ratio = log_GWishPrConst_ratio + log_GraphPr_ratio + log_GraphMove_proposal + log_LL_GWishPr_ratio + log_rj_proposal + log_jacobian_mv;
	else if(mv_type == MoveType::Remove)
		log_acceptance_ratio = log_GWishPrConst_ratio + log_GraphPr_ratio + log_GraphMove_proposal + log_LL_GWishPr_ratio - log_rj_proposal - log_jacobian_mv;

	double acceptance_ratio = std::min(1.0, std::exp(log_acceptance_ratio)); 
	if(isInf_old)
		acceptance_ratio = 1.0;
	else if(isInf_new)
		acceptance_ratio = 0.0;

	//4) Perform the move and return
	int accepted;
	if( rand(engine) < acceptance_ratio ){ //Accepted
		Gold = std::move(Gnew); //Move semantic is available
		accepted = 1;
	}
	else{ //Refused
		accepted = 0;
	}
	this->Kprior.set_matrix(Gold_complete, 
		utils::rgwish<CompleteSkeleton, T, utils::ScaleForm::CholUpper_InvScale, utils::MeanNorm>(Gold_complete, this->Kprior.get_shape() + n, this->chol_inv_DplusU , this->trGwishSampler, engine ) ); 
	this->Kprior.compute_Chol();
	return std::make_tuple(this->Kprior.get_matrix(), accepted);
}

#endif

