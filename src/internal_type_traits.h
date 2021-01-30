#ifndef __TYPETRAITS_HPP__
#define __TYPETRAITS_HPP__
#include "include_headers.h"

//Handcrafted type traits
namespace internal_type_traits
{

	/*Type_traits to check is a Graph is in Block form.
	  Usage example:
	  internal_type_traits::isBlockGraph<BlockGraph, int>::value --> true
	  internal_type_traits::isBlockGraph<CompleteView, bool>::value --> false
	*/

	template< template <typename> class GraphStructure, typename T >
	struct isBlockGraph
	{
		//Using is_base_of_v< , > all specializations of BlockGraphBaseCRTP are automatically inserted as BlockGraph
		static constexpr bool value { std::is_base_of_v< BlockGraphBaseCRTP<GraphStructure<T> ,T>, GraphStructure<T> >  ||
									  std::is_same_v<GraphStructure<T>, BlockGraphDyn<T> > || std::is_same_v<GraphStructure<T>, BlockGraphAdjDyn<T> >
									};
	};
	//--------------------------------------------------------------------
	/* Type_traits to check is a Graph is in Complete form
	  --> isCompleteGraph is simply wrapping std::is_same_v
	  --> my_isCompleteGraph is the one written by me and of course it does not work as well as the other. The problem is in RJ.h and DRJ.h when CompleteSkeleton has to be checked.
	  	  the problem probably is that CompleteSkeleton is a template alias and not a template parameter. isCompleteGraph instead is checking the concrete type and does not suffer from this problem
	  	  Meno romantico ma più efficace, va bene anche cosi dai. 
	  Usage example:
	  internal_type_traits::isCompleteGraph<BlockGraph,T>::value --> false
	  internal_type_traits::isCompleteGraph<CompleteView,T>::value --> true

	*/
	template< template <typename> class GraphStructure>
	struct my_isCompleteGraph : public std::false_type {};
	template<>
	struct my_isCompleteGraph<GraphType> : public std::true_type{};
	template<>
	struct my_isCompleteGraph<CompleteViewAdj> : public std::true_type{};
	template<>
	struct my_isCompleteGraph<CompleteView> : public std::true_type{};
	template<>
	struct my_isCompleteGraph<CompleteViewAdjDyn> : public std::true_type{};
	template<>
	struct my_isCompleteGraph<CompleteViewDyn> : public std::true_type{};

	template< template <typename> class GraphStructure, typename T >
	struct isCompleteGraph
	{
		static constexpr bool value {std::is_same_v<GraphStructure<T>, GraphType<T> >       || 
									 std::is_same_v<GraphStructure<T>, CompleteView<T> >    || std::is_same_v<GraphStructure<T>, CompleteViewAdj<T> > ||
									 std::is_same_v<GraphStructure<T>, CompleteViewDyn<T> > || std::is_same_v<GraphStructure<T>, CompleteViewAdjDyn<T> >
									};
	};
	//--------------------------------------------------------------------

	/* Type trait to be used to define the correct Complete Type given a generic Graph.
		Two version are provided:
		1) Complete_type, takes two arguments as usual, the first parameter is a template itself and the second is the type of the stored values.
		   -> this version provides two typedefs, CompleteSkeleton that is the template of the Graph in complete form and CompleteType that is the concrete type, i.e CompleteSkeleton<T>.
		2) Complete_skeleton works exactly as the previous version but taking only the first parameter and returning only the template of the graph. It is provided only because the role of parameter
		      T may result confusing and often can be avoided.

		Note that within a typedef it is not possible to define only the template skeleton. A complete type has always to be defined. Template aliasing has to be used in order to work around this problem.
		Usage with template type (BlockGraph - GraphType ecc..):
		1)
		template<typename S>
		using CompleteSkeleton = typename internal_type_traits::Complete_type<BlockGraph, unsigned int>::CompleteSkeleton<S>;
		using Complete         = typename internal_type_traits::Complete_type<BlockGraph, unsigned int>::CompleteType;
		--> Note the used of template aliasing when definig the skeleton. CompleteSkeleton is now totally independed from the choice of T (T = unsigned int in this example) and can be used with every type,
		i.e CompleteSkeleton<bool> or CompleteSkeleton<unsigned int>. 
		Complete is instead bounded to the choice of T. That is why when only the skeleton is needed it may be more convinient to use the second version which hide the choice of T.
		2)
		template<typename S>
		using CompleteSkeleton = typename internal_type_traits::Complete_skeleton<GraphType>::CompleteSkeleton<S>;
		
		This second example shows how to use this type trait when the Graph type to be interrogated is itself a template parameter. This is the situation where it is actually useful.
		Usage with template parameter:
		3)
		template<template <typename> class GraphStructure = GraphType, typename T = unsigned int>
		struct Example{

			template<typename S>
			using CompleteSkeleton  = typename internal_type_traits::Complete_skeleton<GraphStructure>::template CompleteSkeleton<S>;
			using CompleteType 		= typename internal_type_traits::Complete_type<GraphStructure, T>::CompleteType;
			
			template<typename S2>
			using CompleteSkeleton2 = typename internal_type_traits::Complete_type<GraphStructure, T>::template CompleteSkeleton<S2>;
		};
		Note the use of template keyword. 
	*/
	template< template <typename> class GraphStructure, typename T>
	struct Complete_type
	{
		private:
		// Result0 = BlockGraph vs BlockGraphAdj
		template <typename Partial0>
		using CompleteBlockType0 = std::conditional_t< 	std::is_same_v<GraphStructure<T>,  BlockGraph<T> >, //static condition
														CompleteView<Partial0> ,  //type defined if true   --> CompleteView is the complete type of BlockGraph
														CompleteViewAdj<Partial0> //type defined if false  --> CompleteViewAdj is the complete type of BlockGraphAdj
													 > ;
		// Result1 = Result0 vs BlockGraphDyn
		template <typename Partial1>
		using CompleteBlockType1 = std::conditional_t< 	std::is_same_v<GraphStructure<T>,  BlockGraphDyn<T> >, //static condition
														CompleteViewDyn<Partial1> ,    //type defined if true  --> CompleteViewDyn is the complete type of BlockGraphDyn
														CompleteBlockType0<Partial1>   //type defined if false --> keep the one defined above
													 > ;	
		//Result2 = Result1 vs BlockGraphAdjDyn											 
		template <typename Partial2>
		using CompleteBlockType2 = std::conditional_t< 	std::is_same_v<GraphStructure<T>,  BlockGraphAdjDyn<T> >, //static condition
														CompleteViewAdjDyn<Partial2> ,  //type defined if true  --> BlockGraphAdjDyn is the complete type of BlockGraphAdjDyn
														CompleteBlockType1<Partial2>    //type defined if false --> keep the one defined above
													 > ;														 
		public:
		//Result = Result2 vs GraphType	
		template <typename Final>
		using CompleteSkeleton  = std::conditional_t< 	std::is_same_v<GraphStructure<T> , GraphType<T> >, //static condition
														GraphType<Final>, 		  //type defined if true  --> GraphType is the complete type of itself
														CompleteBlockType2<Final> //type defined if false --> keep the one defined above 
													> ;
		using CompleteType 	= CompleteSkeleton<T>;
	};

	template< template <typename> class GraphStructure >
	struct Complete_skeleton
	{
		private:
		using T = bool;
		// Result0 = BlockGraph vs BlockGraphAdj
		template <typename Partial0>
		using CompleteBlockType0 = std::conditional_t< 	std::is_same_v<GraphStructure<T>,  BlockGraph<T> >, //static condition
														CompleteView<Partial0> ,  //type defined if true   --> CompleteView is the complete type of BlockGraph
														CompleteViewAdj<Partial0> //type defined if false  --> CompleteViewAdj is the complete type of BlockGraphAdj
													 > ;											 
		// Result1 = Result0 vs BlockGraphDyn
		template <typename Partial1>
		using CompleteBlockType1 = std::conditional_t< 	std::is_same_v<GraphStructure<T>,  BlockGraphDyn<T> >, //static condition
														CompleteViewDyn<Partial1> ,    //type defined if true  --> CompleteViewDyn is the complete type of BlockGraphDyn
														CompleteBlockType0<Partial1>   //type defined if false --> keep the one defined above
													 > ;	
		//Result2 = Result1 vs BlockGraphAdjDyn											 
		template <typename Partial2>
		using CompleteBlockType2 = std::conditional_t< 	std::is_same_v<GraphStructure<T>,  BlockGraphAdjDyn<T> >, //static condition
														CompleteViewAdjDyn<Partial2> ,  //type defined if true  --> BlockGraphAdjDyn is the complete type of BlockGraphAdjDyn
														CompleteBlockType1<Partial2>    //type defined if false --> keep the one defined above
													 > ;														 
		public:
		//Result = Result2 vs GraphType	
		template <typename Final>
		using CompleteSkeleton  = std::conditional_t< 	std::is_same_v<GraphStructure<T> , GraphType<T> >, //static condition
														GraphType<Final>, 		  //type defined if true  --> GraphType is the complete type of itself
														CompleteBlockType2<Final> //type defined if false --> keep the one defined above 
													> ;
	};
	
	
}
#endif
