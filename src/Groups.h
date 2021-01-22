#ifndef __GROUPS_H__
#define __GROUPS_H__

#ifndef NORCPP
  #define STRICT_R_HEADERS
  #include <Rcpp.h>
#endif

#include "include_headers.h"

 //#ifdef PARALLELEXEC
 //#  if defined(__GNUC__) && (__GNUC__ >= 9)
 //#    include <execution>
 //#	 warning "Using parallel implementation"
 //#  else
 //#    undef PARALLELEXEC
 //#  endif
 //#endif


struct GroupsTraits{
  using InnerContainer 		       	= std::vector<unsigned int>;
  using InnerContainer_Iterator 	= InnerContainer::iterator;
  using InnerContainer_Citerator 	= InnerContainer::const_iterator;
  using Container 			         	= std::vector<InnerContainer>;
  using Container_Iterator 	    	= Container::iterator;
  using Container_Citerator   		= Container::const_iterator;
  using IdxType 					        = InnerContainer::size_type;
  using IdxMap                    = std::vector<unsigned int> ;
  //using IdxMap                    = std::map<unsigned int, unsigned int>;
  //using IdxMap                    = std::unordered_map<unsigned int, unsigned int>;
  //Map could be slightly better. To use std::vector, uncomment up here, in Groups.cpp, createMapIdx() -> uncomment the resize line and 
  //BlockGraph.h -> find_group_idx(IdxType const & i) uncomment the return line and comment the one below
};

//Il container dei Groups è una sorta di rho function. Il contenitore esterno rappresenta i nodi del grafo a blocchi, quindi (*this)[i] sono tutti i nodi del 
//grafo grande associati ad i. 
// map_of_indeces -> invece rappresenta l'operazione inversa, è indicizzata rispetto ai nodi del grafo completo e mappa nei nodi di quello a blocchi.
// 

class Groups : public GroupsTraits,std::vector<std::vector<unsigned int>>
{
public:
  //Constructors
  Groups(unsigned int const & _N);
  Groups(unsigned int const & _M, unsigned int const & _p);
  //Groups(Container const & _C);
  Groups(Container const & _C);
  #ifdef STRICT_R_HEADERS
   Groups(Rcpp::List const & _L);
  #endif
 
  Groups()=default;
  //Getters
  unsigned int get_n_groups() const{
    return this->size();
  }
  unsigned int get_n_singleton() const{
    return std::count_if(this->cbegin(), this->cend(), [](InnerContainer const & v){return (v.size() == 1);} );
  }
  unsigned int get_n_elements()const{
    unsigned int res = 0;
    for(Container_Citerator it = this->cbegin(); it != this->cend(); ++it) 
      res += it->size();

    return res;
  }
  unsigned int get_group_size(IdxType const & i)const{
    if(i > this->get_n_groups())
      throw std::runtime_error("Invalid group index request");
    return (*this)[i].size();
  }
  InnerContainer get_pos_singleton() const;

  std::vector<unsigned int> get_group(IdxType const & i) const{
    if(i > this->get_n_groups())
      throw std::runtime_error("Invalid group index request");
    return (*this)[i];
  }
  unsigned int get_possible_block_links()const{
    return (0.5*this->size()*(this->size()-1) + this->size() - this->get_n_singleton());
  }
  //Setters -> not used and dangerous. Watch out for createMapIdx();
  //inline void set_n_groups(unsigned int const & N){
    //this->resize(N);
  //}
  //void add_group(InnerContainer const & v);

  //Find element
  unsigned int find(IdxType const & i)const;
  InnerContainer find_and_get(IdxType const & i) const;
  void createMapIdx();
  //Streaming operator
  friend std::ostream & operator<<(std::ostream &str, Groups & gr);
  IdxMap map_of_indeces;
private:
  
};

#endif
