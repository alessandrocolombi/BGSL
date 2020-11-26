#include "Groups.h"

using InnerContainer 			      = GroupsTraits::InnerContainer;
using InnerContainer_Iterator   = GroupsTraits::InnerContainer_Iterator;
using InnerContainer_Citerator 	= GroupsTraits::InnerContainer_Citerator;
using Container 				        = GroupsTraits::Container;
using Container_Iterator 		    = GroupsTraits::Container_Iterator;
using Container_Citerator 		  = GroupsTraits::Container_Citerator;
using IdxType 					        = GroupsTraits::IdxType;



Groups::Groups(unsigned int const & _N){
  this->resize(_N);
}
Groups::Groups(unsigned int const & _M, unsigned int const & _p){
  if(_p < _M)
    throw std::runtime_error("Number of groups greater then number of nodes");
  this->resize(_M);

  unsigned int nel = _p/_M;
  unsigned int idx{0};

  for(IdxType i = 0; i <= _M-2; ++i){
    (*this)[i].resize(nel);
    for(IdxType j = 0; j < nel; ++j)
      (*this)[i][j] = idx++;
  }
  for(IdxType j = 0; j < nel + _p%_M ; ++j){
    (*this)[_M-1].resize(nel + _p%_M );
    (*this)[_M-1][j] = idx++;
  }
  this->createMapIdx();
}
Groups::Groups(Container const & C){
  this->resize(C.size());
  for(IdxType i = 0; i < this->size(); ++i){
    if( std::is_sorted(C[i].cbegin(), C[i].cend()) )
      (*this)[i] = C[i];
    else{
      InnerContainer sorted(C[i].cbegin(), C[i].cend());
      std::sort(sorted.begin(), sorted.end());
      (*this)[i] = sorted;
    }
  }
  this->createMapIdx();
}

#ifndef NORCPP
  Groups::Groups(Rcpp::List const & _L){
    this->resize(_L.size());
    for(IdxType i = 0; i < this->size(); ++i){
      InnerContainer v = _L[i];
      if( std::is_sorted(v.cbegin(), v.cend()) )
        (*this)[i] = v;
      else{
        InnerContainer sorted(v.cbegin(), v.cend());
        std::sort(sorted.begin(), sorted.end());
        (*this)[i] = sorted;
      }
    }
    this->createMapIdx();
  }
#endif


InnerContainer Groups::get_pos_singleton() const{

  InnerContainer res;
  res.resize(this->get_n_singleton());
  IdxType idx{0};
  for(Container_Citerator it = this->cbegin(); it != this->cend(); ++it){
    if(it->size() == 1){
      res[idx] = it - this->cbegin();
      idx++;
    }
  }

  return res;
}

//void Groups::add_group(InnerContainer const & v){
//
  //for(IdxType i = 0; i < this->size(); ++i){
    //if((*this)[i].size() == 0){
      //(*this)[i] = v;
      //break;
    //}
  //}
//}

unsigned int Groups::find(IdxType const & i)const {
  unsigned int res{0};
  for(Container_Citerator it_est = this->cbegin(); it_est != this->cend(); ++it_est){
    if(std::binary_search(it_est->cbegin(), it_est->cend(), i))
      return res;
    else
      res++;
    /*
      InnerContainer_Citerator it_inner = std::find(it_est->cbegin(), it_est->cend(), i); 
    if(it_inner != it_est->cend())
      return res;
    res++;
    */
  }
  throw std::runtime_error("Index not found");
}
InnerContainer Groups::find_and_get(IdxType const & i) const{
  for(Container_Citerator it_est = this->cbegin(); it_est != this->cend(); ++it_est){

    if(std::binary_search(it_est->cbegin(), it_est->cend(), i))
      return *it_est;
  }
  throw std::runtime_error("Index not found");
}

void Groups::createMapIdx(){
  //map_of_indeces.resize(this->get_n_elements());
  for(unsigned int i = 0; i < this->size(); ++i)
    for(unsigned int j = 0; j < (*this)[i].size(); ++j)
      map_of_indeces[(*this)[i][j]] = i;  
}


std::ostream & operator<<(std::ostream & str, Groups & gr){

  for(Container_Citerator it = gr.cbegin(); it != gr.cend(); ++it){
    str<<it - gr.cbegin()<<" -> ";
    for(InnerContainer_Citerator inner_it = it->cbegin(); inner_it != it->cend(); ++inner_it){
      str<<*inner_it<<"  ";
    }
    str<<std::endl;
  }
  return str;
}
