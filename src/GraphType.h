#ifndef __GRAPHTYPETEMPLATE_H__
#define __GRAPHTYPETEMPLATE_H__

//#include <Rcpp.h>

#include "include_headers.h"

template<class T=unsigned int>
struct GraphTypeTraits{
  using value_type    = T;
  using Adj  		      = std::vector<T>;
  using InnerData     = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IdxType	      = std::size_t;
  using Neighbourhood = std::map<unsigned int, std::vector<unsigned int> >;
  using ColType       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using RowType       = Eigen::Matrix<T, 1, Eigen::Dynamic>;
};


template<class T=unsigned int>
class GraphType : public GraphTypeTraits<T>{
public:
  using value_type    = typename GraphTypeTraits<T>::value_type;
  using Adj 		    	= typename GraphTypeTraits<T>::Adj;
  using IdxType		    = typename GraphTypeTraits<T>::IdxType;
  using InnerData     = typename GraphTypeTraits<T>::InnerData;
  using Neighbourhood = typename GraphTypeTraits<T>::Neighbourhood;

  //Constructors
  //GraphType()=default;
  GraphType(Adj const & _A){
    data = InnerData::Identity( 0.5 * ( 1 + std::sqrt(1 + 8*_A.size() )),  0.5 * ( 1 + std::sqrt(1 + 8*_A.size() )));
    IdxType pos{0};
    for(IdxType i = 0; i < data.rows() - 1; ++i)
      for(IdxType j = i+1; j < data.cols(); ++j){
        data(i,j) = _A[pos++];
      }
    this->find_neighbours();
  };
  GraphType(IdxType const & _N): data(InnerData::Identity(_N, _N)){
    this->find_neighbours();
  }; //takes only the number of nodes

  //GraphType(GraphType const & _Gr); default is ok
  //GraphType(GraphType&& _Gr); default is ok

  GraphType(InnerData const & _M): data(_M){
    if(data.rows() != data.cols())
      throw std::runtime_error("Matrix insereted as graph is not squared");
    data.diagonal().array()=1;
    this->find_neighbours();
  }
  GraphType(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> const & _M): data(_M){
    if(data.rows() != data.cols())
      throw std::runtime_error("Matrix insereted as graph is not squared");
    data.diagonal().array()=1;
    this->find_neighbours();
  }
  //Getters
  inline InnerData get_graph() const{
    return data;
  }
  inline Adj get_adj_list()const{
    Adj res(this->get_possible_links());
    int add{0};
    for(unsigned int i = 0; i < this->get_size()-1; ++i)
      for(unsigned int j = i+1; j < this->get_size(); ++j)
        res[add++] = data(i,j);
    return res;  
  }
  inline unsigned int get_size() const{
    return data.cols();
  }
  inline unsigned int get_complete_size() const{ //for coherence and symmetry with BlockGraphs
    return data.cols();
  }
  inline Neighbourhood get_nbd() const{
    return this->neighbours;
  }
  inline std::vector<unsigned int> get_nbd(IdxType const & i)const{
    if(i >= neighbours.size())
      throw std::runtime_error("get_nbd(i) : index exceeds matrix dimension");
    else
      return this->neighbours.find(i)->second;
  }
  unsigned int get_n_links() const{
    //Compute number of links
    unsigned int n_links = 0;
    for(unsigned int i = 0; i < data.rows() - 1; ++i)
      for(unsigned int j = i+1; j < data.cols(); ++j)
        if(data(i,j) == true){
          n_links++;
        }

    return n_links;    
  }

  inline unsigned int get_possible_links() const{
    return 0.5*data.cols()*(data.cols()-1);
  }
  //Set the entire graph
  void set_graph(Adj const & A){
    data = InnerData::Identity( 0.5 * ( 1 + std::sqrt(1 + 8*A.size() )),  0.5 * ( 1 + std::sqrt(1 + 8*A.size() )));
    IdxType pos{0};
    for(IdxType i = 0; i < data.rows() - 1; ++i)
      for(IdxType j = i+1; j < data.cols(); ++j)
        data(i,j) = A[pos++];

    neighbours.clear();
    find_neighbours();  
  }
  void set_graph(Adj&& A){
    data = InnerData::Identity( 0.5 * ( 1 + std::sqrt(1 + 8*A.size() )),  0.5 * ( 1 + std::sqrt(1 + 8*A.size() )));
    IdxType pos{0};
    for(IdxType i = 0; i < data.rows() - 1; ++i)
      for(IdxType j = i+1; j < data.cols(); ++j)
        data(i,j) = A[pos++];
    A.clear();
    neighbours.clear();
    find_neighbours();  
  }
  void set_empty_graph(){
    Adj adj_empty(0.5*this->get_size()*(this->get_size()-1), false);
    this->set_graph(adj_empty);
  }
  void fillRandom(double sparsity = 0.5, unsigned int seed = 0);
  //Set-Remove single link
  inline void add_link(IdxType const & i, IdxType const & j){
    data(i,j) = true;
  }
  inline void remove_link(IdxType const & i, IdxType const & j){
    data(i,j) = false;
  }
  //Converters
  std::pair<unsigned int, unsigned int> pos_to_ij(IdxType const & pos) const;
  inline GraphType& completeview(){
    return (*this);
  }
  // Operators
  T operator()(IdxType const & i, IdxType const & j)const{
    return (i<=j) ? (data(i,j)) : (data(j,i));
  }
  T& operator()(IdxType const & i, IdxType const & j){
    return (i<=j) ? (data(i,j)) : (data(j,i));
  }

  friend std::ostream & operator<<(std::ostream & str, const GraphType & G) //introverse friend
  {
       for(IdxType i = 0; i < G.data.rows(); ++i){
         for(IdxType j = 0; j < G.data.cols(); ++j){
           if(i == j)
              str<<1<<" ";
            else if(i < j)
              str<<G.data(i,j)<<" ";
            else
              str<<G.data(j,i)<<" ";
          }
          str<<std::endl;
        }
        return str;
  }

  IdxType compute_diagonal_position(IdxType const & i) const{
    if(i == 0)
      return 0;
    IdxType res = 0;
    #pragma omp parallel for reduction (+ : res)
    for(IdxType k = 1; k <= i ; ++k){
      res += (data.cols() - k);
    }
    return res-1;
  }
private:
  InnerData data;
  Neighbourhood neighbours;
  void find_neighbours();
};






//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


template<class T>
void GraphType<T>::fillRandom(double sparsity, unsigned int seed){

  if(sparsity > 1.0){
    std::cerr<<"Sparsity larger then 1, set to 0.5";
    sparsity = 0.5;
  }
  if(seed==0){
   seed = static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count());
   std::seed_seq seq = {seed}; //seed provived here has to be random. Than std::seed_seq adds entropy becasuse steady_clock is not sufficientyl widespread
   std::vector<unsigned int> seeds(1);
   seq.generate(seeds.begin(), seeds.end());
   seed = seeds[0];
  }
  // I use the standard engine for random numbers
  std::default_random_engine engine(seed);
  // A uniform distribution between 0 and 1
  std::uniform_real_distribution<double> rand(0.,1.);
  //fill the graph
  for(unsigned int i = 0; i < data.rows()-1; ++i)
    for(unsigned int j = i+1; j < data.cols(); ++j)
      (rand(engine) < sparsity) ? data(i,j)=true : data(i,j)=false;

  neighbours.clear();
  find_neighbours();   
}

template<class T>
void GraphType<T>::find_neighbours(){
  using IdxType = typename GraphType<T>::IdxType;

  for(IdxType i=0; i < this->get_size(); ++i){
    std::set<unsigned int> temp;
    for(IdxType j = 0; j < this->get_size(); ++j){
      if((*this)(i,j) == true && i != j)
        temp.insert(j);
        //neighbours[i].emplace_back(j);
    }
    std::vector<unsigned int> my_nbds(temp.begin(), temp.end());
    neighbours.insert(std::make_pair(i, my_nbds));
  }
}

template<class T>
std::pair<unsigned int, unsigned int> 
GraphType<T>::pos_to_ij(typename GraphType<T>::IdxType const & pos) const{

  using IdxType   = typename GraphType<T>::IdxType;
  if(pos > this->get_possible_links())
    throw std::runtime_error("Requested position exceeds matrix dimension");
  if(pos == 0)
    return static_cast<std::pair<unsigned int, unsigned int> > ( std::make_pair(0,1) );
  IdxType last(this->get_size() - 1);
  IdxType cde_old{0}; 
  IdxType cde_new{0};
  for(IdxType h = 0; h < last; ++h){
    cde_new = compute_diagonal_position(h+1);
    if(pos == cde_new)
      return static_cast<std::pair<unsigned int, unsigned int> >( std::make_pair(h,last) );
    if(pos > cde_old && pos < cde_new)
      return (h == 0) ? static_cast<std::pair<unsigned int, unsigned int> >(std::make_pair(h, pos - cde_old + h + 1)) : 
                        static_cast<std::pair<unsigned int, unsigned int> >(std::make_pair(h, pos - cde_old + h));
    cde_old = cde_new; //Continues
  }
  return static_cast<std::pair<unsigned int, unsigned int> >(std::make_pair(last,last));  
}





#endif
