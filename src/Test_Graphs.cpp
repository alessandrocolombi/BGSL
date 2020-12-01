#ifndef __TESTGRAPH_HPP__
#define __TESTGRAPH_HPP__


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppParallel)]]
#define STRICT_R_HEADERS
#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "include_headers.h"
#include "include_graphs.h"

using namespace std;
using MyEigenMat 	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatRow        = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatCol 		= Eigen::MatrixXd;
using Neighbourhood = BlockGraphBase<bool>::Neighbourhood;
using ColType       = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using RowType       = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using VecColB		= Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>;

//' Function for testing properties of Graph classes
//'
//' @param No parameters are required
//' @export
// [[Rcpp::export]]
void GraphTest() {
   //auto& str = std::cout;
	auto& str = Rcpp::Rcout;

   	unsigned int  p = 7;
   	unsigned int  M = 3;
   	Groups Gr_tanti(M, p);
   	Groups Gr({ {0,1,2}, {3}, {4,5}, {6} });
   	Groups Gr_nosing({ {0,1}, {2,3}, {4,5,6} });

   	std::vector<unsigned int> adj{true, true, true, true, false, true, true, false};
   	std::shared_ptr<const Groups> ptr_gruppi(std::make_shared<const Groups>(Gr));
   	std::shared_ptr<const Groups> ptr_gruppi_nosin(std::make_shared<const Groups>(Gr_nosing));
   	std::shared_ptr<const Groups> ptr_gruppi_tanti(std::make_shared<const Groups>(std::move(Gr_tanti)));

   	str<<"Ho creato uno shared pointer con move operator"<<std::endl;
   	str<<"Chi in quello vecchio? size = "<<Gr_tanti.get_n_groups()<<std::endl;
   	str<<"Chi in quello nuovo? size = "<<ptr_gruppi_tanti->get_n_groups()<<std::endl;

   	auto start = std::chrono::high_resolution_clock::now();
   	BlockGraph BG(adj, ptr_gruppi);
   	std::shared_ptr<const CompleteView<unsigned int> > ptr_cv_BG = std::make_shared<const CompleteView<unsigned int> >(BG);
   	auto stop = std::chrono::high_resolution_clock::now();
   	std::chrono::duration<double, std::milli> timer = stop - start;
   	str << "Time:  " << timer.count()<<" ms"<< endl;

   	start = std::chrono::high_resolution_clock::now();
   	BlockGraphAdj BAD(adj, ptr_gruppi);
   	std::shared_ptr<const CompleteViewAdj<unsigned int> > ptr_cv_BAD = std::make_shared<const CompleteViewAdj<unsigned int> >(BAD);
   	stop = std::chrono::high_resolution_clock::now();
   	timer = stop - start;
   	str << "Time:  " << timer.count()<<" ms"<< endl;
   	//Ne creo uno anche bool
   	std::vector<bool>adj_bool{true, true, true, true, false, true, true, false};
   	BlockGraph<bool> BG_bool(adj_bool, ptr_gruppi);

   	//Completi
   	CompleteView CV(BG);
   	CompleteViewAdj CVA(BAD);
   	GraphType C(p);
   	GraphType<bool> C_bool(p);

   	start = std::chrono::high_resolution_clock::now();
   	C.fillRandom(0.5, 1234);
   	C_bool.fillRandom(0.5, 1234);
   	stop = std::chrono::high_resolution_clock::now();
   	std::shared_ptr<const GraphType<unsigned int> > ptr_cv_C = std::make_shared<const GraphType<unsigned int> >(C);
   	timer = stop - start;
   	str << "Time:  " << timer.count()<<" ms"<< endl;
   	str<<"Gruppi = "<<std::endl<<Gr<<std::endl;
   	str<<"Blocchi = "<<std::endl<<BG<<std::endl;
   	str<<"Blocch bool:"<<std::endl<<BG_bool<<std::endl;
   	str<<"CompleteView (BG)= "<<std::endl<<CV<<std::endl;
   	str<<"CompleteView (BAD)= "<<std::endl<<CVA<<std::endl;
   	str<<"Completo = "<<std::endl<<C<<std::endl;
   	str<<"ADJ = "<<std::endl<<BAD<<std::endl;
   	str<<std::endl;

   	str<<"Cerco il nbd"<<std::endl;
   	Neighbourhood nbd_CV(CV.get_nbd());
   	for(auto const & mappa : nbd_CV){
   		str<<mappa.first<<"->";
   		for(auto const & idx : mappa.second)
   			str<<idx<<", ";
   	str<<std::endl;
   	}
   	str<<"From Complete"<<std::endl;
   	Neighbourhood nbd_C(C.get_nbd());
   	for(auto const & mappa : nbd_C){
   		str<<mappa.first<<"->";
   		for(auto const & idx : mappa.second)
   			str<<idx<<", ";
   	str<<std::endl;
   	}
   	str<<"----------------------------------------------------------------------"<<std::endl;
   	str<<"Provo le nuove features"<<std::endl;

   	CompleteView CV_nuova(BG.completeview());
   	CompleteViewAdj CVA_nuova(BAD.completeview());
   	GraphType CVC_nuova(C.completeview());
   	str<<"dimensioni"<<std::endl;
   	str<<CV_nuova.get_size()<<std::endl;
   	str<<CVA_nuova.get_size()<<std::endl;
   	str<<CVC_nuova.get_size()<<std::endl;

   	BlockGraph BG_empty(ptr_gruppi);
   	BG_empty.set_empty_graph();
   	str<<"BG_empty"<<endl<<BG_empty<<std::endl;
   	str<<"n links = "<<BG_empty.get_n_links()<<std::endl;
   	str<<"n block links = "<<BG_empty.get_n_block_links()<<std::endl;
   	str<<"Singletons: "<<std::endl<<BG_empty.get_row_with_singleton()<<std::endl;


   	BlockGraph BG_empty_nosin(ptr_gruppi_nosin);
   	BG_empty_nosin.set_empty_graph();
   	str<<"BG_empty_nosin"<<endl<<BG_empty_nosin<<std::endl;
   	str<<"n links = "<<BG_empty_nosin.get_n_links()<<std::endl;
   	str<<"n block links = "<<BG_empty_nosin.get_n_block_links()<<std::endl;
   	str<<"Singletons: "<<std::endl<<BG_empty_nosin.get_row_with_singleton()<<std::endl;


   	BlockGraphAdj BAD_empty(ptr_gruppi);
   	BAD_empty.set_empty_graph();
   	str<<"BAD_empty"<<endl<<BAD_empty<<std::endl;
   	str<<"n links = "<<BAD_empty.get_n_links()<<std::endl;
   	str<<"n block links = "<<BAD_empty.get_n_block_links()<<std::endl;

   	GraphType C_empty(p);
   	C_empty.set_empty_graph();
   	str<<"C_empty"<<endl<<C_empty<<std::endl;
   	str<<"n links = "<<C_empty.get_n_links()<<std::endl;
   	str<<std::endl;



   	str<<"Vettore dei singleton"<<std::endl;
   	std::vector<unsigned int> sing(BG_empty.get_pos_singleton());
   	std::vector<unsigned int> sing2(BAD_empty.get_pos_singleton());


   	str<<"----------------------------------------------------------------------"<<std::endl;
   	str<<"Provo copy - assignment operator dei grafi"<<std::endl;
   	BlockGraph BG_copycons(BG);
   	BlockGraph BG_copy(ptr_gruppi);
   	BG_copy = BG;
   	str<<"BG_copycons"<<endl<<BG_copycons<<std::endl;
   	str<<"n links = "<<BG_copycons.get_n_links()<<std::endl;
   	str<<"n block links = "<<BG_copycons.get_n_block_links()<<std::endl;
   	str<<"BG_copy"<<endl<<BG_copy<<std::endl;
   	str<<"n links = "<<BG_copy.get_n_links()<<std::endl;
   	str<<"n block links = "<<BG_copy.get_n_block_links()<<std::endl;

   	GraphType C_copycons(C);
   	GraphType C_copy(p);
   	C_copy = C;
   	str<<"C_copycons"<<endl<<C_copycons<<std::endl;
   	str<<"n links = "<<C_copycons.get_n_links()<<std::endl;
   	str<<"C_copy"<<endl<<C_copy<<std::endl;
   	str<<"n links = "<<C_copy.get_n_links()<<std::endl;
   	str<<std::endl;

   	str<<"----------------------------------------------------------------------"<<std::endl;
   	str<<"Provo move operators dei grafi"<<std::endl;
   	BlockGraph BG_movecons(std::move(BG_copycons));
   	BlockGraph BG_move(ptr_gruppi);
   	BG_move = std::move(BG_copy);
   	str<<"BG_movecons"<<endl<<BG_movecons<<std::endl;
   	str<<"n links = "<<BG_movecons.get_n_links()<<std::endl;
   	str<<"n block links = "<<BG_movecons.get_n_block_links()<<std::endl;
   	str<<"BG_move"<<endl<<BG_move<<std::endl;
   	str<<"n links = "<<BG_move.get_n_links()<<std::endl;
   	str<<"n block links = "<<BG_move.get_n_block_links()<<std::endl;
   	str<<"BG_copycons size = "<<BG_copycons.get_size()<<std::endl;
   	str<<"BG_copy size = "<<BG_copy.get_size()<<std::endl;

   	GraphType C_movecons(std::move(C_copycons)); //questa è stata l'unica cosa mossa per davvero
   	GraphType C_move(p);
   	C_move = std::move(C_copy);
   	str<<"C_movecons"<<endl<<C_movecons<<std::endl;
   	str<<"n links = "<<C_movecons.get_n_links()<<std::endl;
   	str<<"C_move"<<endl<<C_move<<std::endl;
   	str<<"n links = "<<C_move.get_n_links()<<std::endl;
   	str<<"C_copycons size = "<<C_copycons.get_size()<<std::endl;
   	str<<"C_copy size = "<<C_copy.get_size()<<std::endl;
   	str<<std::endl;

   	str<<"----------------------------------------------------------------------"<<std::endl;
   	Groups Gr_new({ {0}, {1}, {2,3,4}, {5}, {6} });

   	std::shared_ptr<const Groups> ptr_gruppi_new(std::make_shared<const Groups>(Gr_new));

   	BlockGraph BG_new(ptr_gruppi_new);

   	str<<"Provo pos to if (Blocchi)"<<std::endl;
   	str<<"get_possible_block_links = "<<BG.get_possible_block_links()<<std::endl;
   	for(auto h = 0; h < BG.get_possible_block_links(); ++h){
   		std::pair<unsigned int, unsigned int> ij=BG.pos_to_ij(h);
   		str<<"pos = "<<h<<" -> ("<<ij.first<<", "<<ij.second<<")"<<std::endl;
   	}

   	str<<"Provo pos to if (Blocchi no singleton)"<<std::endl;
   	str<<"get_possible_block_links = "<<BG_empty_nosin.get_possible_block_links()<<std::endl;
   	for(auto h = 0; h < BG_empty_nosin.get_possible_block_links(); ++h){
   		std::pair<unsigned int, unsigned int> ij=BG_empty_nosin.pos_to_ij(h);
   		str<<"pos = "<<h<<" -> ("<<ij.first<<", "<<ij.second<<")"<<std::endl;
   	}

   	str<<"Provo pos to if (Blocchi new)"<<std::endl;
   	str<<"get_possible_block_links = "<<BG_new.get_possible_block_links()<<std::endl;
   	for(auto h = 0; h < BG_new.get_possible_block_links(); ++h){
   		std::pair<unsigned int, unsigned int> ij=BG_new.pos_to_ij(h);
   		str<<"pos = "<<h<<" -> ("<<ij.first<<", "<<ij.second<<")"<<std::endl;
   	}

   	str<<"Provo pos to ij (Completo)"<<std::endl;
   	str<<"get_possible_links = "<<C.get_possible_links()<<std::endl;
   	for(auto h = 0; h < C.get_possible_links(); ++h){
   		std::pair<unsigned int, unsigned int> ij=C.pos_to_ij(h);
   		str<<"pos = "<<h<<" -> ("<<ij.first<<", "<<ij.second<<")"<<std::endl;
   	}


   	str<<std::endl;
   	str<<"----------------------------------------------------------------------"<<std::endl;
   	str<<"Prodotti cartesiani"<<std::endl;
   	str<<"Gruppi = "<<std::endl<<Gr<<std::endl;
   	std::vector<std::pair<unsigned int, unsigned int>> ixj0(BG.map_to_complete(0,1));
   	std::vector<std::pair<unsigned int, unsigned int>> ixj1(BG.map_to_complete(1,2));
   	std::vector<std::pair<unsigned int, unsigned int>> ixj2(BG.map_to_complete(2,3));
   	std::vector<std::pair<unsigned int, unsigned int>> ixj3(BG.map_to_complete(2,1));
   	std::vector<std::pair<unsigned int, unsigned int>> ixj4(BG.map_to_complete(0,0));



   	str<<"(0,1)"<<std::endl;
   	for(auto __v : ixj0)
   		str<<__v.first<<", "<<__v.second<<std::endl;
   	str<<std::endl;
   	str<<"(1,2)"<<std::endl;
   	for(auto __v : ixj1)
   		str<<__v.first<<", "<<__v.second<<std::endl;
   	str<<std::endl;
   	str<<"(2,3)"<<std::endl;
   	for(auto __v : ixj2)
   		str<<__v.first<<", "<<__v.second<<std::endl;
   	str<<std::endl;
   	str<<"(2,1)"<<std::endl;
   	for(auto __v : ixj3)
   		str<<__v.first<<", "<<__v.second<<std::endl;
   	str<<std::endl;
   	str<<"(0,0)"<<std::endl;
   	for(auto __v : ixj4)
   		str<<__v.first<<", "<<__v.second<<std::endl;
   	str<<std::endl;


}


#endif
