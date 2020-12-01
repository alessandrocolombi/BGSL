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
//#include "include_helpers.h"

/*
using namespace std;
using MyEigenMat 	= Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatRow        = MyEigenMat;
using MatCol 		= Eigen::MatrixXd;
using Neighbourhood = BlockGraphBase<bool>::Neighbourhood;
using ColType       = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using RowType       = Eigen::Matrix<double, 1, Eigen::Dynamic>;


//' Function for testing properties of GWishart, Spline and Random numbers generator
//'
//' @param No parameters are required
//' @export
// [[Rcpp::export]]
void TestGWish()
{
	//auto& str = std::cout;
	auto& str = Rcpp::Rcout;
	unsigned int  p = 7;
	unsigned int  M = 3;
	Groups Gr_tanti(M, p);
	Groups Gr({ {0,1,2}, {3}, {4,5}, {6} });

	std::vector<unsigned int> adj{true, true, true, true, false, true, true, false};
	std::shared_ptr<const Groups> ptr_gruppi(std::make_shared<const Groups>(Gr));
	std::shared_ptr<const Groups> ptr_gruppi_tanti(std::make_shared<const Groups>(Gr_tanti));

	auto start = std::chrono::high_resolution_clock::now();
	BlockGraph BG(adj, ptr_gruppi);
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> timer = stop - start;
	str << "Time:  " << timer.count()<<" ms"<< endl;

	BlockGraphAdj BAD(adj, ptr_gruppi);
	std::vector<bool>    adj_bool{true, true, true, true, false, true, true, false};
	BlockGraph<bool> 	BG_bool(adj_bool, ptr_gruppi);

	//Completi
	CompleteView CV(BG);
	GraphType C(p);

	C.fillRandom(0.5, 1234);
	str<<"------------------------------------------------------------------------"<<std::endl;
	GWishart< CompleteView > K1( std::make_shared<const CompleteView<> >(CV) );
	GWishart K2(std::make_shared<const GraphType<> >(C));

	//Creo una matrice sdp
	MyEigenMat X(p,p);
	for(unsigned int i = 0; i < p; i++)
	{
		X(i,i) = 1;
	    if(i < p-1)
			X(i+1,i) = -0.9;
	}

	str<<"------------------------------------------------------------------------"<<std::endl;
	str<<"Provo ad estrarre sottomatrici parte due the great return"<<std::endl;
		//p = 6;
	MyEigenMat Mat(MyEigenMat::Random(p,p));
	//str<<"Mat: "<<endl<<Mat<<std::endl;

	str<<"Sottomatrice quadrata generica"<<std::endl;
	str<<"Mat: "<<endl<<Mat<<std::endl;
	MyEigenMat Sotto0( utils::SubMatrix({0,1}, Mat) );
	MyEigenMat Sotto1 = utils::SubMatrix<utils::Symmetric::True>({0,1}, Mat);
	str<<"Sotto0 (non sym): "<<endl<<Sotto0<<std::endl;
	str<<"Sotto1 (sym): "<<endl<<Sotto1<<std::endl;

	str<<"Sottomatrice tutti tranne uno"<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	MyEigenMat Sotto2 = utils::SubMatrix(2, Mat);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"Fill diretto, non sym -> "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	MyEigenMat Sotto3 = utils::SubMatrix<utils::Symmetric::True>(2, Mat);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"Fill diretto, sym -> "<<timer.count()<<" ms "<<std::endl;

	str<<"Sotto2 (non sym): "<<endl<<Sotto2<<std::endl;
	str<<"Sotto3 (sym): "<<endl<<Sotto3<<std::endl;

	str<<"Sottomatrice rettangolare"<<std::endl;
	MyEigenMat Sotto4 = utils::SubMatrix({0,1,2},{3,4}, Mat);
	str<<"Sotto4: "<<endl<<Sotto4<<std::endl;

	str<<"Colonna"<<std::endl;
	Eigen::VectorXd Sotto5 = utils::SubMatrix({0,2},1, Mat);
	str<<"Sotto5: "<<endl<<Sotto5<<std::endl;

	str<<"Riga"<<std::endl;
	MyEigenMat Sotto6 = utils::SubMatrix(1,{1,3}, Mat);
	str<<"Sotto6: "<<endl<<Sotto6<<std::endl;
	str<<std::endl;

	str<<"------------------------------------------------------------------------"<<std::endl;
	str<<"Provo operazioni con le matrici appena estratte"<<std::endl;

	MyEigenMat Prodotto = Sotto0 * Sotto1;
	str<<"Prodotto: "<<endl<<Prodotto<<std::endl;

	MyEigenMat Solve = Sotto0.lu().solve(Sotto5);
	str<<"Solve: "<<endl<<Solve<<std::endl;
	str<<std::endl;


	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"############################ TEST RGWISH ###############################"<<std::endl;

	unsigned int dim = 4;
	GraphType Grafo(dim);
	Grafo.fillRandom(0.5, 123);
	str<<"Grafo: "<<endl<<Grafo<<std::endl;
	GWishartTraits::Shape b = 3;
	GWishartTraits::InvScale D = MyEigenMat::Identity(dim,dim);
	//MyEigenMat provaA(MyEigenMat::Random(3,3));
	//MyEigenMat provaB(MyEigenMat::Random(3,3));
	//str<<"|| A - B || = "<<utils::MeanNorm::norm(provaA,provaB)<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	MyEigenMat res_inf( utils::rgwish<GraphType, unsigned int, utils::NormInf>(Grafo, b, D) );
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"NormInf -> "<<timer.count()<<" ms "<<std::endl;
	str<<"Rispetta la struttura? "<<utils::check_structure(Grafo, res_inf)<<std::endl;
	str<<"Risultato : "<<std::endl<<res_inf<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	MyEigenMat res_sq( utils::rgwish<GraphType, unsigned int, utils::NormSq>(Grafo, b, D) );
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"NormSq -> "<<timer.count()<<" ms "<<std::endl;
	str<<"Rispetta la struttura? "<<utils::check_structure(Grafo, res_sq)<<std::endl;
	str<<"Risultato : "<<std::endl<<res_sq<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	MyEigenMat res_mean( utils::rgwish(Grafo, b, D) );
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"Mean norm -> "<<timer.count()<<" ms "<<std::endl;
	str<<"Rispetta la struttura? "<<utils::check_structure(Grafo, res_mean)<<std::endl;
	str<<"Risultato : "<<std::endl<<res_mean<<std::endl;
	//str<<"Returned value: "<<endl<<res<<std::endl;
	//str<<"Grafo: "<<endl<<Grafo<<std::endl;
	str<<std::endl;
	str<<"------------------------------------------------------------------------"<<std::endl;
	GWishart<CompleteView, unsigned int > T( std::make_shared<const CompleteView<> >(CV) );
	str<<"Grafo: "<<endl<<CV<<std::endl;
	T.set_random();
	//T.print();
	str<<"Rispetta la struttura? "<<T.check_structure()<<std::endl;
	str<<std::endl;

	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"####################### TEST NORMALIZING CONSTANT ######################"<<std::endl;

	str<<"Inizio il testing per I_G: test grafo Massam"<<std::endl;
	GraphType G1_massam({1,0,1,1,0,1});
	str<<"Grafo 1 Massam : "<<endl<<G1_massam<<std::endl;

	GWishart K1_massam( std::make_shared<const GraphType<> >(G1_massam) );
	Eigen::MatrixXd new_D(Eigen::MatrixXd::Identity(4,4)*10);
	new_D(0,1) = 1;
	new_D(0,2) = 1;
	new_D(0,3) = 1;
	K1_massam.set_inv_scale(new_D.selfadjointView<Eigen::Upper>());
	start = std::chrono::high_resolution_clock::now();
	double IG_K1_massam = K1_massam.log_normalizing_constat(1000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"Time : "<<timer.count()<<" ms "<<std::endl;
	str<<"res = "<<IG_K1_massam<<std::endl;

	double free_function = utils::log_normalizing_constat(G1_massam, 3, new_D.selfadjointView<Eigen::Upper>(), 1000, 0);
	str<<"res free function = "<<free_function<<std::endl;

	p = 10;
	GraphType G_IG(p);
	G_IG.fillRandom(0.5, 1234);
	//str<<"Grafo completo: "<<endl<<G_IG<<std::endl;
	GWishart T_IG( std::make_shared<const GraphType<> >(G_IG) );

	str<<"########################################################################"<<std::endl;
	str<<"########################## Confronto con BDgraph #######################"<<std::endl;
	str<<"Confronto con BDgraph"<<std::endl;
	str<<std::endl<<"********* G piccolo (paper massam): ********* "<<std::endl;
	//GraphType G1_massam({1,0,1,1,0,1});
	GWishart K1_massam_test( std::make_shared<const GraphType<> >(G1_massam) );
	double res_test;

	start = std::chrono::high_resolution_clock::now();
	res_test = K1_massam_test.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D identità -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	//--> risultato uguale (9.25), BD ci mette 20.36 ms, io 37 con debug, 7.1 ms con ottimizzazione 03

	K1_massam_test.set_inv_scale(Eigen::MatrixXd::Identity(4,4)*10);
	start = std::chrono::high_resolution_clock::now();
	res_test = K1_massam_test.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D diagonale -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	// --> risultato uguale (-13.76499), BD ci mette 20.37 ms, io 37 ms con debug, 8.36 ms con ottimizzazione 03

	new_D = Eigen::MatrixXd::Identity(4,4)*10;
	new_D(0,1) = 1;
	new_D(0,2) = 1;
	new_D(0,3) = 1;
	K1_massam_test.set_inv_scale(new_D.selfadjointView<Eigen::Upper>());
	start = std::chrono::high_resolution_clock::now();
	res_test = K1_massam_test.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D generica -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	// --> risultato uguale ( -13.72538 ), BD ci mette 27 ms, io 92 ms con debug, 9.61 ms con ott 03

	str<<std::endl<<"********* G 10x10 ********* "<<std::endl;
	p = 10;
	//GraphTYpe G_IG(p);
	//G_IG.fillRandom(0.5, 1234);
	//str<<"Grafo 10x10:"<<endl<<G_IG<<std::endl;
	GWishart T_IG_test( std::make_shared<const GraphType<> >(G_IG) );

	start = std::chrono::high_resolution_clock::now();
	res_test = T_IG_test.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D identità -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	// --> risultato uguale ( 38.38657 ), BD ci mette 54,6 ms, io 315 con degub, 26.89 ms con ottimizzazine 03

	Eigen::MatrixXd D_temp(Eigen::MatrixXd::Identity(p,p)*5);
	for(auto i = 0; i < p-1; ++i){
	    D_temp(i,i+1) = -2;
	}
	Eigen::MatrixXd D_vs_BDgraph = D_temp + D_temp.transpose();
	T_IG_test.set_inv_scale(D_vs_BDgraph);
	start = std::chrono::high_resolution_clock::now();
	res_test = T_IG_test.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D generica -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	// --> risultato uguale (-41.51813), BD ci mette 68 ms, io 901 ms con debug, 43 ms con ottimizzazione 03
	str<<std::endl;
	str<<std::endl<<"********* Grafo a blocchi ********* "<<std::endl;
	//GWishart<CompleteView<> > T( std::make_shared<const CompleteView<> >(CV) );
	//str<<"Grafo: "<<endl<<CV<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	res_test = T.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D identità (member function) -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	//BDgraph da come risultato 32.74
	start = std::chrono::high_resolution_clock::now();
	res_test = utils::log_normalizing_constat(CV,3,Eigen::MatrixXd::Identity(CV.get_size(),CV.get_size()),10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D identità (free function) -> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	//BDgraph da come risultato 32.74
	D_temp = Eigen::MatrixXd::Identity(CV.get_size(),CV.get_size())*5;
	for(auto i = 0; i < CV.get_size()-1; ++i){
	    D_temp(i,i+1) = -2;
	}
	D_vs_BDgraph = D_temp + D_temp.transpose();

	T.set_inv_scale(D_vs_BDgraph);
	start = std::chrono::high_resolution_clock::now();
	res_test = T.log_normalizing_constat(10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D generica (member function)-> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	//BDgraph da come risusltato -29.92038
	start = std::chrono::high_resolution_clock::now();
	res_test = utils::log_normalizing_constat(CV,3,D_vs_BDgraph,10000,0);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"D generica (free function)-> : "<<timer.count()<<" ms "<<"; risultato = "<<res_test<<std::endl;
	//BDgraph da come risusltato -29.92038

	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"############################## TEST SPLINE #############################"<<std::endl;


	MatCol Basemat_fun(spline::generate_design_matrix(4, 7, 0.0, 10.0, {0.0,2.0,4.0,6.0,8.0,10.0}));
	str<<"Basemat_fun 1"<<std::endl;
	//str<<Basemat_fun<<std::endl;

	Basemat_fun = spline::generate_design_matrix(4, 7, 0.0, 10.0, 6);
	str<<"Basemat_fun 2"<<std::endl;
	//str<<Basemat_fun<<std::endl;

	int nderiv{2};
	std::vector<MatCol> vect_BaseDer(spline::evaluate_spline_derivative(4, 7, 0.0, 10.0, {0.0,2.0,4.0,6.0,8.0,10.0}, nderiv));
	str<<"Con anche le derivate"<<std::endl;
	//for(auto __v : vect_BaseDer)
				//str<<__v<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"############################## TEST SAMPLE #############################"<<std::endl;
	unsigned int N=1000;
	sample::GSL_RNG engine_gsl;
	//sample::GSL_RNG engine_gsl(111120);
	str<<"Some information about the generator"<<std::endl;
	engine_gsl.print_info();
	str<<std::endl;

	double res;
	start = std::chrono::high_resolution_clock::now();
	for(size_t i=0; i<N; ++i)
			res = sample::rnorm()(engine_gsl);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rnorm "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	for(size_t i=0; i<N; ++i)
			res = sample::rgamma()(engine_gsl, 5, 2);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rgamma "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
	for(size_t i=0; i<N; ++i)
			res = sample::rchisq()(5);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rchisq "<<timer.count()<<" ms "<<std::endl;


	MyEigenMat MY_D(MyEigenMat::Identity(p,p));
	MyEigenMat res_wish(MyEigenMat::Identity(p,p));
	start = std::chrono::high_resolution_clock::now();
	for(size_t i=0; i<N; ++i)
			res_wish = sample::rwish()(engine_gsl, 3, MY_D);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish "<<timer.count()<<" ms "<<std::endl;
	//str<<"Print fuori"<<std::endl;
	//str<<res_wish<<std::endl;
	str<<std::endl;

	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"############################## TEST RWISH ##############################"<<std::endl;
	//creo matrici di passare in input
	//colonne
	Eigen::MatrixXd Inv_D_GSL(D_vs_BDgraph.llt().solve(Eigen::MatrixXd::Identity(D_vs_BDgraph.rows(),D_vs_BDgraph.cols())));
	//Righe
	MyEigenMat D_temp2 = MyEigenMat::Identity(CV.get_size(),CV.get_size())*5;
	for(auto i = 0; i < CV.get_size()-1; ++i){
	    D_temp2(i,i+1) = -2;
	}
	MyEigenMat D_vs_BDgraph2 = D_temp2 + D_temp2.transpose();
	MyEigenMat Inv_D_GSL2(D_vs_BDgraph2.llt().solve(MyEigenMat::Identity(D_vs_BDgraph2.rows(),D_vs_BDgraph2.cols())));

	//Creo matrici con i risultati
	MyEigenMat 		res_GSL_row(MyEigenMat::Identity(D_vs_BDgraph.rows(), D_vs_BDgraph.rows()));
	Eigen::MatrixXd res_GSL_col(MyEigenMat::Identity(D_vs_BDgraph.rows(), D_vs_BDgraph.rows()));
	//Creo oggetto rwish
	sample::rwish<MatCol> 	  WishC;
	sample::rwish<MyEigenMat> WishR;

	//Run test
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishC(engine_gsl, 3, Inv_D_GSL);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra colonne, esce colonne) "<<timer.count()<<" ms "<<std::endl;
	//str<<"Risultato con GSL (entra colonne, esce colonne):"<<std::endl<<res_GSL_col<<std::endl;


	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_row = WishR(engine_gsl, 3, Inv_D_GSL);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra colonne, esce righe) "<<timer.count()<<" ms "<<std::endl;
	//str<<"Risultato con GSL (entra colonne, esce righe):"<<std::endl<<res_GSL_row<<std::endl;


	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_row = sample::rwish<MyEigenMat>()(engine_gsl, 3, Inv_D_GSL2);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra righe, esce righe) "<<timer.count()<<" ms "<<std::endl;
	//str<<"Risultato con GSL (entra righe, esce righe):"<<std::endl<<res_GSL_row<<std::endl;

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = sample::rwish()(engine_gsl, 3, Inv_D_GSL2);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra righe, esce colonne) "<<timer.count()<<" ms "<<std::endl;
	//str<<"Risultato con GSL (entra righe, esce colonne):"<<std::endl<<res_GSL_col<<std::endl;

	str<<std::endl;

	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"###################### TEST RNORM PRECISION ############################"<<std::endl;


	dim = 50;

	// 2) Creo una matrice sdp per righe
	MyEigenMat X_mvn(MyEigenMat::Zero(dim,dim));
	for(unsigned int i = 0; i < dim; i++)
	{
		X_mvn(i,i) = 1;
	    if(i < dim-1)
			X_mvn(i+1,i) = -0.9;
	}
	MyEigenMat XX_mvn = X_mvn * X_mvn.transpose();
	// 3) Calcolo il suo chol upper e lower trg
	MyEigenMat XX_mvn_L = XX_mvn.llt().matrixL();
	MyEigenMat XX_mvn_U = XX_mvn.llt().matrixU();
	//str<<"XX_mvn:"<<std::endl<<XX_mvn<<std::endl;
	//str<<"XX_mvn_L:"<<std::endl<<XX_mvn_L<<std::endl;
	//str<<"XX_mvn_U:"<<std::endl<<XX_mvn_U<<std::endl;
	// 4) Passando precisione per colonne
	MatCol P_col(MatCol::Zero(dim,dim));
	for(unsigned int i = 0; i < dim; i++)
	{
		P_col(i,i) = 1;
	    if(i < dim-1)
			P_col(i+1,i) = -0.9;
	}
	MatCol Prec_col = P_col * P_col.transpose();
	// 5) Calcolo is suo chol upper e lower trg
	MyEigenMat Prec_col_L = Prec_col.llt().matrixL();
	MyEigenMat Prec_col_U = Prec_col.llt().matrixU();
	//str<<"Prec_col:"<<std::endl<<Prec_col<<std::endl;
	//str<<"Prec_col_L:"<<std::endl<<Prec_col_L<<std::endl;
	//str<<"Prec_col_U:"<<std::endl<<Prec_col_U<<std::endl;

	// Ora inizia il vero sampling
	//sample::GSL_RNG engine_gsl;
	Eigen::VectorXd mean_mvn(Eigen::VectorXd::Zero(dim));
	Eigen::VectorXd res_mvn(Eigen::VectorXd::Zero(dim));
	str<<" ############ Passo una matrice per righe ############ "<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm_prec()(engine_gsl, mean_mvn, XX_mvn);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (non chol) "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm_prec<sample::isChol::Upper>()(engine_gsl, mean_mvn, XX_mvn_U);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (chol upper) "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm_prec<sample::isChol::Lower>()(engine_gsl, mean_mvn, XX_mvn_L);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (chol lower) "<<timer.count()<<" ms "<<std::endl;
	str<<std::endl;
	str<<" ############ Passo una matrice per colonne ############ "<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm_prec()(engine_gsl, mean_mvn, Prec_col);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (non chol) "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm_prec<sample::isChol::Upper>()(engine_gsl, mean_mvn, Prec_col_U);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (chol upper) "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm_prec<sample::isChol::Lower>()(engine_gsl, mean_mvn, Prec_col_L);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (chol lower) "<<timer.count()<<" ms "<<std::endl;
	str<<std::endl;

	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"###################### TEST RNORM COVARIANCE ###########################"<<std::endl;

	// 1) Creo una matrice sdp per righe e la inverto
	MyEigenMat XX_inv(XX_mvn.inverse());
	// 4) Passando covarianza per colonne
	MatCol Prec_col_inv(Prec_col.inverse());
	//str<<"XX_inv:"<<std::endl<<XX_inv<<std::endl;
	//str<<"P_col_inv:"<<std::endl<<Prec_col_inv<<std::endl;

	// Ora inizia il vero sampling
	//sample::GSL_RNG engine_gsl;
	Eigen::VectorXd mean2_mvn(Eigen::VectorXd::Random(dim)*100);
	//str<<"media = "<<endl<<mean2_mvn<<std::endl;
	//Eigen::VectorXd res_mvn(Eigen::VectorXd::Zero(dim));

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm()(engine_gsl, mean2_mvn, XX_inv);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm (righe) "<<timer.count()<<" ms "<<std::endl;

	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_mvn = sample::rmvnorm()(engine_gsl, mean2_mvn, Prec_col_inv);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rmvnorm_prec (colonne) "<<timer.count()<<" ms "<<std::endl;

	//str<<"res = "<<endl<<res_mvn<<std::endl;



	str<<"########################################################################"<<std::endl;
	str<<"########################################################################"<<std::endl;
	str<<"###################### TEST RWISH WITH CHOL ############################"<<std::endl;


	//creo matrici da passare in input
	N = 1000;
	//Colonne
	//Eigen::MatrixXd Inv_D_GSL(D_vs_BDgraph.llt().solve(Eigen::MatrixXd::Identity(D_vs_BDgraph.rows(),D_vs_BDgraph.cols())));
	MatCol Inv_D_GSL_L(Inv_D_GSL.llt().matrixL());
	MatCol Inv_D_GSL_U(Inv_D_GSL.llt().matrixU());

	//Righe
	//MyEigenMat Inv_D_GSL2(D_vs_BDgraph2.llt().solve(MyEigenMat::Identity(D_vs_BDgraph2.rows(),D_vs_BDgraph2.cols())));
	MatRow  Inv_D_GSL2_L(Inv_D_GSL2.llt().matrixL());
	MatRow  Inv_D_GSL2_U(Inv_D_GSL2.llt().matrixU());


	//Creo matrici con i risultati
	//MyEigenMat 		res_GSL_row(MyEigenMat::Identity(D_vs_BDgraph.rows(), D_vs_BDgraph.rows()));
	//Eigen::MatrixXd res_GSL_col(MyEigenMat::Identity(D_vs_BDgraph.rows(), D_vs_BDgraph.rows()));

	//template<typename RetType = MatCol, isChol isCholType = isChol::False>
	//Creo oggetto rwish
	sample::rwish<MatCol, sample::isChol::False>  WishF;
	sample::rwish<MatCol, sample::isChol::Upper>  WishU;
	sample::rwish<MatCol, sample::isChol::Lower>  WishL;
	N = 1000;
	//Run test
	str<<" ############ Passo una matrice per colonne ############ "<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishF(engine_gsl, 3, Inv_D_GSL);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra colonne, non chol) "<<timer.count()<<" ms "<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishL(engine_gsl, 3, Inv_D_GSL_L);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra colonne, chol lower) "<<timer.count()<<" ms "<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishU(engine_gsl, 3, Inv_D_GSL_U);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra colonne, chol upper) "<<timer.count()<<" ms "<<std::endl;
	str<<" ############ Passo una matrice per righe ############ "<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishF(engine_gsl, 3, Inv_D_GSL2);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra righe, non chol) "<<timer.count()<<" ms "<<std::endl;
		//str<<"Risultato con GSL (entra colonne, esce colonne):"<<std::endl<<res_GSL_col<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishL(engine_gsl, 3, Inv_D_GSL2_L);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra righe, chol lower) "<<timer.count()<<" ms "<<std::endl;
		//str<<"Risultato con GSL (entra colonne, esce righe):"<<std::endl<<res_GSL_row<<std::endl;
	start = std::chrono::high_resolution_clock::now();
		for(size_t i=0; i<N; ++i)
			res_GSL_col = WishU(engine_gsl, 3, Inv_D_GSL2_U);
	stop = std::chrono::high_resolution_clock::now();
	timer = stop - start;
	str<<"sample::rwish (entra righe, chol upper) "<<timer.count()<<" ms "<<std::endl;
	str<<std::endl;
}

*/


#endif
