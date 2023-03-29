library(BDgraph)
library(BGSL)
library(tidyverse)


# p40 - experiment 1 ----------------------------------------------------------
data("examples")
True_val = examples$Trueval_p40

data = True_val$data
Gcomp = True_val$Gcomp
G = True_val$G
Ktrue = True_val$Ktrue

#Set dimensions
p    = 40
Nrep = 1
n    = 500
M    = 20

niter     = 400000
nburn     = niter/2
thin      =      1
thinG     =      1



Testp40_exampledata_BDRJ_BDgraph_DRJ = vector("list",length = 3)

# p40 - BDRJ --------------------------------------------------------------------
#Block DRJ
file_name = "Testp40_exampledata_BDRJ"
file_name_ext = paste0(file_name,".h5")
cat('\n inizio BDRJ \n')

#Parameters and hyperparameters
threshold =  1e-14
hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
param = sampler_parameters(threshold = threshold)
algo    = "DRJ"
prior   = "Uniform"
form    = "Block"

#initial values
init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

#Run
result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                      Param = param, HyParam = hy, Init = init,
                      prior = prior, form = form, algo = algo, file_name = file_name,
                      groups = NULL, n_groups = M, seed = 123, print_info = F)

info = Read_InfoFile(file_name = file_name_ext)
#1) plinks
plinks = result$plinks
#2) bfdr selection
an = BFDR_selection(plinks, min_rate = 0.05, diag = T)
threshold = an$best_treshold
#3) complete form
PL = matrix(0,M,M)
PL[plinks >= threshold] = 1
Gfinal_est = Block2Complete(PL, CreateGroups(p,M))
#4) misclassified links
diff = abs(Gcomp - Gfinal_est)
diff[lower.tri(diff)] = 0
errors = sum(diff)
#5) Table
Gcomp_vett = Gcomp[upper.tri(Gcomp)]
Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
Table = table(Gcomp_vett, Gfinal_est_vett)
TP = Table[2,2]
FP = Table[1,2]
FN = Table[2,1]
TN = Table[1,1]
Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN, "errors"=errors, "std_hamming"=errors/(p*(p-1)))
#6) Save
Testp40_exampledata_BDRJ_BDgraph_DRJ[[1]] = Gfinal_est


# p40 - BDgraph -----------------------------------------------------------------

#BDgraph
#run
res = bdgraph(data = data, n = n, iter = niter, burnin = nburn, threshold = 1e-14)
#1) plinks
plinks_BD = plinks(res)
#2) bfdr selection
an_BD = BFDR_selection(plinks_BD,min_rate = 0.05, diag = F)
threshold_BD = an_BD$best_treshold
#3) complete form
Gfinal_est_BD = matrix(0,p,p)
Gfinal_est_BD[plinks_BD >= threshold_BD] = 1
#4) misclassified links
diff_BD = abs(Gcomp - Gfinal_est_BD)
diff_BD[lower.tri(diff_BD)] = 0
errors_BD = sum(diff_BD) - p
#5) Table
Gcomp_vett = Gcomp[upper.tri(Gcomp)]
Gfinal_est_vett_BD = Gfinal_est_BD[upper.tri((Gfinal_est_BD))]
Table_BD = table(Gcomp_vett, Gfinal_est_vett_BD)
TP_BD = Table_BD[2,2]
FP_BD = Table_BD[1,2]
FN_BD = Table_BD[2,1]
TN_BD = Table_BD[1,1]
Table_res_BD = list("Table"=Table_BD, "TP"=TP_BD, "FP"=FP_BD, "FN"=FN_BD, "TN"=TN_BD,
                    "errors"=errors_BD, "std_hamming"=errors_BD/(p*(p-1)))

#6) Save
Testp40_exampledata_BDRJ_BDgraph_DRJ[[2]] = Gfinal_est_BD

# p40 - DRJ ---------------------------------------------------------------------

file_name = "Testp40_exampledata_DRJ"
file_name_ext = paste0(file_name,".h5")
cat('\n inizio DRJ = \n')

#Block DRJ
#Parameters and hyperparameters
threshold =  1e-14
hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
param = sampler_parameters(threshold = threshold)
algo    = "DRJ"
prior   = "Uniform"
form    = "Complete"

#initial values
init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

#Run
result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                      Param = param, HyParam = hy, Init = init,
                      prior = prior, algo = algo, file_name = file_name,
                      groups = NULL, seed = 123, print_info = F)

info = Read_InfoFile(file_name = file_name_ext)
#1) plinks
plinks = result$plinks
#2) bfdr selection
an = BFDR_selection(plinks,min_rate = 0.05, diag = T)
threshold = an$best_treshold
#3) complete form
PL = matrix(0,p,p)
PL[plinks >= threshold] = 1
Gfinal_est = PL
#4) misclassified links
diff = abs(Gcomp - Gfinal_est)
diff[lower.tri(diff)] = 0
errors = sum(diff)
#5) Table
Gcomp_vett = Gcomp[upper.tri(Gcomp)]
Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
Table = table(Gcomp_vett, Gfinal_est_vett)
TP = Table[2,2]
FP = Table[1,2]
FN = Table[2,1]
TN = Table[1,1]
Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN, "errors"=errors, "std_hamming"=errors/(p*(p-1)))

#6) Save
Testp40_exampledata_BDRJ_BDgraph_DRJ[[3]] = Gfinal_est


names(Testp40_exampledata_BDRJ_BDgraph_DRJ) = c("BDRJ","BDgraph","DRJ")


# p40 - plots --------------------------------------------------------------------

GBDgraph = Testp40_exampledata_BDRJ_BDgraph_DRJ$BDgraph
GBDgraph = GBDgraph + t(GBDgraph)
diag(GDBgraph) =  rep(2,p)
GBDRJ = Testp40_exampledata_BDRJ_BDgraph_DRJ$BDRJ
GBDRJ = GBDRJ + t(GBDRJ)
diag(GBDRJ) =  rep(2,p)
GDRJ = Testp40_exampledata_BDRJ_BDgraph_DRJ$DRJ
GDRJ = GDRJ + t(GDRJ)
diag(GDRJ) =  rep(2,p)
diag(Gcomp) = rep(2,p)


par(mfrow = c(2,2), mar = c(1,1,1,1))
ACheatmap(
  Gcomp,
  use_x11_device = F,
  horizontal = F,
  main = "True Graph",
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)

ACheatmap(
  GBDRJ,
  use_x11_device = F,
  horizontal = F,
  main = "BDRJ",
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)

ACheatmap(
  GBDgraph,
  use_x11_device = F,
  horizontal = F,
  main = "BDgraph",
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)


ACheatmap(
  GDRJ,
  use_x11_device = F,
  horizontal = F,
  main = "DRJ",
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)


# p30 - experiment 1 ----------------------------------------------------------
data("examples")
True_val = examples$Trueval_p30

data = True_val$data
Gcomp = True_val$Gcomp
G = True_val$G
Ktrue = True_val$Ktrue

#Set dimensions
p    = 30
Nrep = 1
n    = 500
M    = 15

niter     = 400000
nburn     = niter/2
thin      =      1
thinG     =      1



Testp30_exampledata_BDRJ_BDgraph_DRJ = vector("list",length = 3)

# p30 - BDRJ --------------------------------------------------------------------
#Block DRJ
file_name = "Testp30_exampledata_BDRJ"
file_name_ext = paste0(file_name,".h5")
cat('\n inizio BDRJ \n')

#Parameters and hyperparameters
threshold =  1e-14
hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
param = sampler_parameters(threshold = threshold)
algo    = "DRJ"
prior   = "Uniform"
form    = "Block"

#initial values
init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

#Run
result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                      Param = param, HyParam = hy, Init = init,
                      prior = prior, form = form, algo = algo, file_name = file_name,
                      groups = NULL, n_groups = M, seed = 123, print_info = F)

info = Read_InfoFile(file_name = file_name_ext)
#1) plinks
plinks = result$plinks
#2) bfdr selection
an = BFDR_selection(plinks,min_rate = 0.05, diag = T)
threshold = an$best_treshold
#3) complete form
PL = matrix(0,M,M)
PL[plinks >= threshold] = 1
Gfinal_est = Block2Complete(PL, CreateGroups(p,M))
#4) misclassified links
diff = abs(Gcomp - Gfinal_est)
diff[lower.tri(diff)] = 0
errors = sum(diff)
#5) Table
Gcomp_vett = Gcomp[upper.tri(Gcomp)]
Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
Table = table(Gcomp_vett, Gfinal_est_vett)
TP = Table[2,2]
FP = Table[1,2]
FN = Table[2,1]
TN = Table[1,1]
Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN, "errors"=errors, "std_hamming"=errors/(p*(p-1)))
#6) Save
Testp30_exampledata_BDRJ_BDgraph_DRJ[[1]] = Gfinal_est


# p30 - BDgraph -----------------------------------------------------------------

#BDgraph
#run
res = bdgraph(data = data, n = n, iter = niter, burnin = nburn, threshold = 1e-14)
#1) plinks
plinks_BD = plinks(res)
#2) bfdr selection
an_BD = BFDR_selection(plinks_BD,min_rate = 0.05, diag = F)
threshold_BD = an_BD$best_treshold
#3) complete form
Gfinal_est_BD = matrix(0,p,p)
Gfinal_est_BD[plinks_BD >= threshold_BD] = 1
#4) misclassified links
diff_BD = abs(Gcomp - Gfinal_est_BD)
diff_BD[lower.tri(diff_BD)] = 0
errors_BD = sum(diff_BD) - p
#5) Table
Gcomp_vett = Gcomp[upper.tri(Gcomp)]
Gfinal_est_vett_BD = Gfinal_est_BD[upper.tri((Gfinal_est_BD))]
Table_BD = table(Gcomp_vett, Gfinal_est_vett_BD)
TP_BD = Table_BD[2,2]
FP_BD = Table_BD[1,2]
FN_BD = Table_BD[2,1]
TN_BD = Table_BD[1,1]
Table_res_BD = list("Table"=Table_BD, "TP"=TP_BD, "FP"=FP_BD, "FN"=FN_BD, "TN"=TN_BD,
                    "errors"=errors_BD, "std_hamming"=errors_BD/(p*(p-1)))

#6) Save
Testp30_exampledata_BDRJ_BDgraph_DRJ[[2]] = Gfinal_est_BD

# p30 - DRJ ---------------------------------------------------------------------

file_name = "Testp30_exampledata_DRJ"
file_name_ext = paste0(file_name,".h5")
cat('\n inizio DRJ = \n')

#Block DRJ
#Parameters and hyperparameters
threshold =  1e-14
hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
param = sampler_parameters(threshold = threshold)
algo    = "DRJ"
prior   = "Uniform"
form    = "Complete"

#initial values
init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

#Run
result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                      Param = param, HyParam = hy, Init = init,
                      prior = prior, algo = algo, file_name = file_name,
                      groups = NULL, seed = 123, print_info = F)

info = Read_InfoFile(file_name = file_name_ext)
#1) plinks
plinks = result$plinks
#2) bfdr selection
an = BFDR_selection(plinks,min_rate = 0.05, diag = T)
threshold = an$best_treshold
#3) complete form
PL = matrix(0,p,p)
PL[plinks >= threshold] = 1
Gfinal_est = PL
#4) misclassified links
diff = abs(Gcomp - Gfinal_est)
diff[lower.tri(diff)] = 0
errors = sum(diff)
#5) Table
Gcomp_vett = Gcomp[upper.tri(Gcomp)]
Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
Table = table(Gcomp_vett, Gfinal_est_vett)
TP = Table[2,2]
FP = Table[1,2]
FN = Table[2,1]
TN = Table[1,1]
Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN, "errors"=errors, "std_hamming"=errors/(p*(p-1)))

#6) Save
Testp30_exampledata_BDRJ_BDgraph_DRJ[[3]] = Gfinal_est


names(Testp30_exampledata_BDRJ_BDgraph_DRJ) = c("BDRJ","BDgraph","DRJ")


# p30 - plots --------------------------------------------------------------------

GBDgraph = Testp30_exampledata_BDRJ_BDgraph_DRJ$BDgraph
GBDgraph = GBDgraph + t(GBDgraph)
diag(GDBgraph) =  rep(2,p)
GBDRJ = Testp30_exampledata_BDRJ_BDgraph_DRJ$BDRJ
GBDRJ = GBDRJ + t(GBDRJ)
diag(GBDRJ) =  rep(2,p)
GDRJ = Testp30_exampledata_BDRJ_BDgraph_DRJ$DRJ
GDRJ = GDRJ + t(GDRJ)
diag(GDRJ) =  rep(2,p)
diag(Gcomp) = rep(2,p)


par(mfrow = c(2,2), mar = c(1,1,1,1))
ACheatmap(
  Gcomp,
  use_x11_device = F,
  horizontal = F,
  main = "True Graph",
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)

ACheatmap(
  GBDRJ,
  use_x11_device = F,
  horizontal = F,
  main = "BDRJ",
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)

ACheatmap(
  GBDgraph,
  use_x11_device = F,
  horizontal = F,
  main = "BDgraph",
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)


ACheatmap(
  GDRJ,
  use_x11_device = F,
  horizontal = F,
  main = "DRJ",
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)



# p40 - repeated experiments ---------------------------------------------
library(BDgraph)
library(BGSL)
library(tidyverse)

#Set dimensions
p    = 40
Nrep = 1
n    = 500
M    = p/2

niter     = 400000
nburn     = niter/2
thin      =      1
thinG     =      1
data("examples")
True_val_rep = examples$Trueval_p40_rep
Test_p40_rep = vector("list", length = 3)
Nsim = length(True_val_rep)
names(Test_p40_rep) = c("BDRJ","BDgraph","DRJ")
Test_p40_rep$BDRJ = vector("list", length = Nsim)
Test_p40_rep$BDgraph = vector("list", length = Nsim)
Test_p40_rep$DRJ = vector("list", length = Nsim)

for(i in 1:Nsim){

  file_name = paste0("GGM_n500p40_BDRJ_",i)
  file_name_ext = paste0(file_name,".h5")
  cat('\n inizio BDRJ rep = ',i,'\n')
  True_val = True_val_rep[[i]]
  data = True_val$data
  Gcomp = True_val$Gcomp
  G = True_val$G
  Ktrue = True_val$Ktrue
  #Block DRJ
  #Parameters and hyperparameters
  threshold =  1e-14
  hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
  param = sampler_parameters(threshold = threshold)
  algo    = "DRJ"
  prior   = "Uniform"
  form    = "Block"

  #initial values
  init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

  #Run
  result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                        Param = param, HyParam = hy, Init = init,
                        prior = prior, form = form, algo = algo, file_name = file_name,
                        groups = NULL, n_groups = M, seed = 123, print_info = F)


  # Read
  info = Read_InfoFile(file_name = file_name_ext)
  #1) plinks
  plinks = result$plinks
  #2) bfdr selection
  an = BFDR_selection(plinks,min_rate = 0.05, diag = T)
  threshold = an$best_treshold
  #3) complete form
  PL = matrix(0,M,M)
  PL[plinks >= threshold] = 1
  Gfinal_est = Block2Complete(PL, CreateGroups(p,M))
  #4) misclassified links
  diff = abs(Gcomp - Gfinal_est)
  diff[lower.tri(diff)] = 0
  errors = sum(diff)
  #5) Table
  Gcomp_vett = Gcomp[upper.tri(Gcomp)]
  Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
  Table = table(Gcomp_vett, Gfinal_est_vett)
  TP = Table[2,2]
  FP = Table[1,2]
  FN = Table[2,1]
  TN = Table[1,1]
  Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN,
                   "errors"=errors, "std_hamming"=errors/(p*(p-1)))
  #6) Save
  save_iter = list()
  save_iter[[1]] = Table_res
  save_iter[[2]] = Gfinal_est
  names(save_iter) = c( "Table_res","Gfinal")
  Test_p40_rep$BDRJ[[i]] = save_iter

  #BDgraph
  #run
  res = bdgraph(data = data, n = n, iter = niter, burnin = nburn, threshold = 1e-14)
  #1) plinks
  plinks_BD = plinks(res)
  #2) bfdr selection
  an_BD = BFDR_selection(plinks_BD,min_rate = 0.05, diag = F)
  threshold_BD = an_BD$best_treshold
  #3) complete form
  Gfinal_est_BD = matrix(0,p,p)
  Gfinal_est_BD[plinks_BD >= threshold_BD] = 1
  #4) misclassified links
  diff_BD = abs(Gcomp - Gfinal_est_BD)
  diff_BD[lower.tri(diff_BD)] = 0
  errors_BD = sum(diff_BD) - p
  #5) Table
  Gcomp_vett = Gcomp[upper.tri(Gcomp)]
  Gfinal_est_vett_BD = Gfinal_est_BD[upper.tri((Gfinal_est_BD))]
  Table_BD = table(Gcomp_vett, Gfinal_est_vett_BD)
  TP_BD = Table_BD[2,2]
  FP_BD = Table_BD[1,2]
  FN_BD = Table_BD[2,1]
  TN_BD = Table_BD[1,1]
  Table_res_BD = list("Table"=Table_BD, "TP"=TP_BD, "FP"=FP_BD, "FN"=FN_BD, "TN"=TN_BD,
                      "errors"=errors_BD, "std_hamming"=errors_BD/(p*(p-1)))

  #6) Save
  save_iter = list()
  save_iter[[1]] = Table_res_BD
  save_iter[[2]] = Gfinal_est_BD
  names(save_iter) = c("Table_res","Gfinal")
  Test_p40_rep$BDgraph[[i]] = save_iter


  #DRJ
  file_name = paste0("GGM_n500p40_DRJ_",i)
  file_name_ext = paste0(file_name,".h5")
  cat('\n inizio DRJ rep = ',i,'\n')



  #Parameters and hyperparameters
  threshold =  1e-14
  hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
  param = sampler_parameters(threshold = threshold)
  algo    = "DRJ"
  prior   = "Uniform"
  form    = "Complete"

  #initial values
  init = GM_init(p = p, n = n, empty = T, form = form)

  #Run
  result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                        Param = param, HyParam = hy, Init = init,
                        prior = prior, form = form, algo = algo, file_name = file_name,
                        groups = NULL, seed = 123, print_info = F)


  info = Read_InfoFile(file_name = file_name_ext)
  #1) plinks
  plinks = result$plinks
  #2) bfdr selection
  an = BFDR_selection(plinks,min_rate = 0.05, diag = T)
  threshold = an$best_treshold
  #3) complete form
  PL = matrix(0,p,p)
  PL[plinks >= threshold] = 1
  Gfinal_est = PL
  #4) misclassified links
  diff = abs(Gcomp - Gfinal_est)
  diff[lower.tri(diff)] = 0
  errors = sum(diff)
  #5) Table
  Gcomp_vett = Gcomp[upper.tri(Gcomp)]
  Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
  Table = table(Gcomp_vett, Gfinal_est_vett)
  TP = Table[2,2]
  FP = Table[1,2]
  FN = Table[2,1]
  TN = Table[1,1]
  Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN, "errors"=errors, "std_hamming"=errors/(p*(p-1)))
  #6) Save
  save_iter = list()
  save_iter[[1]] = Table_res
  save_iter[[2]] = Gfinal_est
  names(save_iter) = c("Table_res","Gfinal")
  Test_p40_rep$DRJ[[i]] = save_iter
}




# p40 repeated experiments - plot -----------------------------------------------------------

Test_p40_res_rep = tibble(TP = as.integer(), TN = as.integer(),
                          FP = as.integer(), FN = as.integer(),
                          SHD = as.numeric(), F1score = as.numeric(),
                          type = as.factor(NULL) )


for(i in 1:Nsim){
  par_res = tibble("TP"=c(Test_p40_rep$BDRJ[[i]]$Table_res$TP,
                          Test_p40_rep$BDgraph[[i]]$Table_res$TP,
                          Test_p40_rep$DRJ[[i]]$Table_res$TP),
                   "TN"=c(Test_p40_rep$BDRJ[[i]]$Table_res$TN,
                          Test_p40_rep$BDgraph[[i]]$Table_res$TN,
                          Test_p40_rep$DRJ[[i]]$Table_res$TN),
                   "FP"=c(Test_p40_rep$BDRJ[[i]]$Table_res$FP,
                          Test_p40_rep$BDgraph[[i]]$Table_res$FP,
                          Test_p40_rep$DRJ[[i]]$Table_res$FP),
                   "FN"=c(Test_p40_rep$BDRJ[[i]]$Table_res$FN,
                          Test_p40_rep$BDgraph[[i]]$Table_res$FN,
                          Test_p40_rep$DRJ[[i]]$Table_res$FN),
                   "SHD"=c(Test_p40_rep$BDRJ[[i]]$Table_res$std_hamming,
                           Test_p40_rep$BDgraph[[i]]$Table_res$std_hamming,
                           Test_p40_rep$DRJ[[i]]$Table_res$std_hamming),
                   "F1score" = (2*TP)/(2*TP+FP+FN),
                   "type" = as.factor(c("BDRJ","BDgraph","DRJ"))
  )
  Test_p40_res_rep = Test_p40_res_rep %>% rbind(par_res)
}

data = Test_p40_res_rep

# SHD
data %>% group_by(type) %>%summarise(median(SHD))
data %>% group_by(type) %>%summarise(sd(SHD))

# F1 score
data %>% group_by(type) %>%summarise(median(F1score))
data %>% group_by(type) %>%summarise(sd(F1score))

# reorder labels for plot
data$type <- factor(data$type, levels = c("BDRJ", "DRJ", "BDgraph"))


# SHD
data %>% mutate(type = as.factor(type)) %>%
  ggplot(aes(y=SHD, x=type, fill=type)) + geom_boxplot() +
  labs(y="Standardized - SHD", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25)) +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))

# F1 SCORE
data %>% mutate(type = as.factor(type)) %>%
  ggplot(aes(y=F1score, x=type, fill=type)) + geom_boxplot() +
  labs(y="F1 - score", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25))  +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))

# SENS
data %>% mutate(type = as.factor(type)) %>%
  mutate(Sens = TP/(TP+FN)) %>%
  ggplot(aes(y=Sens, x=type, fill=type)) + geom_boxplot() +
  labs(y="Sensibility", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25))  +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))


# SPEC
data %>% mutate(type = as.factor(type)) %>%
  mutate(Spec = TN/(TN+FP)) %>%
  ggplot(aes(y=Spec, x=type, fill=type)) + geom_boxplot() +
  labs(y="Specificity", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25))  +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))

# p30 - repeated experiments ---------------------------------------------
library(BDgraph)
library(BGSL)
library(tidyverse)

#Set dimensions
p    = 30
Nrep = 1
n    = 500
M    = p/2

niter     = 400000
nburn     = niter/2
thin      =      1
thinG     =      1
data("examples")
True_val_rep = examples$Trueval_p30_rep
Test_p30_rep = vector("list", length = 3)
Nsim = length(True_val_rep)
names(Test_p30_rep) = c("BDRJ","BDgraph","DRJ")
Test_p30_rep$BDRJ = vector("list", length = Nsim)
Test_p30_rep$BDgraph = vector("list", length = Nsim)
Test_p30_rep$DRJ = vector("list", length = Nsim)

for(i in 1:Nsim){

  file_name = paste0("GGM_n500p30_BDRJ_",i)
  file_name_ext = paste0(file_name,".h5")
  cat('\n inizio BDRJ rep = ',i,'\n')
  True_val = True_val_rep[[i]]
  data = True_val$data
  Gcomp = True_val$Gcomp
  G = True_val$G
  Ktrue = True_val$Ktrue
  #Block DRJ
  #Parameters and hyperparameters
  threshold =  1e-14
  hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
  param = sampler_parameters(threshold = threshold)
  algo    = "DRJ"
  prior   = "Uniform"
  form    = "Block"

  #initial values
  init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

  #Run
  result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                        Param = param, HyParam = hy, Init = init,
                        prior = prior, form = form, algo = algo, file_name = file_name,
                        groups = NULL, n_groups = M, seed = 123, print_info = F)


  # Read
  info = Read_InfoFile(file_name = file_name_ext)
  #1) plinks
  plinks = result$plinks
  #2) bfdr selection
  an = BFDR_selection(plinks, min_rate = 0.05, diag = T)
  threshold = an$best_treshold
  #3) complete form
  PL = matrix(0,M,M)
  PL[plinks >= threshold] = 1
  Gfinal_est = Block2Complete(PL, CreateGroups(p,M))
  #4) misclassified links
  diff = abs(Gcomp - Gfinal_est)
  diff[lower.tri(diff)] = 0
  errors = sum(diff)
  #5) Table
  Gcomp_vett = Gcomp[upper.tri(Gcomp)]
  Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
  Table = table(Gcomp_vett, Gfinal_est_vett)
  TP = Table[2,2]
  FP = Table[1,2]
  FN = Table[2,1]
  TN = Table[1,1]
  Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN,
                   "errors"=errors, "std_hamming"=errors/(p*(p-1)))
  #6) Save
  save_iter = list()
  save_iter[[1]] = Table_res
  save_iter[[2]] = Gfinal_est
  names(save_iter) = c( "Table_res","Gfinal")
  Test_p30_rep$BDRJ[[i]] = save_iter

  #BDgraph
  #run
  res = bdgraph(data = data, n = n, iter = niter, burnin = nburn, threshold = 1e-14)
  #1) plinks
  plinks_BD = plinks(res)
  #2) bfdr selection
  an_BD = BFDR_selection(plinks_BD, min_rate = 0.05, diag = F)
  threshold_BD = an_BD$best_treshold
  #3) complete form
  Gfinal_est_BD = matrix(0,p,p)
  Gfinal_est_BD[plinks_BD >= threshold_BD] = 1
  #4) misclassified links
  diff_BD = abs(Gcomp - Gfinal_est_BD)
  diff_BD[lower.tri(diff_BD)] = 0
  errors_BD = sum(diff_BD) - p
  #5) Table
  Gcomp_vett = Gcomp[upper.tri(Gcomp)]
  Gfinal_est_vett_BD = Gfinal_est_BD[upper.tri((Gfinal_est_BD))]
  Table_BD = table(Gcomp_vett, Gfinal_est_vett_BD)
  TP_BD = Table_BD[2,2]
  FP_BD = Table_BD[1,2]
  FN_BD = Table_BD[2,1]
  TN_BD = Table_BD[1,1]
  Table_res_BD = list("Table"=Table_BD, "TP"=TP_BD, "FP"=FP_BD, "FN"=FN_BD, "TN"=TN_BD,
                      "errors"=errors_BD, "std_hamming"=errors_BD/(p*(p-1)))

  #6) Save
  save_iter = list()
  save_iter[[1]] = Table_res_BD
  save_iter[[2]] = Gfinal_est_BD
  names(save_iter) = c("Table_res","Gfinal")
  Test_p30_rep$BDgraph[[i]] = save_iter


  #DRJ
  file_name = paste0("GGM_n500p30_DRJ_",i)
  file_name_ext = paste0(file_name,".h5")
  cat('\n inizio DRJ rep = ',i,'\n')



  #Parameters and hyperparameters
  threshold =  1e-14
  hy = GM_hyperparameters(p = p, sigmaG = 0.5, D_K = diag(p) )
  param = sampler_parameters(threshold = threshold)
  algo    = "DRJ"
  prior   = "Uniform"
  form    = "Complete"

  #initial values
  init = GM_init(p = p, n = n, empty = T, form = form)

  #Run
  result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                        Param = param, HyParam = hy, Init = init,
                        prior = prior, form = form, algo = algo, file_name = file_name,
                        groups = NULL, seed = 123, print_info = F)


  info = Read_InfoFile(file_name = file_name_ext)
  #1) plinks
  plinks = result$plinks
  #2) bfdr selection
  an = BFDR_selection(plinks,min_rate = 0.05, diag = T)
  threshold = an$best_treshold
  #3) complete form
  PL = matrix(0,p,p)
  PL[plinks >= threshold] = 1
  Gfinal_est = PL
  #4) misclassified links
  diff = abs(Gcomp - Gfinal_est)
  diff[lower.tri(diff)] = 0
  errors = sum(diff)
  #5) Table
  Gcomp_vett = Gcomp[upper.tri(Gcomp)]
  Gfinal_est_vett = Gfinal_est[upper.tri((Gfinal_est))]
  Table = table(Gcomp_vett, Gfinal_est_vett)
  TP = Table[2,2]
  FP = Table[1,2]
  FN = Table[2,1]
  TN = Table[1,1]
  Table_res = list("Table"=Table, "TP"=TP, "FP"=FP, "FN"=FN, "TN"=TN, "errors"=errors, "std_hamming"=errors/(p*(p-1)))
  #6) Save
  save_iter = list()
  save_iter[[1]] = Table_res
  save_iter[[2]] = Gfinal_est
  names(save_iter) = c("Table_res","Gfinal")
  Test_p30_rep$DRJ[[i]] = save_iter
}




# p30 repeated experiments - plot -----------------------------------------------------------

Test_p30_res_rep = tibble(TP = as.integer(), TN = as.integer(),
                          FP = as.integer(), FN = as.integer(),
                          SHD = as.numeric(), F1score = as.numeric(),
                          type = as.factor(NULL) )


for(i in 1:Nsim){
  par_res = tibble("TP"=c(Test_p30_rep$BDRJ[[i]]$Table_res$TP,
                          Test_p30_rep$BDgraph[[i]]$Table_res$TP,
                          Test_p30_rep$DRJ[[i]]$Table_res$TP),
                   "TN"=c(Test_p30_rep$BDRJ[[i]]$Table_res$TN,
                          Test_p30_rep$BDgraph[[i]]$Table_res$TN,
                          Test_p30_rep$DRJ[[i]]$Table_res$TN),
                   "FP"=c(Test_p30_rep$BDRJ[[i]]$Table_res$FP,
                          Test_p30_rep$BDgraph[[i]]$Table_res$FP,
                          Test_p30_rep$DRJ[[i]]$Table_res$FP),
                   "FN"=c(Test_p30_rep$BDRJ[[i]]$Table_res$FN,
                          Test_p30_rep$BDgraph[[i]]$Table_res$FN,
                          Test_p30_rep$DRJ[[i]]$Table_res$FN),
                   "SHD"=c(Test_p30_rep$BDRJ[[i]]$Table_res$std_hamming,
                           Test_p30_rep$BDgraph[[i]]$Table_res$std_hamming,
                           Test_p30_rep$DRJ[[i]]$Table_res$std_hamming),
                   "F1score" = (2*TP)/(2*TP+FP+FN),
                   "type" = as.factor(c("BDRJ","BDgraph","DRJ"))
  )
  Test_p30_res_rep = Test_p30_res_rep %>% rbind(par_res)
}

data = Test_p30_res_rep

# SHD
data %>% group_by(type) %>%summarise(median(SHD))
data %>% group_by(type) %>%summarise(sd(SHD))

# F1 score
data %>% group_by(type) %>%summarise(median(F1score))
data %>% group_by(type) %>%summarise(sd(F1score))

# reorder labels for plot
data$type <- factor(data$type, levels = c("BDRJ", "DRJ", "BDgraph"))


# SHD
data %>% mutate(type = as.factor(type)) %>%
  ggplot(aes(y=SHD, x=type, fill=type)) + geom_boxplot() +
  labs(y="Standardized - SHD", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25)) +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))
#ylim(0,1)

# F1 SCORE
data %>% mutate(type = as.factor(type)) %>%
  ggplot(aes(y=F1score, x=type, fill=type)) + geom_boxplot() +
  labs(y="F1 - score", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25))  +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))
#ylim(0,1)



# SENS
data %>% mutate(type = as.factor(type)) %>%
  mutate(Sens = TP/(TP+FN)) %>%
  ggplot(aes(y=Sens, x=type, fill=type)) + geom_boxplot() +
  labs(y="Sensibility", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25))  +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))


# SPEC
data %>% mutate(type = as.factor(type)) %>%
  mutate(Spec = TN/(TN+FP)) %>%
  ggplot(aes(y=Spec, x=type, fill=type)) + geom_boxplot() +
  labs(y="Specificity", x = " ") + theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.position="none",
        text = element_text(size = 25))  +
  scale_fill_manual(values = c("forestgreen", "deepskyblue3", "darkred"))




# Application ------------------------------------------------------------

library(BGSL)
library(plot.matrix)
library(fields)
library(latex2exp)


# Load data
data("purees")
data = purees$data
wavelengths = as.numeric(purees$wavelengths[,1])

# Set dimensions
p =  40
n = dim(data)[1]
r = dim(data)[2]
range_x = range(wavelengths)
basis = Generate_Basis(n_basis = p, range = range_x, grid_points = wavelengths, order =3)
BaseMat = basis$BaseMat

# Bspline knots
nodes= rep(0,p+1)
nodes = c(wavelengths[1],
          wavelengths[1] + (basis$InternalKnots[1]-wavelengths[1])/2,
          basis$InternalKnots,
          wavelengths[length(wavelengths)] - (wavelengths[length(wavelengths)]-basis$InternalKnots[length(basis$InternalKnots)])/2,
          wavelengths[length(wavelengths)])
A = round(nodes[1:p],digits = 1)
B = round(nodes[2:(p+1)],digits = 1)
names = rep("",p)
for(i in 1:p){
  names[i] = paste0(A[i],"-",B[i])
}


# plot
plot_curves(data1 = data, n_plot = 351, internal_knots = basis$InternalKnots,
            range = range(wavelengths), grid_points = wavelengths)

# niter and groups
niter     =   500000
nburn     =    50000
thin      =     5000
thinG     =       50

groups = list(0:3, 4:5, 6:8, 9:12, 13:17, 18:21, 22:27, 28:32, 33:39)

n_groups = length(groups)

#Set parameters and hyperparameters
sigmaG  =    1.0
threshold =  1e-15
hy = GM_hyperparameters(p = p, sigmaG = sigmaG, D = 1*diag(1,p), Gprior = 0.25)
param = sampler_parameters(threshold = threshold, BaseMat = BaseMat)
algo    = "DRJ"
prior   = "Bernoulli"
form    = "Block"

#Set initial values
init    = GM_init(p = p, n = n, empty = T, form = form, groups = groups, n_groups = n_groups)

#Run
file_name = "FGMpurees"
file_name_ext = paste0(file_name, ".h5")
result = FGM_sampling(p = p, data = t(data), niter = niter, burnin = nburn, thin = thin, thinG = thinG,
                      Param = param, HyParam = hy, Init = init, file_name = file_name, form = form,
                      prior = prior, algo = algo, groups = groups, n_groups = n_groups, print_info = F,
                      seed = 23242526)

#0) read info
info = Read_InfoFile(file_name_ext)
#1) get results and compute quantiles
Beta_est   = result$PosteriorMeans$MeanBeta
K_est      = result$PosteriorMeans$MeanK

plinks = result$GraphAnalysis$plinks
plinks
#2) bfdr selection
an = BFDR_selection(plinks,tol = seq(0.75, 1, by = 0.0000001), diag = T)
threshold = an$best_treshold
#3) complete form
PL = matrix(0,n_groups,n_groups)
PL[plinks >= threshold] = 1
Gfinal_est = Block2Complete(PL, groups = groups)
Gfinal_est = Gfinal_est + t(Gfinal_est)
diag(Gfinal_est) = rep(1,p)

# plot
par(mfrow = c(2,2), mar = c(1,1,1,1))
ACheatmap(
  Gfinal_est,
  use_x11_device = F,
  horizontal = F,
  main = file_name,
  center_value = NULL,
  col.center = "darkolivegreen",
  col.upper = "grey50",
  col.lower = "white"
)































