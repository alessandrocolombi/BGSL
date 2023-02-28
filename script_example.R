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

niter     = 40#0000
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
an = BFDR_selection(plinks, diag = T)
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
an_BD = BFDR_selection(plinks_BD, diag = F)
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
an = BFDR_selection(plinks, diag = T)
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


# p30 - experiment 2 ----------------------------------------------------------
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

niter     = 40#0000
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
an = BFDR_selection(plinks, diag = T)
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
an_BD = BFDR_selection(plinks_BD, diag = F)
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
an = BFDR_selection(plinks, diag = T)
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

