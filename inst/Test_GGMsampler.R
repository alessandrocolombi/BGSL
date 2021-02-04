#Test GGMsampler
library(BDgraph)
library(fields)
library(plot.matrix)
library(BGSL)

#Define dimensions
p =  20
M =  10
n = 500

#Simulate data
sim   = SimulateData_GGM(p = p, n = n, form = "Block", n_groups = M, adj_mat = NULL, seed = 2709)
G     = sim$G_true
Gcomp = sim$G_complete
data  = sim$U

#Parameters and hyperparameters
threshold =  1e-14
hy = GM_hyperparameters(p = p, sigmaG = 0.5)
param = sampler_parameters(threshold = threshold)
algo    = "DRJ"
prior   = "Uniform"
form    = "Block"

#initial values
init = GM_init(p = p, n = n, empty = T, form = form, n_groups = M)

#Run
niter     =   100000
nburn     =    50000
thin      =        1
thinG     =        1
file_name = "GGMtest"
file_name_ext = paste0(file_name,".h5")
?GGM_sampling
result = GGM_sampling(data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                      Param = param, HyParam = hy, Init = init, 
                      prior = prior, form = form, algo = algo, file_name = file_name, 
                      groups = NULL, n_groups = M, seed = 0, print_info = TRUE) #Runtime is around 8 minutes


info = Read_InfoFile(file_name = file_name_ext)

plinks = result$plinks
#1) plot
x11();layout(matrix(c(1,2),nrow = 1, ncol = 2, byrow = T))
plot(G)
plot(plinks)
#2) bfdr selection
an = BFDR_selection(plinks,diag = T)
threshold = an$best_treshold
x11();layout(matrix(c(1,2),nrow = 1, ncol = 2, byrow = T))
plot(G)
plot(an$best_truncated_graph)
#3) complete form
PL = matrix(0,M,M)
PL[plinks >= threshold] = 1
Gfinal_est = Block2Complete(PL, CreateGroups(p,M))
x11();layout(matrix(c(1,2),nrow = 1, ncol = 2, byrow = T))
plot(Gcomp)
plot(Gfinal_est)
#4) misclassified links
diff = abs(Gcomp - Gfinal_est)
diff[lower.tri(diff)] = 0
sum(diff)





