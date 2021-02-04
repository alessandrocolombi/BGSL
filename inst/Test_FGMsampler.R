#Test FGMsampler
library(BDgraph)
library(plot.matrix)
library(fields)
library(BGSL)

#Define dimensions
p =  20
n_groups =  10
n = 500
r = 250
range_x = c(100,200)

#Create random graph 
form   = "Block"
Glist  = Create_RandomGraph(p = p, n_groups = n_groups, form = form,
                            groups = CreateGroups(p,n_groups), sparsity = 0.25, seed = 2709)
Gblock = Glist$G
Gcomp  = Glist$G_Complete

#Simulate dataset
tau_eps = 200
sim = simulate_curves(p = p, n = n, r = r, range_x = range_x, D = 0.1*diag(p),
                      G = Gcomp, K = NULL, tau_eps = tau_eps,
                      n_plot = n/2, n_picks = 2 )
data = t(sim$data)

#Set parameters and hyperparameters
BaseMat = sim$basemat
sigmaG  =    0.5
threshold =  1e-14
hy = GM_hyperparameters(p = p, sigmaG = sigmaG)
param = sampler_parameters(threshold = threshold, BaseMat = BaseMat)
algo    = "DRJ"
prior   = "Uniform"
form    = "Block"

#Set initial values
init = GM_init(p = p, n = n, empty = T, form = form, n_groups = n_groups)


#Run
niter     =   600000
nburn     =   200000
thin      =       20
thinG     =        1
file_name = "FGMtest"
file_name_ext = paste0(file_name, ".h5")
?FGM_sampling
result = FGM_sampling(p = p, data = data, niter = niter, burnin = nburn, thin = thin, thinG = thinG,
                      Param = param, HyParam = hy, Init = init, file_name = file_name, form = form,
                      prior = prior, algo = algo, n_groups = n_groups, print_info = T)



#0) read info
info = Read_InfoFile(file_name_ext)

#1) Functional analysis -------------------------------------------------
#1.1) get results and compute quantiles
Beta_est   = result$PosteriorMeans$MeanBeta
mu_est     = result$PosteriorMeans$MeanMu
taueps_est = result$PosteriorMeans$MeanTaueps

quantiles = Compute_Quantiles(file_name = file_name_ext, Precision = T, Mu = T,
                              Beta = T, TauEps = T )#quantiles computation of Beta takes some minutes

#1.2) plot smoothed_curves
n_plot = 2
Y_smo = smooth_curves(beta = Beta_est, BaseMat = BaseMat, n_plot = 0, range = range_x)
plot_curves(data1 = sim$data, data2 = Y_smo, range = range_x, n_plot = n_plot,
            legend_name1 = "True curves", legend_name2 = "Smoothed curves")

#1.3) plot curves with credible bands
Y_bands = smooth_curves_credible_bands(beta = Beta_est, betaLower = quantiles$Beta$Lower, 
                                       betaUpper = quantiles$Beta$Upper, range = range_x, 
                                       BaseMat = BaseMat, n_plot = 1, data = sim$data )


#1.4) analysis for mu
mu_true = sim$mu
mu_low = quantiles$Mu$Lower
mu_upp = quantiles$Mu$Upper
x_grid = seq(range_x[1], range_x[2], length.out = p)
x_grid = round(x_grid, digits = 1)

x11()
plot(mu_est, type = 'p', pch = 16, ylab = " ", xlab = "x", xaxt = 'n')
title(main = "Summary of mu coefficients")
mtext(text = x_grid, side = 1, line = 0.3, at = 1:length(x_grid), las = 2, cex = 0.9)
points(mu_true, type = 'p', pch = 16, col = 'red')
segments(x0 = 1:length(mu_true), y0 = mu_low, x1 = 1:length(mu_true), y1 = mu_upp,
         col = 'grey50', lwd = 4)
legend("topleft", legend = c("true value", "0.95 credible interval", "mean value"),
       col = c('red','grey50', 'black'), lty = c(1,1,1), lwd = 3)

# traceplot
index = 5
mu_chain = Extract_Chain(file_name = file_name_ext, variable = "Mu", index1 = index)

x11();plot(mu_chain, type = 'l', col = 'grey50', ylab = " ", xlab = "iteration")
abline(h = sim$mu[index], lty = 2, col = 'red', lwd = 2)
abline(h = mu_est[index], lty = 2, col = 'green', lwd = 2)
abline(h = mu_low[index], lty = 1, col = 'black', lwd = 2)
abline(h = mu_upp[index], lty = 1, col = 'black', lwd = 2)
title(main = "Summary for Mu")
legend("topright", legend = c("true value", "0.95 quantiles", "mean value"),
       col = c('red','black', 'green'),  lty = c(2,1,2), lwd = 3)

#1.5) Beta analysis
Beta_low = quantiles$Beta$Lower
Beta_upp = quantiles$Beta$Upper
beta_true = sim$beta[1,1]
chain_example = Extract_Chain(file_name = file_name_ext, variable =  "Beta", index1 = 1, index2 = 1)


x11();plot(chain_example, type = 'l', col = 'grey50', ylab = " ", xlab = "iteration")
abline(h = beta_true, lty = 2, col = 'red', lwd = 2)
abline(h = Beta_est[1,1], lty = 2, col = 'green', lwd = 2)
abline(h = Beta_low[1,1], lty = 1, col = 'black', lwd = 2)
abline(h = Beta_upp[1,1], lty = 1, col = 'black', lwd = 2)
title(main = "Summary for Beta(1,1)")
legend("topright", legend = c("true value", "0.95 quantiles", "mean value"),
       col = c('grey50','black', 'green'),  lty = c(2,1,2), lwd = 3)


Beta_true = t(sim$beta)
Beta_error = abs(Beta_true - Beta_est)

x11();layout(matrix(c(1,2,3),nrow = 1, ncol = 3, byrow = T))
fields::image.plot(x = 1:p, y = 1:n, (Beta_true),xlab = "Bands", ylab = "Curves", graphics.reset = TRUE)
title(main = "True Values for beta coefficients")
fields::image.plot(x = 1:p, y = 1:n, (Beta_est),xlab = "Bands", ylab = "Curves", graphics.reset = TRUE)
title(main = "Mean Values for beta coefficients (Gdiag)")
fields::image.plot(x = 1:p, y = 1:n, (Beta_error),xlab = "Bands", ylab = "Curves", graphics.reset = TRUE)
title(main = "Error")


#1.6) Analysis for tau_eps
taueps_chain = Extract_Chain(file_name = file_name_ext, variable = "TauEps")
taueps_low = quantiles$TauEps$Lower
taueps_upp = quantiles$TauEps$Upper
d = density(taueps_chain)
x11();plot(d, ylab = " ", xlab = " ", col = 'grey50', lwd = 3, main = " ")
title(main = "Summary for taueps")
abline(v = tau_eps, lty = 2, col = 'red', lwd = 2)
abline(v = taueps_est, lty = 2, col = 'green', lwd = 2)
abline(v = taueps_low, lty = 1, col = 'black', lwd = 2)
abline(v = taueps_upp, lty = 1, col = 'black', lwd = 2)
legend("topright", legend = c("true value", "mean value", "0.95 quantiles" ),
       col = c('red','green', 'black'),  lty = c(2,2,1), lwd = 3)



# 2)  Graphical analysis --------------------------------------------------

#2.1) traceplot of graph size
traceplot = result$GraphAnalysis$TracePlot_Gsize
x11();plot(traceplot, type = 'l', xlab = "Iteration", ylab = "Visited graph size ")
title(main = "Trace of graph size")

plinks = result$GraphAnalysis$plinks
#2.2) plot
x11();layout(matrix(c(1,2),nrow = 1, ncol = 2, byrow = T))
plot(Gblock)
plot(plinks)
#2) bfdr selection
an = BFDR_selection(plinks,diag = T)
threshold = an$best_treshold
x11();layout(matrix(c(1,2),nrow = 1, ncol = 2, byrow = T))
plot(Gblock)
plot(an$best_truncated_graph)
#3) complete form
PL = matrix(0,n_groups,n_groups)
PL[plinks >= threshold] = 1
Gfinal_est = Block2Complete(PL, CreateGroups(p,n_groups))
x11();layout(matrix(c(1,2),nrow = 1, ncol = 2, byrow = T))
plot(Gcomp)
plot(Gfinal_est)
#4) misclassified links
diff = abs(Gcomp - Gfinal_est)
diff[lower.tri(diff)] = 0
sum(diff)
