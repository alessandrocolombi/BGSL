#Testing how to read from file
library(BDgraph)
library(plot.matrix)
library(fields)
library(BGSL)

file_name_ext = "FGMtest.h5"

#0) read info
info = Read_InfoFile(file_name_ext)
p = info$p
n = info$n
r = 250
range_x = c(100,200)
BaseMat = Generate_Basis(n_basis = p, range = range_x, n_points = r)$BaseMat
#1) Functional analysis -------------------------------------------------
#1.1) get results and compute quantiles
post_means = Compute_PosteriorMeans(file_name = file_name_ext, Beta = T, Mu = T, TauEps = T)

Beta_est   = post_means$MeanBeta
mu_est     = post_means$MeanMu
taueps_est = post_means$MeanTaueps

quantiles = Compute_Quantiles(file_name = file_name_ext,
                              Mu = T, Beta = T, TauEps = T )#quantiles computation of Beta takes some minutes

#1.2) plot smoothed_curves
n_plot = 2
Y_smo = smooth_curves(beta = Beta_est, BaseMat = BaseMat, n_plot = 10, range = range_x)


#1.3) plot curves with credible bands
Y_bands = smooth_curves_credible_bands(beta = Beta_est, betaLower = quantiles$Beta$Lower, 
                                       betaUpper = quantiles$Beta$Upper, range = range_x, 
                                       BaseMat = BaseMat, n_plot = 1 )


#1.4) analysis for mu
mu_low = quantiles$Mu$Lower
mu_upp = quantiles$Mu$Upper
x_grid = seq(range_x[1], range_x[2], length.out = p)
x_grid = round(x_grid, digits = 1)

x11();plot(mu_est, type = 'p', pch = 16, ylab = " ", xlab = "x", xaxt = 'n')
title(main = "Summary of mu coefficients")
mtext(text = x_grid, side = 1, line = 0.3, at = 1:length(x_grid), las = 2, cex = 0.9)
segments(x0 = 1:length(mu_est), y0 = mu_low, x1 = 1:length(mu_est), y1 = mu_upp,
         col = 'grey50', lwd = 4)
legend("topleft", legend = c("0.95 credible interval", "mean value"),
       col = c('grey50', 'black'), lty = c(1,1,1), lwd = 3)

# traceplot
index = 5
mu_chain = Extract_Chain(file_name = file_name_ext, variable = "Mu", index1 = index)

x11();plot(mu_chain, type = 'l', col = 'grey50', ylab = " ", xlab = "iteration")
abline(h = mu_est[index], lty = 2, col = 'green', lwd = 2)
abline(h = mu_low[index], lty = 1, col = 'black', lwd = 2)
abline(h = mu_upp[index], lty = 1, col = 'black', lwd = 2)
title(main = "Summary for Mu")
legend("topright", legend = c("0.95 quantiles", "mean value"),
       col = c('black', 'green'),  lty = c(2,1,2), lwd = 3)

#1.5) Beta analysis
Beta_low = quantiles$Beta$Lower
Beta_upp = quantiles$Beta$Upper
chain_example = Extract_Chain(file_name = file_name_ext, variable =  "Beta", index1 = 1, index2 = 1)


x11();plot(chain_example, type = 'l', col = 'grey50', ylab = " ", xlab = "iteration")
abline(h = Beta_est[1,1], lty = 2, col = 'green', lwd = 2)
abline(h = Beta_low[1,1], lty = 1, col = 'black', lwd = 2)
abline(h = Beta_upp[1,1], lty = 1, col = 'black', lwd = 2)
title(main = "Summary for Beta(1,1)")
legend("topright", legend = c( "0.95 quantiles", "mean value"),
       col = c('black', 'green'),  lty = c(2,1,2), lwd = 3)


#1.6) Analysis for tau_eps
taueps_chain = Extract_Chain(file_name = file_name_ext, variable = "TauEps")
taueps_low = quantiles$TauEps$Lower
taueps_upp = quantiles$TauEps$Upper
d = density(taueps_chain)
x11();plot(d, ylab = " ", xlab = " ", col = 'grey50', lwd = 3, main = " ")
title(main = "Summary for taueps")
abline(v = taueps_est, lty = 2, col = 'green', lwd = 2)
abline(v = taueps_low, lty = 1, col = 'black', lwd = 2)
abline(v = taueps_upp, lty = 1, col = 'black', lwd = 2)
legend("topright", legend = c("mean value", "0.95 quantiles" ),
       col = c('green', 'black'),  lty = c(2,2,1), lwd = 3)



# 2)  Graphical analysis --------------------------------------------------

graph_analysis = Summary_Graph(file_name = file_name_ext, groups = CreateGroups(p,p/2))

#2.1) traceplot of graph size
traceplot =graph_analysis$TracePlot_Gsize
x11();plot(traceplot, type = 'l', xlab = "Iteration", ylab = "Visited graph size ")
title(main = "Trace of graph size")

plinks = graph_analysis$plinks
#2.2) plot
x11();plot(plinks)
#2) bfdr selection
an = BFDR_selection(plinks,diag = T)
threshold = an$best_treshold
x11();plot(an$best_truncated_graph)
#3) complete form
PL = matrix(0,p/2,p/2)
PL[plinks >= threshold] = 1
Gfinal_est = Block2Complete(PL, CreateGroups(p,p/2))
x11();plot(Gfinal_est)
