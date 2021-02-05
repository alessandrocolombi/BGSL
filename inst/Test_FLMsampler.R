#Test FLMsampler
library(BDgraph)
library(plot.matrix)
library(BGSL)

#Define dimensions
p = 20
n = 500
r = 250
range_x = c(100,200)

#Simulate dataset
tau_eps = 1000
sim = simulate_curves(p = p, n = n, r = r, range_x = range_x,
                      G = NULL, n_plot = n, n_picks = 2,
                      tau_eps = tau_eps)
data = t(sim$data)

#Set parameters and hyperparameters
BaseMat = sim$basemat
init = LM_init(p = p, n = n, empty = T)
hy = LM_hyperparameters(p = p)
param = sampler_parameters(BaseMat = BaseMat)

#Run
niter = 50000
nburn = 25000
thin  =     1
file_name = "FLMtest"
file_name_ext = paste0(file_name, ".h5")
?FLM_sampling
res = FLM_sampling(p = p, data = data, niter = niter, burnin = nburn, thin = thin, diagonal_graph = T,
                   Param = param, HyParam = hy, Init = init, file_name = file_name, G = NULL,
                   print_info =  T, seed = 0) #Runtime is about 2 minutes


#0) read info 
info = Read_InfoFile(file_name_ext)
#1) get results and compute quantiles
Beta_est = res$MeanBeta
mu_est = res$MeanMu
tauK_est = res$MeanTauK
taueps_est = res$MeanTaueps

quantiles = Compute_Quantiles(file_name = file_name_ext, Precision = T, Mu = T,
                              Beta = T, TauEps = T )#quantiles computation of Beta takes some minutes

#2) plot smoothed_curves
n_plot = 2
Y_smo = smooth_curves(beta = Beta_est, BaseMat = BaseMat, n_plot = 0, range = range_x)
plot_curves(data1 = sim$data, data2 = Y_smo, range = range_x, n_plot = n_plot,
            legend_name1 = "True curves", legend_name2 = "Smoothed curves")

#3) plot curves with credible bands
Y_bands = smooth_curves_credible_bands(beta = Beta_est, betaLower = quantiles$Beta$Lower, 
                                       betaUpper = quantiles$Beta$Upper, range = range_x, 
                                       BaseMat = BaseMat, n_plot = 1, data = sim$data )


#4) analysis for mu
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


#5) TauK analysis
tauK_low  = quantiles$Precision$Lower
tauK_upp  = quantiles$Precision$Upper
tauK_true = diag(sim$K)

x11();plot(tauK_est, type = 'p', pch = 16, ylab = " ", xlab = "x", xaxt = 'n', ylim = c(min(tauK_low),max(tauK_upp)))
title(main = "Summary of TauK coefficients")
mtext(text = x_grid, side = 1, line = 0.3, at = 1:length(x_grid), las = 2, cex = 0.9)
points(tauK_true, type = 'p', pch = 16, col = 'red')
segments(x0 = 1:length(tauK_upp), y0 = tauK_low, x1 = 1:length(tauK_upp), y1 = tauK_upp,
         col = 'grey50', lwd = 4)
legend("topleft", legend = c("true value", "0.95 credible interval", "mean value"),
       col = c('red','grey50', 'black'), lty = c(1,1,1), lwd = 3)




#6) Beta analysis
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
       col = c('red','black', 'green'),  lty = c(2,1,2), lwd = 3)


Beta_true = t(sim$beta)
Beta_error = abs(Beta_true - Beta_est)

x11();layout(matrix(c(1,2,3),nrow = 1, ncol = 3, byrow = T))
fields::image.plot(x = 1:p, y = 1:n, (Beta_true),xlab = "Bands", ylab = "Curves", graphics.reset = TRUE)
title(main = "True Values for beta coefficients")
fields::image.plot(x = 1:p, y = 1:n, (Beta_est),xlab = "Bands", ylab = "Curves", graphics.reset = TRUE)
title(main = "Mean Values for beta coefficients (Gdiag)")
fields::image.plot(x = 1:p, y = 1:n, (Beta_error),xlab = "Bands", ylab = "Curves", graphics.reset = TRUE)
title(main = "Error")




#7) Analysis for tau_eps
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


