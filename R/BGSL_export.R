#' Tuning the threshold for the GWishart sampler
#'
#' This function select possible threshold for the convergence of the matrix in the GWishart sampler. The sampler starts drawing a sampler from a Wishart
#' and then starts a loop for imposing all the zeros in the right positions. The loop has to be stopped according to some convergence criteria and this
#' that not always the structure of the graph is pefectly respected. This function propose some threshold for stopping the loop according to the percentage
#' of sampled matrix that have the right structure out of nrep trials.
#' Four possibilities are returned according to four possible levels, that are 0.95, 0.98, 0.99 and 1. If some level is not reach in the selected threshold level, then NA is returned
#' @param p The dimension of the graph in complete form
#' @param nrep How many repetitions has to be performed for each possible threshold
#' @param initial_threshold The first and larger threshold to be tested
#' @param max_threshold The last and smaller threshold to be tested
#' @param norm The norm with respect to which convergence happens
#'
#' @return A list with four possible threshold levels and the exepected number of iteration for each threshold
#' @export
GWish_sampler_tuning = function(p, nrep = 1000, initial_threshold = 1e-2, max_threshold = 1e-14, norm = "Mean"){
  if(nrep <= 0 || initial_threshold <= 0 || max_threshold <= 0)
    stop("Error, parameters has to be positive.");
  n = 200
  level95 = list("Threshold" = NA, "ExpectedIter" = NA)
  level98 = list("Threshold" = NA, "ExpectedIter" = NA)
  level99 = list("Threshold" = NA, "ExpectedIter" = NA)
  level1  = list("Threshold" = NA, "ExpectedIter" = NA)
  found_lv95 = F
  found_lv98 = F
  found_lv99 = F
  found_lv1  = F
  tr = initial_threshold
  while(!(found_lv1 & found_lv99  & found_lv98 & found_lv95) & tr >= max_threshold){
    b = 3
    D = diag(p)
    n_converged = 0
    n_structure = 0
    iterations = rep(0,nrep)
    for(i in 1:nrep){
      G = Create_RandomGraph(p, sparsity = 0.5)
      K = rGwish(G = G, b = b, D = D)$Matrix
      U = matrix(0, nrow = p, ncol = p)
      for (j in 1:n) {
        beta_i = rmvnormal(mean = rep(0,p), Mat = K)
        U = U + (beta_i) %*% t(beta_i)
      }
      PrecList = rGwish(G = G, b = b+n, D = D+U, norm = norm, threshold_conv = tr)
      n_converged = n_converged+PrecList$Converged
      n_structure = n_structure+PrecList$CheckStructure
      iterations[i] =PrecList$iterations
    }
    iterations = iterations[iterations > 0]
    iterations = sum(iterations)/length(iterations)
    n_converged = n_converged/nrep
    n_structure = n_structure/nrep
    if(!found_lv95 & n_structure >= 0.95){
      found_lv95 = T
      level95$Threshold = tr
      level95$ExpectedIter = iterations
    }
    if(!found_lv98 & n_structure >= 0.98){
      found_lv98 = T
      level98$Threshold = tr
      level98$ExpectedIter = iterations
    }
    if(!found_lv99 & n_structure >= 0.99){
      found_lv99 = T
      level99$Threshold = tr
      level99$ExpectedIter = iterations
    }
    if(!found_lv1 & n_structure >= 1){
      found_lv1 = T
      level1$Threshold = tr
      level1$ExpectedIter = iterations
    }
    cat('\n Finished with threshold = ',tr,'\n')
    tr = tr/10
  }
  return ( list("level95"=level95, "level98"=level98, "level99"=level99, "level1"=level1) )
}


#' Tuning MC iterations for prior normalizing constant of GWishart distribution
#'
#' This function selects the number of MonteCarlo iterations when computing a the GWishart prior normalizing constant. The problem is that the MonteCarlo
#' method by ATACK-MASSAM requires computing exp(-0.5 sum(Phi_ij)^2). As the number of nodes in the graph grows, that quantity is smaller and smaller and
#' it is possible that it is lower than zero machina. If that is the case, the number of MC iterarions needs to be larger.
#' This function propose some possible MC iterations loop according to the percentage of computed constant different from -Inf out of nrep trials.
#' Four possibilities are returned according to four possible levels, that are 0.95, 0.98, 0.99 and 1. If some level is not reach in the selected threshold level, then NA is returned
#'
#' @param p The dimension of the graph in complete form
#' @param b Shape parameter. It has to be larger than 2 in order to have a well defined distribution.
#' @param D Inverse scale matrix. It has to be symmetric and positive definite.
#' @param nrep How many repetitions has to be performed for each possible threshold
#' @param MCiter_list The list containg the Monte Carlo iterations to be taken into consideration
#'
#' @return A list with four possible MCiteration levels
#' @export
GWish_PriorConst_tuning = function(p, b=3, D = diag(p), nrep = 100, MCiter_list = c(100,250,500,1000,2500,5000,10000)){
  if(nrep <= 0 )
    stop("Error, parameters has to be positive.");
  if(!isSymmetric(D))
    stop("Inverse scale matrix D has to the symmetric.")
  level95 = list("MCiterations" = NA)
  level98 = list("MCiterations" = NA)
  level99 = list("MCiterations" = NA)
  level1  = list("MCiterations" = NA)
  found_lv95 = F
  found_lv98 = F
  found_lv99 = F
  found_lv1  = F

  counter = 1
  while(!(found_lv1 & found_lv99  & found_lv98 & found_lv95) & counter <= length(MCiter_list) ){

    MCiter = MCiter_list[counter]
    result = rep(0,nrep)
    for(i in 1:nrep){
      G = Create_RandomGraph(p, sparsity = 0.5)
      result[i] = log_Gconstant(G = G, b = b, D = D, MCiteration = MCiter)
    }
    count_inf = length(result[result == -Inf])
    count_inf = count_inf/nrep
    count_inf = 1 - count_inf
    if(!found_lv95 & count_inf >= 0.95){
      found_lv95 = T
      level95$MCiterations = MCiter
    }
    if(!found_lv98 & count_inf >= 0.98){
      found_lv98 = T
      level98$MCiterations = MCiter
    }
    if(!found_lv99 & count_inf >= 0.99){
      found_lv99 = T
      level99$MCiterations = MCiter
    }
    if(!found_lv1 & count_inf >= 1){
      found_lv1 = T
      level1$MCiterations = MCiter
    }
    cat('\n Finished with MCiter = ',MCiter,'\n')
    counter = counter + 1
  }
  return ( list("level95"=level95, "level98"=level98, "level99"=level99, "level1"=level1) )
}


#' Tuning MC iterations for posterior normalizing constant of GWishart distribution
#'
#' This function selects the number of MonteCarlo iterations when computing a the GWishart posterior normalizing constant. The problem is that the MonteCarlo
#' method by ATACK-MASSAM requires computing exp(-0.5 sum(Phi_ij)^2). As the number of nodes in the graph grows, that quantity is smaller and smaller and
#' it is possible that it is lower than zero machina. If that is the case, the number of MC iterarions needs to be larger.
#' This function propose some possible MC iterations loop according to the percentage of computed constant different from -Inf out of nrep trials.
#' Four possibilities are returned according to four possible levels, that are 0.95, 0.98, 0.99 and 1. If some level is not reach in the selected threshold level, then NA is returned
#'
#' @param p The dimension of the graph in complete form
#' @param n The number of observations
#' @param b Shape parameter. It has to be larger than 2 in order to have a well defined distribution.
#' @param D Inverse scale matrix. It has to be symmetric and positive definite.
#' @param nrep How many repetitions has to be performed for each possible threshold
#' @param MCiter_list The list containg the Monte Carlo iterations to be taken into consideration
#'
#' @return A list with four possible MCiteration levels
#' @export
GWish_PostConst_tuning = function(p, n, b=3, D = diag(p), nrep = 100, MCiter_list = c(100,250,500,1000,2500,5000,10000)){
  if(nrep <= 0 )
    stop("Error, parameters has to be positive.")
  if(b <= 2)
    stop("Error, shape parametes has to be larger than 2.");
  if(!isSymmetric(D))
    stop("Inverse scale matrix D has to the symmetric.")
  level95 = list("MCiterations" = NA)
  level98 = list("MCiterations" = NA)
  level99 = list("MCiterations" = NA)
  level1  = list("MCiterations" = NA)
  found_lv95 = F
  found_lv98 = F
  found_lv99 = F
  found_lv1  = F

  counter = 1
  while(!(found_lv1 & found_lv99  & found_lv98 & found_lv95) & counter <= length(MCiter_list) ){

    MCiter = MCiter_list[counter]
    result = rep(0,nrep)
    for(i in 1:nrep){
      G = Create_RandomGraph(p, sparsity = 0.5)
      K = rGwish(G = G, b = b, D = D)$Matrix
      #K_inv = solve(K)
      U = matrix(0, nrow = p, ncol = p)
      for (j in 1:n) {
        beta_i = rmvnormal(mean = rep(0,p), Mat = K)
        U = U + (beta_i) %*% t(beta_i)
      }
      result[i] = log_Gconstant(G = G, b = b + n, D = D + U, MCiteration = MCiter)
    }
    count_inf = length(result[result == -Inf])
    count_inf = count_inf/nrep
    count_inf = 1 - count_inf
    if(!found_lv95 & count_inf >= 0.95){
      found_lv95 = T
      level95$MCiterations = MCiter
    }
    if(!found_lv98 & count_inf >= 0.98){
      found_lv98 = T
      level98$MCiterations = MCiter
    }
    if(!found_lv99 & count_inf >= 0.99){
      found_lv99 = T
      level99$MCiterations = MCiter
    }
    if(!found_lv1 & count_inf >= 1){
      found_lv1 = T
      level1$MCiterations = MCiter
    }
    cat('\n Finished with MCiter = ',MCiter,'\n')
    counter = counter + 1
  }
  return ( list("level95"=level95, "level98"=level98, "level99"=level99, "level1"=level1) )
}


#' Computes and plot smoothed curves
#'
#' This function gets the regression coefficients and the evaluated splines and builds the smooted curves.
#' @param beta matrix of dimension \eqn{n_basis \times n_curves} containing the values of regression coefficients.
#' @param BaseMat matrix of dimension \eqn{n_grid_points \times n_basis} containing the evaluation of all the spline in all the grid points.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param range the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.
#' @param grid_points vector of size \code{n_grid_points} with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis}}. Default is null.
#' but if provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \eqn{[1,n_basis]}
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \eqn{[1,n_basis]}
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#'
#' @export
smooth_curves = function( beta, BaseMat, n_plot = 0, range = NULL, grid_points = NULL,
                          internal_knots = NULL, highlight_band1 = NULL, highlight_band2 = NULL,
                          title_plot = "Smoothed Curves", xtitle = " ", ytitle = " ")
{
  if (!is.matrix(beta)) {
    stop("Beta should be a (p x n) matrix")
  }
  p = dim(beta)[1]
  n = dim(beta)[2]
  r = dim(BaseMat)[1]
  if( dim(BaseMat)[2] != p)
    stop("Inchoerent dimensions. The size of each beta is not equal to the number of basis")
  #Compute smoothed valules
  result = matrix(0, nrow = n, ncol = r)
  for (i in 1:n) {
    y_mean = BaseMat %*% beta[,i]
    result[i, ] = y_mean
  }
  #Plot
  if(n_plot > 0) #Plot n_plot curves
  {
    #Check dimensions
    if(is.null(range))
      stop("The range has to be provided in order to plot the curves.")
    if(!(length(range)==2 && range[1] < range[2]))
      stop("Invalid range, it has to be a vector of length 2 containing first the lower bound of the interval and then the upper bound.")
    #Computes grid_points
    if(!is.null(grid_points)){
      if(grid_points != r)
        stop("The number of points provided in grid_points is not equal to the size of BaseMat.")
      X = grid_points;
    }else{
      X = seq(range[1], range[2], length.out = r)
    }
    #Classical plot, does not depend on the size of the graph
    if(is.null(internal_knots)){
      if(n_plot > 1){
        x11(height=4)
        matplot( x = X, t(result[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                lwd = 3, ylim = c(min(result[1:n_plot,]),max(result[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
      }
      else if(n_plot == 1){
        x11(height=4)
        matplot( x = X, (result[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                lwd = 3, ylim = c(min(result),max(result)), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
      }
    }
    else{ #Plot with bands representing the domanin of the spline
        knots <- c(range[1],
                 range[1] + (internal_knots[1]-range[1])/2,
                 internal_knots,
                 range[2] - (range[2]-internal_knots[length(internal_knots)])/2,
                 range[2] )
        names <- rep("", length(knots))
        for (i in 1:length(knots)) {
          names[i] <- paste0(knots[i])
        }
        names_y = round(seq(min(result[1:n_plot,]), max(result[1:n_plot,]), length.out = 10), digits = 2)
        if(n_plot > 1){
          x11(height=4)
            matplot(x = X, t(result[1:n_plot,]), type = 'l', lty = 1,
                    col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                    lwd = 3, ylim = c(min(result[1:n_plot,]),max(result[1:n_plot,])), axes = F,
                    main = title_plot, xlab = xtitle,
                    ylab = ytitle)
            abline(v = knots, lty = 2, col = 'black')
                if(!is.null(highlight_band1)){
                  for(i in c(highlight_band1, highlight_band1[length(highlight_band1)]+1) )
                    abline(v = knots[i], lty = 2, col = 'red')
                }
                if(!is.null(highlight_band2)){
                  for(i in c(highlight_band2[1]-1,highlight_band2) )
                    abline(v = knots[i], lty = 2, col = 'red')
                }
            mtext(text = names, side=1, line=0.3, at = knots , las=2, cex=0.7)
            mtext(text = names_y, side=2, line=0.3, at=names_y, las=1, cex=0.9)
        }
        else if(n_plot == 1){
            x11(height=4)
              matplot(x = X, (result[1:n_plot,]), type = 'l', lty = 1,
                      col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                      lwd = 3, ylim = c(min(result[1:n_plot,]),max(result[1:n_plot,])), axes = F,
                      main = title_plot, xlab = xtitle,
                      ylab = ytitle)
              abline(v = knots, lty = 2, col = 'black')
                  if(!is.null(highlight_band1)){
                    for(i in c(highlight_band1, highlight_band1[length(highlight_band1)]+1) )
                      abline(v = knots[i], lty = 2, col = 'red')
                  }
                  if(!is.null(highlight_band2)){
                    for(i in c(highlight_band2[1]-1,highlight_band2) )
                      abline(v = knots[i], lty = 2, col = 'red')
                  }
              mtext(text = names, side=1, line=0.3, at = knots , las=2, cex=0.7)
              mtext(text = names_y, side=2, line=0.3, at=names_y, las=1, cex=0.9)
        }
    }
  }
  return(result)
}

#' Computes and plot smoothed curves with credible bands
#'
#' This function gets the mean values of the regression coefficients as well as the lower and upper quantiles and builds the smooted curves with their credible bands.
#' It does not computes the quantiles nor the mean starting from all the sampled values, they have to be previously computed using \code{\link{Compute_QuantileBeta}} function.
#' @param beta matrix of dimension \eqn{n_basis \times n_curves} containing the mean values of regression coefficients
#' @param betaLower matrix of dimension \eqn{n_basis \times n_curves} containing the lower quantiles values of regression coefficients. Can be obtained by \code{\link{Compute_QuantileBeta}} function.
#' @param betaUpper matrix of dimension \eqn{n_basis \times n_curves} containing the upper quantiles values of regression coefficients. Can be obtained by \code{\link{Compute_QuantileBeta}} function.
#' @param BaseMat matrix of dimension \eqn{n_grid_points \times n_basis} containing the evaluation of all the spline in all the grid points.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param range the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.
#' @param grid_points vector of size \code{n_grid_points} with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis.}} Default is null.
#' but if provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \eqn{[1,n_basis]}.
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \eqn{[1,n_basis]}.
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#'
#' @export
smooth_curves_credible_bands = function( beta, betaLower, betaUpper, BaseMat, n_plot = 0, range = NULL, grid_points = NULL,
                                          internal_knots = NULL, highlight_band1 = NULL, highlight_band2 = NULL,
                                          title_plot = "Smoothed Curves", xtitle = " ", ytitle = " ")
{
  if (!is.matrix(beta)) {
    stop("Beta should be a (p x n) matrix")
  }
  p = dim(beta)[1]
  n = dim(beta)[2]
  r = dim(BaseMat)[1]
  if( dim(BaseMat)[2] != p)
    stop("Inchoerent dimensions. The size of each beta is not equal to the number of basis")
  #Compute smoothed valules
  Ymean  = matrix(0, nrow = n, ncol = r)
  Ylower = matrix(0, nrow = n, ncol = r)
  Yupper = matrix(0, nrow = n, ncol = r)
  for (i in 1:n) {
    Ymean[i, ]  = BaseMat %*% beta[,i]
    Ylower[i, ] = BaseMat %*% betaLower[,i]
    Yupper[i, ] = BaseMat %*% betaUpper[,i]
  }
  #Plot
  if(n_plot > 0) #Plot n_plot curves
  {
    #Check dimensions
    if(is.null(range))
      stop("The range has to be provided in order to plot the curves.")
    if(!(length(range)==2 && range[1] < range[2]))
      stop("Invalid range, it has to be a vector of length 2 containing first the lower bound of the interval and then the upper bound.")
    #Computes grid_points
    if(!is.null(grid_points)){
      if(grid_points != r)
        stop("The number of points provided in grid_points is not equal to the size of BaseMat.")
      X = grid_points;
    }else{
      X = seq(range[1], range[2], length.out = r)
    }
    #Classical plot, does not depend on the size of the graph
    if(is.null(internal_knots)){
      if(n_plot > 1){
        x11(height=4)
        matplot( x = X, t(Ymean[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
        matplot( x = X, t(Ylower[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
        matplot( x = X, t(Yupper[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
      }
      else if(n_plot == 1){
        x11(height=4)
        matplot( x = X, (Ymean[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
        matplot( x = X, (Ylower[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
        matplot( x = X, (Yupper[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
      }
    }
    else{ #Plot with bands representing the domanin of the spline
        knots <- c(range[1],
                 range[1] + (internal_knots[1]-range[1])/2,
                 internal_knots,
                 range[2] - (range[2]-internal_knots[length(internal_knots)])/2,
                 range[2] )
        names <- rep("", length(knots))
        for (i in 1:length(knots)) {
          names[i] <- paste0(knots[i])
        }
        names_y = round(seq(min(Ymean[1:n_plot,]), max(Ymean[1:n_plot,]), length.out = 10), digits = 2)
        if(n_plot > 1){
          x11(height=4)
            matplot(x = X, t(Ymean[1:n_plot,]), type = 'l', lty = 1,
                    col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                    lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = F,
                    main = title_plot, xlab = xtitle,
                    ylab = ytitle)
            matplot( x = X, t(Ylower[1:n_plot,]), type = 'l', lty = 1,
                    col = c('gray50','gray70','gray40'), lwd = 3, add = T)
            matplot( x = X, t(Yupper[1:n_plot,]), type = 'l', lty = 1,
                    col = c('gray50','gray70','gray40'), lwd = 3, add = T)
            abline(v = knots, lty = 2, col = 'black')
                if(!is.null(highlight_band1)){
                  for(i in c(highlight_band1, highlight_band1[length(highlight_band1)]+1) )
                    abline(v = knots[i], lty = 2, col = 'red')
                }
                if(!is.null(highlight_band2)){
                  for(i in c(highlight_band2[1]-1,highlight_band2) )
                    abline(v = knots[i], lty = 2, col = 'red')
                }
            mtext(text = names, side=1, line=0.3, at = knots , las=2, cex=0.7)
            mtext(text = names_y, side=2, line=0.3, at=names_y, las=1, cex=0.9)
        }
        else if(n_plot == 1){
            x11(height=4)
              matplot(x = X, (Ymean[1:n_plot,]), type = 'l', lty = 1,
                      col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                      lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = F,
                      main = title_plot, xlab = xtitle,
                      ylab = ytitle)
              matplot( x = X, (Ylower[1:n_plot,]), type = 'l', lty = 1,
                      col = c('gray50','gray70','gray40'), lwd = 3, add = T)
              matplot( x = X, (Yupper[1:n_plot,]), type = 'l', lty = 1,
                      col = c('gray50','gray70','gray40'), lwd = 3, add = T)
              abline(v = knots, lty = 2, col = 'black')
                  if(!is.null(highlight_band1)){
                    for(i in c(highlight_band1, highlight_band1[length(highlight_band1)]+1) )
                      abline(v = knots[i], lty = 2, col = 'red')
                  }
                  if(!is.null(highlight_band2)){
                    for(i in c(highlight_band2[1]-1,highlight_band2) )
                      abline(v = knots[i], lty = 2, col = 'red')
                  }
              mtext(text = names, side=1, line=0.3, at = knots , las=2, cex=0.7)
              mtext(text = names_y, side=2, line=0.3, at=names_y, las=1, cex=0.9)
        }
    }
  }
  ret_list = list()
  ret_list[[1]] = Ylower
  ret_list[[2]] = Ymean
  ret_list[[3]] = Yupper
  names(ret_list) = c("LowerBands", "MeanCurves", "UpperBands")
  return(ret_list)
}

#' Plot curves
#'
#' This functions gets one or two dataset representig functional data and plot them. It does not smooth the curves, indeed it requires as input the data, not
#' the regression coefficients. Use \code{\link{smooth_curves}} function for that.
#' @param data1 matrix of dimension \eqn{n_curves \times n_grid_points} representing the first dataset to be plotted.
#' @param data2 matrix of dimension \eqn{n_curves \times n_grid_points} representing the second dataset to be plotted, if needed.
#' @param range the range where the curves has to be plotted. Not needed if\code{n_plot} is 0.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param grid_points vector of size n_grid_points with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis.}}. Default is null.
#' but if provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \eqn{[1,n_basis]}.
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \eqn{[1,n_basis]}.
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#' @param legend_name1 the name for \code{data1} to be printed in the legend. Used only is two datasets are actually plotted.
#' @param legend_name2 the name for \code{data2} to be printed in the legend.
#'
#' @export
plot_curves = function( data1, data2 = NULL, range, n_plot = 1, grid_points = NULL,
                        internal_knots = NULL, highlight_band1 = NULL, highlight_band2 = NULL,
                        title_plot = "Curves", xtitle = " ", ytitle = " ", legend_name1 = "data1", legend_name2 = "data2")
{
  #Plot
  if(n_plot > 0) #Plot n_plot curves
  {
    #Check dimensions
    if(!(length(range)==2 && range[1] < range[2]))
      stop("Invalid range, it has to be a vector of length 2 containing first the lower bound of the interval and then the upper bound.")
    #Computes grid_points
    if(!is.null(grid_points)){
      if(length(grid_points) != r)
        stop("The number of points provided in grid_points is not equal to the size of BaseMat.")
      X = grid_points;
    }else{
      X = seq(range[1], range[2], length.out = r)
    }
    #Classical plot, does not depend on the size of the graph
    if(is.null(internal_knots)){
      if(n_plot > 1){
        x11(height=4)
        matplot( x = X, t(data1[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                lwd = 3, ylim = c(min(data1[1:n_plot,]),max(data1[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
        if(!is.null(data2)){
            matplot( x = X, t(data2[1:n_plot,]), type = 'l', lty = 2,
            col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
            legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,2), lwd = 3)
        }
      }
      else if(n_plot == 1){
        x11(height=4)
        matplot( x = X, (data1[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                lwd = 3, ylim = c(min(data1[1:n_plot,]),max(data1[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
       if(!is.null(data2)){
           matplot( x = X, (data2[1:n_plot,]), type = 'l', lty = 2,
           col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
           legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,2), lwd = 3)
       }
      }
    }
    else{ #Plot with bands representing the domanin of the spline
        knots <- c(range[1],
                 range[1] + (internal_knots[1]-range[1])/2,
                 internal_knots,
                 range[2] - (range[2]-internal_knots[length(internal_knots)])/2,
                 range[2] )
        names <- rep("", length(knots))
        for (i in 1:length(knots)) {
          names[i] <- paste0(knots[i])
        }
        names_y = round(seq(min(data1[1:n_plot,]), max(data1[1:n_plot,]), length.out = 10), digits = 2)
        if(n_plot > 1){
          x11(height=4)
            matplot(x = X, t(data1[1:n_plot,]), type = 'l', lty = 1,
                    col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                    lwd = 3, ylim = c(min(data1[1:n_plot,]),max(data1[1:n_plot,])), axes = F,
                    main = title_plot, xlab = xtitle,
                    ylab = ytitle)
            if(!is.null(data2)){
                matplot( x = X, t(data2[1:n_plot,]), type = 'l', lty = 2,
                col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
                legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,2), lwd = 3)
            }
            abline(v = knots, lty = 2, col = 'black')
                if(!is.null(highlight_band1)){
                  for(i in c(highlight_band1, highlight_band1[length(highlight_band1)]+1) )
                    abline(v = knots[i], lty = 2, col = 'red')
                }
                if(!is.null(highlight_band2)){
                  for(i in c(highlight_band2[1]-1,highlight_band2) )
                    abline(v = knots[i], lty = 2, col = 'red')
                }
            mtext(text = names, side=1, line=0.3, at = knots , las=2, cex=0.7)
            mtext(text = names_y, side=2, line=0.3, at=names_y, las=1, cex=0.9)
        }
        else if(n_plot == 1){
            x11(height=4)
              matplot(x = X, (data1[1:n_plot,]), type = 'l', lty = 1,
                      col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'),
                      lwd = 3, ylim = c(min(data1[1:n_plot,]),max(data1[1:n_plot,])), axes = F,
                      main = title_plot, xlab = xtitle,
                      ylab = ytitle)
              if(!is.null(data2)){
                  matplot( x = X, (data2[1:n_plot,]), type = 'l', lty = 2,
                  col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
                  legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,2), lwd = 3)
              }
              abline(v = knots, lty = 2, col = 'black')
                  if(!is.null(highlight_band1)){
                    for(i in c(highlight_band1, highlight_band1[length(highlight_band1)]+1) )
                      abline(v = knots[i], lty = 2, col = 'red')
                  }
                  if(!is.null(highlight_band2)){
                    for(i in c(highlight_band2[1]-1,highlight_band2) )
                      abline(v = knots[i], lty = 2, col = 'red')
                  }
              mtext(text = names, side=1, line=0.3, at = knots , las=2, cex=0.7)
              mtext(text = names_y, side=2, line=0.3, at=names_y, las=1, cex=0.9)
        }
    }
  }
  else
    stop("The number of curves to be plotted has to be positive.")
}


#' Functional Linear model for smoothing
#'
#' This function performs a linear regression for functional data, according to model (INSERIRE FORMULA MODELLO).
#' It is not a graphical model, the graph has to be fixed. It is possible to fix both a diagonal graph or a generic graph.
#' @param data matrix of dimension \code{n_grid_points \times n} containing the evaluation of \code{n} functional data over a grid of \code{n_grid_points} nodes.
#' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be \code{(niter - burnin)/thin}.
#' @param burnin the number of discarded iterations.
#' @param thin the thining value, it means that only one out of \code{thin} itarations is saved.
#' @param BaseMat matrix of dimension \code{n_grid_points \times p} containing the evalutation of \code{p} Bspline basis over a grid of \code{n_grid_points} nodes.
#' @param diagonal_graph boolean, set true if the graph has to be diagonal. Set false otherwise and pass as input the desired graph through the \code{G} parameter.
#' @param G matrix of size \code{p \times p} representing the graphical part of the model that would remain fixed through out the sampling. Needed only if \code{diagonal_graph} is set to \code{FALSE}. 
#' @param threshold_GWish double, stop algorithm for GWishart samling if the difference between two subsequent iterations is less than \code{threshold_conv.} Used only if \code{diagonal_graph} is set to \code{FALSE}.
#' @param progressBar boolean, set \code{TRUE} to display the progress bar.
#' @return Two lists are returned, one with all the sampler values and the other with the posterior means.
#' @export
FLM_sampling = function(data, niter, burnin, thin, BaseMat, diagonal_graph = T, G = NULL ,threshold_GWish = 1e-8, progressBar = TRUE)
{
  if(diagonal_graph)
  {
    G = matrix(0,2,2)
    return (FLM_sampling_c(data, niter,burnin,thin,BaseMat,G,diagonal_graph,threshold_GWish, progressBar))
  }
  else
    return (FLM_sampling_c(data, niter,burnin,thin,BaseMat,G,diagonal_graph,threshold_GWish, progressBar));
}



#' Simulate curves
#'
#' This function genrates a dataset of n functional curves. Data are generated trying to simulate the shape of one or two Gaussian distributions.
#' @param p Dimension of the true underlying graph.
#' @param n Number of curves to be generated.
#' @param n_grid_points dimension of the grid where the spline are evaluated.
#' @param range_x vector of length two defining the interval for the curves.
#' @param G  matrix of size \eqn{p \times p} representing the true underlying graph. Default is \code{NULL} that corresponds to a diagonal graph.
#' @param K  matrix of size \eqn{p \times p} representing the true underlying precision matrix. If \code{NULL}, a sample from a GWishart is drawn.
#' @param b GWishart shape parameter used if the matrix \code{K} has to be drawn. It has to be larger than 2 in order to have a well defined distribution. Default is 3.
#' @param D GWishart inverse scale matrix parameter used if the matrix \code{K} has to be drawn. It has to be symmetric and positive definite. Default is \code{0.01*diag(p)}.
#' @param spline_order order of the Bsplines. Set four for cubic splines.
#' @param n_picks The number of desired picks in the simulated curves. It may be one for a single, Gaussian like, shape or two for a double Gaussian shape.
#' @param height1 Heigth of the pick of the first Gaussian.
#' @param height2 Heigth of the pick of the second Gaussian.
#' @param width1 Set how large the first pick has to be. This parameter acts like a standard deviation of a Normal distribution, so the larger it is, the thinner is the pick.
#' @param width2 Set how large the second pick has to be. This parameter acts like a standard deviation of a Normal distribution, so the larger it is, the thinner is the pick.
#' @param position1 Set where the first pick has to be located. Position is then computed as \code{range_x[1] + (range_x[2]-range_x[1])/position1}. Set 2 to place it in the middle.
#' @param position2 Set where the second pick has to be located. Position is then computed as \code{range_x[2] - (range_x[2]-range_x[1])/position2}. Set 2 to place it in the middle.
#' @param n_plot The number of curves to be plotted. If it is set to 0, no curves are displayed.
#' @param highlight_band1 Vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \code{[1,n_basis]}.
#' @param highlight_band1 Vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \code{[1,n_basis]}.
#' @param title_plot Title of the plot.
#' @param xtitle Title of the x-axis.
#' @param ytitle Title of the x-axis.
#' @param seed seed used to simulate data. Set 0 for random seed.
#'
#' @return It returns a list with the simulated curves as well as all the other underlying parameters, that are Beta, mu, G, K. It then returns the (n_grid_points x p) design matrix
#' containing the evaluation of all the splines in all the grid points. Finally it also returns the interal knots used in the creation of the splines.
#' @export
simulate_curves = function( p = 10, n = 300, r = 235,range_x = c(100,200), G = NULL, K = NULL, b = 3, D = NULL, tau_eps = 0, spline_order = 3,
                            n_picks = 1, height1 = 1, height2 = 1, width1 = 36, width2 = 24, position1 = 10, position2 = 2,
                            n_plot = n, highlight_band1 = NULL, highlight_band2 = NULL, title_plot = "Curves", xtitle = " ", ytitle = " ",
                            seed = 1212)
{
  if(n_picks <= 0 || n_picks > 2)
    stop("n_picks has to be 1 or 2")
  if(tau_eps < 0)
    stop("tau_eps can only be positive or 0")

  double_gaussian = function(a,b,x,height1, height2, position1, position2, width1, width2){
    if(a > b){
      temp = b
      b = a
      a = temp
    }
    x1 = a + (b-a)/position1
    sd1 = (b-a)/width1
    x2 = b - (b-a)/position2
    sd2 = (b-a)/width2
    return ( height1*exp(-(x-x1)^2/(2*sd1*sd1)) + height2*exp(-(x-x2)^2/(2*sd2*sd2)) )
  }
  single_gaussian = function(a,b,x,height1,position1,width1){
    if(a > b){
      temp = b
      b = a
      a = temp
    }
    x1 = a + (b-a)/position1
    sd1 = (b-a)/width1
    return ( height1*exp(-(x-x1)^2/(2*sd1*sd1))  )
  }
  if(!(length(range_x)==2 || range_x[1] < range_x[2]))
    stop("Invalid range inserted. It has to be of length 2 containing first the lower bound of the interval and then the upper bound.")
  X = seq(range_x[1], range_x[2], length.out = r)
  basis = Generate_Basis(n_basis = p, range = range_x, n_points = r, order = spline_order )
  basemat = basis$BaseMat
  internal_knots = basis$InternalKnots
  knots <- c( range_x[1],
              range_x[1] + (internal_knots[1]-range_x[1])/2,
              internal_knots,range_x[2] - (range_x[2]-internal_knots[length(internal_knots)])/2,
              range_x[2])
  if(n_picks == 1){
    mu = single_gaussian(range_x[1], range_x[2], seq(range_x[1], range_x[2], length.out = p), height1, position1, width1)
  }else if(n_picks == 2){
     mu = double_gaussian(range_x[1], range_x[2], seq(range_x[1], range_x[2], length.out = p), height1, height2, position1, position2, width1, width2)
  }
  if(is.null(G)){
    G = diag(p)
  }
  if(is.null(D)){
    D = 0.01*diag(p)
  }
  if(is.null(K)){
    set.seed(seed)
    K = rgwish(adj = G, b = b, D = D)
  }
  beta <- matrix(0, nrow = n, ncol = p)
  data <- matrix(0, nrow = n, ncol = r)
  for (i in 1:n) {
    set.seed(seed + i)
    beta[i, ] <- rmvnormal(mean = mu, Mat = K, isPrec = T)
    if(tau_eps == 0){
      data[i, ] <- basemat %*% beta[i, ]
    }else{
      data[i, ] <- basemat %*% beta[i, ] + rmvnormal(mean = rep(0,r), Mat = tau_eps*diag(r), isPrec = T)
    }

  }
  simulated_data <- list()
  simulated_data[[1]] <- beta
  simulated_data[[2]] <- mu
  simulated_data[[3]] <- G
  simulated_data[[4]] <- K
  simulated_data[[5]] <- data
  simulated_data[[6]] <- basemat
  simulated_data[[7]] <- internal_knots
  names(simulated_data) <- c("beta", "mu", "G", "K", "data","basemat","internalknots")

  if(n_plot > 0)
    plot_curves(  data1 = data, range = range_x, n_plot = n_plot, grid_points = X,
                  internal_knots = internal_knots, highlight_band1 = highlight_band1, highlight_band2 = highlight_band2,
                  title_plot = title_plot, xtitle = xtitle, ytitle = ytitle  )
  return(simulated_data)
}

#' Bayesian FDR Analysis
#'
#' Given the plinks matrix, this utility computes the False Discovery Rate Index, forcing the false discovery rates to be less than \code{min_rate.}
#' @param plinks matrix of size \eqn{p \times p} containing the posterior inclusion probability for each link. It has to be upper triangular.
#' @param tol sequence of tolerances to be tested trying to select a graph truncating \code{plinks} at that value.
#' @param min_rate fix false discoveries to remain under this selected threshold.
#' @param diag boolean, if the diagonal of \code{plinks} has to be included in the computations. Set \code{FALSE} if the graph is in complete form, set \code{TRUE} for block graphs.
#'
#' @return a list of two elements: best_threshold, the best value of tol according to this analysis.
#' best_truncated_graph, the proposed posterior graph according to the analysis.
#' @export
BFDR_selection = function (plinks, tol = seq(0.1, 1, by = 0.025), min_rate = 0.05, diag = F)
{
  if(dim(plinks)[1] != dim(plinks)[2])
    stop("plinksinks matrix has to be squared")
  p = dim(plinks)[1]
  plinks_vet = plinks[upper.tri(plinks, diag = diag)]
  if (any(tol > max(plinks_vet)))
    tol <- tol[-which(tol > max(plinks_vet))]
  if (is.null(tol))
    stop("No feasible tolerances")
  FDR = rep(0, length(tol))
  for (i in 1:length(tol)) {
    tolerance <- tol[i]
    above_tr = plinks_vet[plinks_vet >= tolerance]
    FDR[i] = sum(1 - above_tr)/length(above_tr)
  }
  if (FDR[1] < min_rate) {
    best_soglia_fdr = tol[1]
  }
  else for (i in 2:length(FDR)) {
    if (FDR[i] < min_rate)
      (break)()
  }
  best_soglia_fdr = tol[i]
  best_graph_fdr = matrix(0, p, p)
  best_graph_fdr[plinks >= best_soglia_fdr] = 1
  result = list()
  result[[1]] = best_soglia_fdr
  result[[2]] = best_graph_fdr
  names(result) = c("best_treshold", "best_truncated_graph")
  return(result)
}

#' Compute AUC Index
#'
#' Function to compute the area under ROC curve exploiting trapezoidal rule.
#' @param x Numeric vector of length \code{n}.
#' @param y Numeric vector of length \code{n}.
#'
#' @return A numeric value which is the area under the ROC curve.
#' @export
Compute_AUC = function (x, y)
{
  l = length(x)
  if (length(x) != length(y))
    stop("length of x and y shoud be the same.")
  auc = 0
  for (i in 2:l) {
    auc = auc + (x[i] - x[i - 1]) * (y[i - 1] + y[i])/2
  }
  return(auc)
}

#' Bayesian Sensitivity Analysis
#'
#' Given the true graph and a \code{plinks} matrix, this utility computes the confusion matrices, plot the ROC curve (if required) and the AUC index,
#' selects the best threshold according to the number of misclassified links and also return the best graph according to the abovementioned analysis.
#' @param PL matrix of size \eqn{p \times p} containing the posterior inclusion probability for each link. It has to be upper triangular.
#' @param G_true matrix of size \eqn{p \times p} containing the true graph. It has to be upper triangular.
#' @param tol sequence of tolerances to be tested trying to select a graph truncating \code{plinks} at that value.
#' @param ROC boolean. If \code{TRUE}, the plot the ROC curve is showed.
#' @param diag
#'
#' @return A list of 5 elements: all_confusion, a list with all the confusion matrices, one for each tol value. best_threshold, the best value of tol according to this analysis.
#' best_confusion, the best confusion matrix, the one corresponding to best_threshold. best_truncated_graph, the proposed posterior graph according to the analysis.
#' AUC, the AUC Index corresponding to the best value selected.
#' @export
Sensitivity_analysis = function (PL, G_true, tol = seq(0.1, 1, by = 0.01), ROC = FALSE, diag = F)
{
  p = dim(PL)[1]
  if(diag){
    diag_PL = diag(PL)
    diag_G  = diag(G)
  }
  G_true_upper = G_true[upper.tri(G_true, diag = F)]
  G_true = matrix(0,p,p)
  G_true[upper.tri(G_true, diag = F)] = G_true_upper
  PL = PL + t(PL)
  G_true = G_true + t(G_true)
  if(diag){
    diag(PL) = diag_PL
    diag(G_true) = diag_G
  }else{
    diag(PL) <- diag(G_true) <- rep(0, p)
  }
  confusion = list()
  if (any(tol > max(PL)))
    tol <- tol[-which(tol > max(PL))]
  if (is.null(tol))
    stop("No feasible tolerances")
  correct = rep(0, length(tol))
  sensitivity = rep(0, length(tol))
  one_minus_specificity = rep(0, length(tol))
  for (i in 1:length(tol))
  {
    tolerance <- tol[i]
    Estimated = matrix(0, p, p)
    Estimated[PL >= tolerance] = 1
    confusion[[i]] = table(G_true, Estimated)
    if(!diag){
      confusion[[i]][1, 1] = confusion[[i]][1, 1] - p
    }
    wrong = confusion[[i]][1, 1] + confusion[[i]][2,2]
    sensitivity[i] = confusion[[i]][2, 2]/(confusion[[i]][2,1] + confusion[[i]][2, 2])
    if (confusion[[i]][1, 1] == 0) {
      one_minus_specificity[i] = 0
    }
    else {
      one_minus_specificity[i] = confusion[[i]][1,1]/(confusion[[i]][1,1] + confusion[[i]][1, 2])
    }

    if(diag){
      correct[i] = wrong/(p * p)
    }else{
      correct[i] = wrong/(p * p - p)
    }
  }
  best_soglia = tol[which.max(correct)]
  best_cut = which.max(correct)
  best_confusion = confusion[[best_cut]]
  best_graph = matrix(0, p, p)
  best_graph[PL >= best_soglia] = 1
  if (isTRUE(ROC)) {
    x11()
    plot(x = (1 - one_minus_specificity), y = sensitivity,
         type = "b", col = "red", pch = 16, lwd = 2, main = "ROC Curve",
         xlab = "1 - spec", ylab = "sens")
    text((1 - one_minus_specificity), sensitivity, tol, col = "black",
         cex = 0.6, pos = 4)
  }
  result = list()
  result[[1]] = confusion
  result[[2]] = best_soglia
  result[[3]] = best_confusion
  result[[4]] = best_graph
  result[[5]] = 1#compute_AUC(one_minus_specificity, sensitivity)
  names(result) = c("all_confusion", "best_threshold", "best_confusion",
                    "best_truncated_graph", "AUC")
  return(result)
}


#' Simulate data from GGM
#'
#' @param p dimension of the response variable that is the dimension of the underlying graph.
#' @param n number of observations.
#' @param n_groups number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as group parameter.
#' @param form string that states if the true graph has to be in \code{"Block"} or \code{"Complete"} form. Only possibilities are \code{"Complete"} and \code{"Block"}.
#' @param groups list representing the groups of block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
#' @param adj_mat matrix representing the desired graph. It has to be a (p x p) matrix if form is \code{"Complete"} and has to be coherent with the number of groups if form is \code{"Block"}.
#' If \code{NULL}, a random graph is generated.
#' @param seed seed used to simulate data. Set 0 for random seed.
#' @param mean_null boolean, set \code{TRUE} if data has to be zero mean.
#' @param sparsity desired sparsity in the graph. Not used if true graph is provided.
#'
#' @return a list containing a matrix called U with sum(Y_iY_i), the true precision, the true graph and if Block form is selected, the graph in its complete form is also returned.
#' @export
SimulateData_GGM = function(p,n,n_groups = 0,form = "Complete",groups = NULL,adj_mat = NULL,
							seed = 0,mean_null = TRUE, sparsity = 0.3 )
{
	if(!(form == "Complete" || form == "Block"))
		stop("Only possible forms are Complete and Block")
	if(form == "Block" && is.null(groups) && n_groups <= 0)
		stop("Groups has to be available if Block form is selected. Set the groups with groups parameter or just set the number of desiderd groups with n_groups parameter")
	if(form == "Block" && !is.null(groups) )
		n_groups = length(groups)
	if(is.null(adj_mat)){
		graph = "random"
		adj_mat = matrix(0,2,2)
	}
	else{
		graph = "fixed"
		if(dim(adj_mat)[1] != dim(adj_mat)[2])
			stop("Adjacency matrix has to be symmetric")
		if(form == "Block"){
			if(dim(adj_mat)[1] != n_groups)
				stop("Size of adjacency matrix is not coherent with the number of groups")
		}
	}
	return (SimulateData_GGM_c(p,n,n_groups,form,graph,adj_mat,seed,mean_null,sparsity,groups))
}

#' Sampler for Guassian Graphical Models
#'
#' This function draws samples a posteriori from a Gaussian Graphical Models.
#' @param data matrix of size \eqn{p \times p} containing \eqn{\sum(Y_i^{T}Y_i)}. Data are required to be zero mean.
#' @param n number of observed data.
#' @param niter number of total iterations to be performed in the sampling. The number of saved iteration will be \code{(niter - burnin)/thin}.
#' @param burnin number of discarded iterations.
#' @param thin thining value, it means that only one out of \code{thin} itarations is saved.
#' @param Param list containing parameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{sampler_parameters}} function. It is indeed recommended to build it through that particular function. \code{BaseMat} field is not needed.
#' @param HyParam list containing hyperparameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{GM_hyperparameters}} function. It is indeed recommended to build it through that particular function.
#' @param form string that may take as values only \code{"Complete"} of \code{"Block"} . It states if the algorithm has to run with \code{"Block"} or \code{"Complete"} graphs.
#' @param prior string with the desidered prior for the graph. Possibilities are \code{"Uniform"}, \code{"Bernoulli"} and for \code{"Block"} graphs only \code{"TruncatedBernoulli"} and \code{"TruncatedUniform"} are also available.
#' @param algo string with the desidered algorithm for sampling from a GGM. Possibilities are \code{"MH"}, \code{"RJ"} and \code{"DRJ"}.
#' @param groups a list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. If \code{"NULL"}, \code{"n_groups"} are automatically generated. Not needed if form is set to \code{"Complete"}.
#' @param n_groups int, number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as group parameter. 
#' @param seed set 0 for random seed.
#' @param print_info boolean, if true progress bar and execution time are displayed.
#' @return This function returns a list with the posterior precision mean, a matrix with the probability of inclusion of each link, the number of accepted moves, the number of visited graphs and the list of all visited graphs.
#' @export
GGM_sampling = function( data, n, niter = 100000, burnin = niter/2, thin = 1, Param = NULL, HyParam = NULL,
                         form = "Complete", prior = "Uniform", algo = "RJ", groups = NULL, n_groups = 0, seed = 0, print_info = TRUE )
{
	p = dim(data)[1]
	if( (dim(data)[1] != dim(data)[2]) && (dim(data)[2] != n) )
		stop("Number of observations in data matrix is not coherent with parameter n")
	if(!(form == "Complete" || form == "Block"))
		stop("Only possible forms are Complete and Block")
	if(form == "Block" && is.null(groups) && n_groups <= 0)
		stop("Groups has to be available if Block form is selected.")
	if(is.null(Param))
		Param = sampler_parameters()
	if(is.null(HyParam))
		HyParam = GM_hyperparameters(p = p)
	else if(dim(HyParam$D_K)[1]!= p){
		stop("Prior inverse scale matrix D is not coherent with data. It has to be a (p x p) matrix. Note that data has to be a (p x p) or a (p x n) matrix.")
	}

	if(form == "Block" && is.null(groups)){
		if(n_groups <= 1 || n_groups > p)
		  stop("Error, invalid number of groups inserted");
		groups = CreateGroups(p,n_groups)
	}

	if( dim(data)[1] != dim(data)[2] ){
		U = data%*%t(data)
		return (GGM_sampling_c( U, p, n, niter, burnin, thin, HyParam$D_K, HyParam$b_K,
								Param$MCprior,Param$MCpost,Param$threshold,form,prior,algo,groups,seed,
								HyParam$Gprior,HyParam$sigmaG,HyParam$p_addrm,print_info ) )
	}else{
		return (GGM_sampling_c( data, p, n, niter, burnin, thin, HyParam$D_K, HyParam$b_K,
								Param$MCprior,Param$MCpost,Param$threshold,form,prior,algo,groups,seed,
								HyParam$Gprior,HyParam$sigmaG,HyParam$p_addrm,print_info ) )
	}
}

#' Skeleton for Hyperparameters in Graphical Models
#'
#' This function simply creates a skeleton for all the hyperparameters that has to be fixed in graphical samplers. All the quantities has a default value,
#' moreover some of them may not be needed in some algorithm. The goal of this function is to fix a precise notation that will be used in all the code.
#' @param a_tau_eps double, shape parameter for gamma prior distribution of \code{tau_eps} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param b_tau_eps double, inverse scale parameter for gamma prior distribution of \code{tau_eps} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param sigma_mu  double, covariance for normal multivariate distribution of mu parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param b_K double, it is prior GWishart shape parameter. It has to be larger than 2 in order to have a well defined distribution. Needed in all graphical model, i.e \code{FGM} and \code{GGM} sampler and in \code{FLM} with fixed graph.
#' @param D_K matrix of double stored columnwise. It is prior GWishart inverse scale parameter. It has to be symmetric and positive definite. Default is identity matrix but \code{p} has to be provided in that case. Needed in all graphical model, i.e \code{FGM} and \code{GGM} sampler and in \code{FLM} with fixed graph.
#' @param p integer, dimension of the graph. It is needed if \code{D_K} parameter is null.
#' @param p_addrm double, probability of proposing a new graph by adding one link. Needed in all graphical model where the graph is random, i.e \code{FGM} and \code{GGM} sampler.
#' @param sigmaG double, the standard deviation used to perturb elements of precision matrix when constructing the new proposed matrix. Needed in all graphical model where the graph is random, i.e \code{FGM} and \code{GGM} sampler. If algorithm is \code{"MH"} it is not used.
#' @param Gprior double representing the prior probability of inclusion of each link in case \code{"Bernoulli"} prior is selected for the graph. Set 0.5 for \code{"Uniform"} prior. Needed in all graphical model where the graph is random, i.e \code{FGM} and \code{GGM} sampler.
#' @return A list with all the inserted hyperparameters.
#' @export
GM_hyperparameters = function(a_tau_eps = 20, b_tau_eps = 0.002, sigma_mu = 100, b_K = 3,
                              D_K = NULL, p = NULL, p_addrm = 0.5, sigmaG = 0.5, Gprior = 0.5)
{
	if(is.null(D_K)){
		if(is.null(p))
			stop("Default for GWishart inverse scale prior parameter is diagonal matrix. Dimension p has to be provided.")
		else
			D_K = diag(p)
	}else{
		if(dim(D_K)[1] != dim(D_K)[2] )
			stop("Inverse scale matrix D_K has to be squared symmetric positive definite matrix.")
	}
	hy = list(  "a_tau_eps" = a_tau_eps,
				"b_tau_eps" = b_tau_eps,
				"sigma_mu"  = sigma_mu,
				"b_K"		= b_K,
				"D_K"		= D_K,
				"p_addrm"   = p_addrm,
				"sigmaG"	= sigmaG,
				"Gprior"	= Gprior     )
	return (hy)
}

#' Skeleton for Hyperparameters in Linear Models
#'
#' This function simply creates a skeleton for all the hyperparameters that has to be fixed in linear samplers. All the quantities has a default value,
#' moreover some of them may not be needed in some algorithm. The goal of this function is to fix a precise notation that will be used in all the code.
#' @param a_tau_eps double, shape parameter for gamma prior distribution of \code{tau_eps} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param b_tau_eps double, inverse scale parameter for gamma prior distribution of \code{tau_eps} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param sigma_mu  double, covariance for normal multivariate distribution of mu parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param b_K double, it is prior GWishart shape parameter. It has to be larger than 2 in order to have a well defined distribution. Needed in all graphical model, i.e \code{FGM} and \code{GGM} sampler and in \code{FLM} with fixed graph.
#' @param D_K matrix of double stored columnwise. It is prior GWishart inverse scale parameter. It has to be symmetric and positive definite. Default is identity matrix but \code{p} has to be provided in that case. Needed in all graphical model, i.e \code{FGM} and \code{GGM} sampler and in \code{FLM} with fixed graph.
#' @param p integer, number of basis. It is needed if \code{D_K} parameter is null.
#' @param a_tauK double, shape parameter for gamma prior distribution for each \code{tau_K} parameters. Needed only in \code{FLM} sampler with diagonal graph.
#' @param b_tauK double, inverse scale parameter for gamma prior distribution for each \code{tau_K} parameters. Needed only in \code{FLM} sampler with diagonal graph.
#' @return A list with all the inserted hyperparameters.
#' @export
LM_hyperparameters = function(a_tau_eps = 20, b_tau_eps = 0.002, sigma_mu = 100, b_K = 3,
                              D_K = NULL, p = NULL, a_tauK = 0.5, b_tauK = 0.5)
{
	if(!is.null(D_K) && dim(D_K)[1] != dim(D_K)[2]){
		stop("Inverse scale matrix D_K has to be squared symmetric positive definite matrix.")
	}
	hy = list(  "a_tau_eps" = a_tau_eps,
				"b_tau_eps" = b_tau_eps,
				"sigma_mu"  = sigma_mu,
				"b_K"		= b_K,
				"D_K"		= D_K,
				"a_tauK"    = a_tauK,
				"b_tauK"	= b_tauK   )
	return (hy)
}

#' Skeleton for Parametes
#'
#' This function simply creates a skeleton for some parameters that has to be fixed in the samplers. All the quantities has a default value,
#' moreover some of them may not be needed in some algorithm. The goal of this function is to fix a precise notation that will be used in all the code.
#' @param MCprior positive integer, the number of iteration for the MonteCarlo approximation of prior normalizing constant of GWishart distribution. Needed in \code{"MH"} and \code{"RJ"} algorithms.
#' @param MCpost positive integer, the number of iteration for the MonteCarlo approximation of posterior normalizing constant of GWishart distribution. Needed only in \code{"MH"} algorithms.
#' @param BaseMat matrix of dimension \code{n_grid_points \times p} containing the evalutation of \code{p} Bspline basis over a grid of \eqn{n_grid_points} nodes. May be defaulted as \code{NULL} but note this case it is not automatically generated.
#' Use \code{\link{Generate_Basis}} to generete it. Needed in all Functional models, i.e \code{FGM} and \code{FLM} samplers.
#' @param threshold double, threshold for convergence in GWishart sampler. It is not needed only in \code{FLM} sampler with diagonal graph.
#' @return A list with all the inserted parameters.
#' @export
sampler_parameters = function(MCprior = 500, MCpost = 750, BaseMat = NULL, threshold = 1e-14)
{

	param = list( "MCprior"   = MCprior,
				  "MCpost"    = MCpost,
				  "BaseMat"   = BaseMat,
				  "threshold" = threshold )
	return (param)
}

GM_init = function(p ,n, empty = TRUE, G0 = NULL, K0 = NULL, Beta0 = NULL, mu0 = NULL, tau_eps0 = NULL, form = "Complete", groups = NULL, n_groups = 0, seed = 0)
{
	if(!(form == "Complete" || form == "Block"))
		stop("Only possible forms are Complete and Block")
	if(form == "Block" && is.null(groups) && n_groups <= 0)
		stop("Groups has to be available if Block form is selected.")

	if(is.null(G0)){
		if(form == "Complete"){
			G0 = Create_RandomGraph(p = p, form = "Complete",  groups = NULL, sparsity = 0.5, seed = seed  )
		}
		else if(form == "Block"){
			if(is.null(groups)){
				if(n_groups <= 1 || n_groups > p)
				  stop("Error, invalid number of groups inserted")

				groups = CreateGroups(p,n_groups)
			}
			G0 = Create_RandomGraph(p = p, form = "Block",  groups = groups, sparsity = 0.5, seed = seed  )
		}
		else
			stop("Only possible forms are Complete and Block")
	}
	if(is.null(K0)){
		K0 = rGwish(G0, 3, diap(p), groups = groups, threshold_conv = 1e-14, seed = seed)$Matrix
	}
	if(is.null(tau_eps0)){
		tau_eps0 = rgamma(n = 1, shape = 50, rate = 2)
	}
	if(is.null(mu0)){
		mu0 = rmvnormal(mean = rep(0,p), Mat = diag(p), isPrec = FALSE)
	}
	if(is.null(Beta0)){
		Beta0 = matrix(0,nrow = p, ncol = n)
		for(i in 1:n){
			Beta0[,i] = rmvnormal(mean = mu0, Mat = K0, isPrec = TRUE, isChol = FALSE)
		}
	}
}

#' Tuning \code{sigmaG} parameter
#'
#' The choice of \code{sigmaG} parameter is a crucial part of \code{"RJ"} and \code{"DRJ"} algorithm. It indeed represents the standard deviation for the proposal of new elements in the
#' precision matrix. This function provides a simple utility for tuning this parameter by simply repeting \code{Nrep} times the first \code{niter} iteration for every proposed
#' \code{sigmaG} in \code{sigmaG_list}.
#' @param data matrix of size \eqn{p \times p} containing \eqn{\sum(Y_i^{T}Y_i)}.
#' @param n number of observed values.
#' @param niter iteration to be performed in the sampling.
#' @param sigmaG_list list containing the values of \code{sigmaG} that has to be tested.
#' @param Nrep how many times each \code{sigmaG} has to be tested.
#' @param Param list containing parameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{sampler_parameters}} function. It is indeed recommended to build it through that particular function. \code{BaseMat} field is not needed.
#' @param HyParam list containing hyperparameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{GM_hyperparameters}} function. It is indeed recommended to build it through that particular function.
#' @param form string that may take as values only \code{"Complete"} of \code{"Block"}. It states if the algorithm has to run with \code{"Block"} or \code{"Complete"} graphs.
#' @param prior string with the desidered prior for the graph. Possibilities are \code{"Uniform"}, \code{"Bernoulli"} and for \code{"Block"} graphs only \code{"TruncatedBernoulli"} and \code{"TruncatedUniform"} are also available.
#' @param algo string with the desidered algorithm for sampling from a \code{GGM}. Possibilities are \code{"MH"}, \code{"RJ"} and \code{"DRJ"}.
#' @param n_groups integer, number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as \code{groups} parameter. 
#' @param groups a list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. If \code{NULL}, \code{n_groups} are automatically generated. Not needed if form is set to \code{"Complete"}.
#' @return plots the mean acceptance ratio for each \code{sigmaG} and returns the highest one.
#' @export
sigmaG_GGM_tuning = function( data, n, niter = 1000, sigmaG_list = seq(0.05, 0.55, by = 0.05), Nrep = 10,
							  Param = NULL, HyParam = NULL, form = "Complete", prior = "Uniform", algo = "RJ", n_groups = 0, groups = NULL  )
{
	nburn = niter-1
	thin  = 1
	accepted = rep(0,length(sigmaG_list))
	counter = 1
	if(is.null(Param))
		Param = sampler_parameters()
	if(is.null(HyParam))
		HyParam = GM_hyperparameters(p = p)
	for(sigmaG in sigmaG_list){
		cat('\n Starting sigmaG = ',sigmaG,'\n')
		#pb <- txtProgressBar(min = 1, max = Nrep, initial = 1, style = 3)
	  	tot = 0
	  	HyParam$sigmaG = sigmaG
	  for(i in 1:Nrep){
	    result = GGM_sampling(	data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                      			Param = param, HyParam = hy, groups = NULL, n_groups = n_groups,
                      			prior = prior, form = form, algo = algo,
                      			seed = 0, print_info = FALSE  )
	    tot = tot + result$AcceptedMoves
	    #setTxtProgressBar(pb, i)
	    cat('  i = ',i,', acc = ',result$AcceptedMoves,'  ||  ')
	  }
	  #close(pb)
	  accepted[counter] = tot/Nrep
	  counter = counter + 1
	}
	accepted = accepted/niter
	x11();plot(accepted, pch = 16, col = 'black', xlab = "sigmaG", ylab = "Acceptance Ratio", xaxt = 'n')
	mtext(text = sigmaG_list, side=1, line=0.3, at = 1:length(sigmaG_list) , las=2, cex=1.0)
	return (sigmaG_list[which.max(accepted)])
}


#' Block to Complete map
#'
#' Function for mapping from a block matrix to its non-block representation. It can be used also for matrices whose entries are not just 1 or 0.
#' @param Gblock matrix of size \eqn{n_groups \times n_groups} to be mapped.
#' @param groups list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. It is possible to generete it through CreateGroups function.
#' @return p x p matrix containing the Complete form of Gblock.
#' @export
Block2Complete = function(Gblock, groups)
{
  m = length(groups)
  p = 0
  for(i in 1:m)
    p = p + length(groups[[i]])
  G = diag(p)
  for(i in 1:m){ #loop over all possible combinations
    for(j in i:m){
      
      if(!(i==j & length(groups[[i]])==1)){ #diagonal element, not important
        if(Gblock[i,j] > 0){
          v = expand.grid(groups[[i]]+1, groups[[j]]+1) #set +1 because groups follow c++ notation, i.e they start from 0
          if(i==j){
            v = v[which(v$Var1<v$Var2),] #for avoiding repetitions
          }
          for(h in 1:dim(v)[1]){
            G[min(v$Var1[h],v$Var2[h]),max(v$Var1[h],v$Var2[h])] = Gblock[i,j] #set elements
          }
        }
      }
      
    }
  }
  return(G)
}
