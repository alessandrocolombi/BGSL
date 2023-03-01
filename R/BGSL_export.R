check_structure = function(G, K, threshold = 1e-5){
  if(dim(G)[1] != dim(G)[2])
    stop("G has to be squared")
  if(dim(K)[1] != dim(K)[2])
    stop("K has to be squared")
  if(dim(K)[1] != dim(G)[1])
    stop("G and K has to have the same number of elements")
  p = dim(G)[1]
  for(i in 1:(p-1)){
    for(j in (i+1):p){
      if( ( G[i,j] == 0 && abs(K[i,j])>threshold ) || (G[i,j] == 1 && abs(K[i,j])<threshold)){
        return (FALSE);
      }
    }
  }
  return (TRUE);
}

#' Tuning MC iterations for posterior normalizing constant of GWishart distribution
#'
#' This function selects the number of MonteCarlo iterations when computing a the GWishart posterior normalizing constant. The problem is that the MonteCarlo
#' method by Atack-Massam requires computing exp(-0.5 sum(Phi_ij)^2). As the number of nodes in the graph grows, that quantity is smaller and smaller and
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
#' \loadmathjax This function gets the regression coefficients and the evaluated splines and builds the smooted curves. It may also plot the curves.
#' @param beta matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the values of regression coefficients.
#' @param BaseMat matrix of dimension \mjseqn{n\_grid\_points \times n\_basis} containing the evaluation of all the \mjseqn{n\_basis} B-spine basis function in all the grid points.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param range the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.
#' @param grid_points vector of size \mjseqn{n\_grid\_points} with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis}}. Default is null.
#' If provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#'
#' @return It returns a \mjseqn{n\_curves \times n\_grid\_points} matrix containing the values of the smoothed curves.
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
        knots_for_plot = round(knots, digits = 2)
        names <- rep("", length(knots_for_plot))
        for (i in 1:length(knots_for_plot)) {
          names[i] <- paste0(knots_for_plot[i])
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
#' \loadmathjax This function gets the mean values of the regression coefficients as well as the lower and upper quantiles and builds the smooted curves with their credible bands.
#' It does not computes the quantiles nor the mean starting from all the sampled values, they have to be previously computed using \code{\link{Compute_Quantiles}} function.
#' @param beta matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the values of regression coefficients.
#' @param betaLower matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the lower quantiles values of regression coefficients. Can be obtained by \code{\link{Compute_Quantiles}} function.
#' @param betaUpper matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the upper quantiles values of regression coefficients. Can be obtained by \code{\link{Compute_Quantiles}} function.
#' @param BaseMat matrix of dimension \mjseqn{n\_grid\_points \times n\_basis} containing the evaluation of all the \mjseqn{n\_basis} B-spine basis function in all the grid points.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param range the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.
#' @param grid_points vector of size \mjseqn{n\_grid\_points} with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis}}. Default is null.
#' If provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#' @param data a \mjseqn{n\_curves \times n\_grid\_points} matrix containing the values of the true curves that are added to the plot if this parameter is not \code{NULL}.
#'
#' @return It returns a list composed of three \mjseqn{n\_curves \times n\_grid\_points} matrices called \code{MeanCurves}, \code{LowerBands} and \code{UpperBands}
#' containing the smoothed curves and their credible bands.
#' @export
smooth_curves_credible_bands = function(  beta, betaLower, betaUpper, BaseMat, n_plot = 0, range = NULL, grid_points = NULL,
                                          internal_knots = NULL, highlight_band1 = NULL, highlight_band2 = NULL,
                                          title_plot = "Smoothed Curves", xtitle = " ", ytitle = " ", data = NULL)
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
  if(n_plot > 0 & is.null(data)) #Plot n_plot curves
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
        knots_for_plot = round(knots, digits = 2)
        names <- rep("", length(knots_for_plot))
        for (i in 1:length(knots_for_plot)) {
          names[i] <- paste0(knots_for_plot[i])
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
  }else if(n_plot > 0 & !is.null(data) ) #Plot n_plot curves and data
  {
    #Check dimensions
    if(is.null(range))
      stop("The range has to be provided in order to plot the curves.")
    if(!(length(range)==2 && range[1] < range[2]))
      stop("Invalid range, it has to be a vector of length 2 containing first the lower bound of the interval and then the upper bound.")
    if(!(dim(data)[1]==n && dim(data)[2]==r))
      stop("data matrix has to be n x r.")
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
                col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'),
                lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
        matplot( x = X, t(Ylower[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
        matplot( x = X, t(Yupper[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
        matplot( x = X, t(data[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'), lwd = 3, add = T)
      }
      else if(n_plot == 1){
        x11(height=4)
        matplot( x = X, (Ymean[1:n_plot,]), type = 'l', lty = 1,
                col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'),
                lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = T,
                main = title_plot, xlab = xtitle,
                ylab = ytitle)
        matplot( x = X, (Ylower[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
        matplot( x = X, (Yupper[1:n_plot,]), type = 'l', lty = 1,
                col = c('gray50','gray70','gray40'), lwd = 3, add = T)
        matplot( x = X, (data[1:n_plot,]), type = 'l', lty = 1,
                col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'), lwd = 3, add = T)
      }
    }
    else{ #Plot with bands representing the domanin of the spline
        knots <- c(range[1],
                 range[1] + (internal_knots[1]-range[1])/2,
                 internal_knots,
                 range[2] - (range[2]-internal_knots[length(internal_knots)])/2,
                 range[2] )
        knots_for_plot = round(knots, digits = 2)
        names <- rep("", length(knots_for_plot))
        for (i in 1:length(knots_for_plot)) {
          names[i] <- paste0(knots_for_plot[i])
        }
        names_y = round(seq(min(Ymean[1:n_plot,]), max(Ymean[1:n_plot,]), length.out = 10), digits = 2)
        if(n_plot > 1){
          x11(height=4)
            matplot(x = X, t(Ymean[1:n_plot,]), type = 'l', lty = 1,
                    col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'),
                    lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = F,
                    main = title_plot, xlab = xtitle,
                    ylab = ytitle)
            matplot( x = X, t(Ylower[1:n_plot,]), type = 'l', lty = 1,
                    col = c('gray50','gray70','gray40'), lwd = 3, add = T)
            matplot( x = X, t(Yupper[1:n_plot,]), type = 'l', lty = 1,
                    col = c('gray50','gray70','gray40'), lwd = 3, add = T)
            matplot( x = X, t(data[1:n_plot,]), type = 'l', lty = 1,
                    col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'), lwd = 3, add = T)
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
                      col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'),
                      lwd = 3, ylim = c(min(Ylower[1:n_plot,]),max(Yupper[1:n_plot,])), axes = F,
                      main = title_plot, xlab = xtitle,
                      ylab = ytitle)
              matplot( x = X, (Ylower[1:n_plot,]), type = 'l', lty = 1,
                      col = c('gray50','gray70','gray40'), lwd = 3, add = T)
              matplot( x = X, (Yupper[1:n_plot,]), type = 'l', lty = 1,
                      col = c('gray50','gray70','gray40'), lwd = 3, add = T)
              matplot( x = X, t(data[1:n_plot,]), type = 'l', lty = 2,
                    col = c('darkolivegreen','darkgreen','darkolivegreen4','forestgreen'), lwd = 3, add = T)
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
#' \loadmathjax This functions gets one or two dataset representig functional data and plot them. It does not smooth the curves, indeed it requires as input the data, not
#' the regression coefficients. Use \code{\link{smooth_curves}} function for that.
#' @param data1 matrix of dimension \mjseqn{n\_curves \times n\_grid\_points} representing the first functional dataset to be plotted.
#' @param data2 matrix of dimension \mjseqn{n\_curves \times n\_grid\_points} representing the second dataset to be plotted, if needed.
#' @param range the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param grid_points vector of size \mjseqn{n\_grid\_points} with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis}}. Default is null.
#' If provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#' @param legend_name1 the name for \code{data1} to be printed in the legend.
#' @param legend_name2 the name for \code{data2} to be printed in the legend. Used only is two datasets are actually plotted.
#'
#' @return No values are returned.
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
            matplot( x = X, t(data2[1:n_plot,]), type = 'l', lty = 1,
            col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
            legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,1), lwd = 3)
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
           matplot( x = X, (data2[1:n_plot,]), type = 'l', lty = 1,
           col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
           legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,1), lwd = 3)
       }
      }
    }
    else{ #Plot with bands representing the domanin of the spline
        knots <- c(range[1],
                 range[1] + (internal_knots[1]-range[1])/2,
                 internal_knots,
                 range[2] - (range[2]-internal_knots[length(internal_knots)])/2,
                 range[2] )
        knots_name = round(knots, digits = 2)
        names <- rep("", length(knots_name))
        for (i in 1:length(knots_name)) {
          names[i] <- paste0(knots_name[i])
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
                matplot( x = X, t(data2[1:n_plot,]), type = 'l', lty = 1,
                col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
                legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,1), lwd = 3)
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
                  matplot( x = X, (data2[1:n_plot,]), type = 'l', lty = 1,
                  col = c('steelblue','steelblue1','skyblue3', 'lightsteelblue1'), lwd = 3, add = T)
                  legend("topright", legend=c(legend_name1, legend_name2), col = c('darkolivegreen','steelblue'), lty = c(1,1), lwd = 3)
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
#' \loadmathjax This function performs a linear regression for functional data.
#' It is not a graphical model, the graph has to be fixed. It is possible to fix both a diagonal graph or a generic graph. In the first case, the model one sample from is the following,
#' \mjtdeqn{$$\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} ~\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), \forall i = 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K~\sim~&N_{p}\left(\mu,K\right) K=diag\left(\tau_{1},\dots,\tau_{p}\right) \cr \mu~\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} ~\sim~& Gamma\left(a,b\right) \cr \tau_{j} ~\sim~& Gamma\left(\frac{a}{2},\frac{b}{2}\right) \forall j = 1:p \end{eqnarray*}$$}{\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} &~\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i &= 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K&~\sim~&N_{p}\left(\mu,K\right) &K&=diag\left(\tau_{1},\dots,\tau_{p}\right) \cr \mu&~\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} &~\sim~& Gamma\left(a,b\right) \cr \tau_{j} &~\sim~& Gamma\left(\frac{a}{2},\frac{b}{2}\right) &\forall j &= 1:p \end{eqnarray*}}{\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} &~\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i &= 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K&~\sim~&N_{p}\left(\mu,K\right) &K&=diag\left(\tau_{1},\dots,\tau_{p}\right) \cr \mu&~\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} &~\sim~& Gamma\left(a,b\right) \cr \tau_{j} &~\sim~& Gamma\left(\frac{a}{2},\frac{b}{2}\right) &\forall j &= 1:p \end{eqnarray*}}
#' Otherwise it is possible to keep the graph fixed even if it is not diagonal. In this case the precision matrix is modeled a priori as a \code{GWishart}. The resulting model is the following,
#' \mjtdeqn{$$\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} ~\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), \forall i = 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K~\sim~&N_{p}\left(\mu,K\right) \cr \mu~\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} ~\sim~& Gamma\left(a,b\right) \cr K~|~ G ~\sim~& GWish\left(d,D\right) \cr G ~~& fixed \end{eqnarray*}$$}{\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} &~\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i &= 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K&~\sim~&N_{p}\left(\mu,K\right) \cr \mu&~\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} &~\sim~& Gamma\left(a,b\right) \cr K~|~ G ~&\sim~& GWish\left(d,D\right) \cr G ~&~ fixed \end{eqnarray*}}{\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} &~\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i &= 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K&~\sim~&N_{p}\left(\mu,K\right) \cr \mu&~\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} &~\sim~& Gamma\left(a,b\right) \cr K~|~ G ~&\sim~& GWish\left(d,D\right) \cr G ~&~ fixed \end{eqnarray*}}
#' @param p number of basis functions.
#' @param data matrix of dimension \mjseqn{n\_curves \times n\_grid\_points} containing the evaluation of \code{n} functional data over a grid of \code{n_grid_points} nodes.
#' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be \mjseqn{(niter - burnin)/thin}.
#' @param burnin the number of discarded iterations.
#' @param thin the thining value, it means that only one out of \code{thin} itarations is saved.
#' @param diagonal_graph boolean, set true if the graph has to be diagonal. Set false otherwise and pass as input the desired graph through the \code{G} parameter.
#' @param G matrix of size \mjseqn{p \times p} representing the graphical part of the model that would remain fixed through out the sampling. Needed only if \code{diagonal_graph} is set to \code{FALSE}.
#' @param Param list containing parameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{sampler_parameters}} function.
#' It is indeed recommended to build it through that particular function. It is important to remember that \code{BaseMat} field is needed and cannot be defaulted. Use \code{\link{Generate_Basis}} to create it.
#' It has to be a matrix of dimension \mjseqn{n\_grid\_points \times p} containing the evalutation of \code{p} Bspline basis over a grid of \code{n_grid_points} nodes.
#' @param HyParam list containing hyperparameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{LM_hyperparameters}} function. It is indeed recommended to build it through that particular function.
#' @param Init list containig initial values for Markov chain. It has to follow the same notation of the one generated by \code{\link{LM_init}} function. It is indeed recommended to build it through that particular function.
#' @param print_info boolean, set \code{TRUE} to display the progress bar.
#' @param seed integer, seeding value. Set 0 for random seed.
#' @param file_name string, the name of the binary \code{".h5"} file where the sampled values are wittern
#'
#' @return It returns a list with the posterior mean of the sampled values. If \code{diagonal_graph} is \code{TRUE}, \code{p} \mjseqn{\tau_{j}} coefficients are returned, if it is \code{FALSE}
#' the full estimated precision matrix is returned. A binary \code{".h5"} file named \code{file_name} is also generated, it contains all the sampled values.
#' @export
FLM_sampling = function( p, data, niter = 100000, burnin = niter/2, thin = 1, diagonal_graph = T, G = NULL ,
                         Param = NULL, HyParam = NULL, Init = NULL, print_info = TRUE, seed = 0, file_name = "FLMresults" )
{
  #n = dim(data)[2]
  n = length(data)
  #Checks Param / HyParam / Init structures
  if(is.null(Param))
    Param = BGSL:::sampler_parameters()
  else if(is.null(Param$BaseMat) || is.null(Param$threshold))
    stop("Param list is incorrectly set. Please use sampler_parameters() function to create it. Hint: in Functional Models, BaseMat field cannot be defaulted. Use Generate_Basis() to create it.")
  if(is.null(Param$Grid)){
    cat('Common grid case \n ')
    r = dim(Param$BaseMat)[1]
    Param$Grid = list()
    for(i in 1:n)
      Param$Grid[[i]] = 0:(r-1)
  }

  if(is.null(HyParam))
    HyParam = BGSL:::LM_hyperparameters(p = p)
  else if( is.null(HyParam$D_K) || is.null(HyParam$b_K) || is.null(HyParam$a_tau_eps) || is.null(HyParam$b_tau_eps) || is.null(HyParam$sigma_mu) || is.null(HyParam$a_tauK) || is.null(HyParam$b_tauK) )
    stop("HyParam list is incorrectly set. Please use LM_hyperparameters() function to create it, or just leave NULL for default values.")
  if(dim(HyParam$D_K)[1]!= p){
    stop("Prior inverse scale matrix D is not coherent with the number of basis function. It has to be a (p x p) matrix.")
  }

  if(is.null(Init))
    Init = BGSL:::LM_init(p = p, n = n, empty = TRUE )
  else if(is.null(Init$Beta0) || is.null(Init$K0) || is.null(Init$mu0) || is.null(Init$tauK0) || is.null(Init$tau_eps0) )
    stop("Init list is incorrectly set. Please use LM_init() function to create it, or just leave NULL for default values.")

  if(diagonal_graph)
    G = matrix(0,2,2)
  else{
    if(is.null(G))
      stop("If diagonal_graph option is selected, the true underlying graph G has to be provided")
    if(dim(G)[1] != dim(G)[2] || dim(G)[1] != p )
      stop("True graph G was wrongly inserted. It has to be a (p x p) matrix")
  }
  return (BGSL:::FLM_sampling_c(  data, niter, burnin, thin, Param$BaseMat, Param$Grid,
                                  G,
                                  Init$Beta0, Init$mu0, Init$tau_eps0, Init$tauK0, Init$K0, #initial values
                                  HyParam$a_tau_eps, HyParam$b_tau_eps, HyParam$sigma_mu, HyParam$a_tauK, HyParam$b_tauK, HyParam$b_K, HyParam$D_K, #hyperparameters
                                  file_name, diagonal_graph, Param$threshold, seed, print_info
                                )

         )
}


#' Simulate curves
#'
#' \loadmathjax This function genrates a dataset of n functional curves. Data are generated trying to simulate the shape of one or two Gaussian distributions.
#' @param p integer, the number of basis functions. It also represents the dimension of the true underlying graph.
#' @param n integer, the number of curves to be generated.
#' @param r integer, the dimension of the grid where the spline are evaluated.
#' @param range_x the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.
#' @param G  matrix of size \mjseqn{p \times p} representing the true underlying graph. Only the upper triangular part is needed. Default is \code{NULL} that corresponds to a diagonal graph.
#' @param K  matrix of size \mjseqn{p \times p} representing the true underlying precision matrix. If \code{NULL}, a sample from a GWishart(\code{b}, \code{D}) is drawn.
#' @param b GWishart Shape parameter used if the matrix \code{K} has to be drawn. It has to be larger than 2 in order to have a well defined distribution.
#' @param D GWishart Inverse Scale matrix parameter used if the matrix \code{K} has to be drawn. It has to be \mjseqn{p \times p}, symmetric and positive definite.
#' @param tau_eps numerical value, it represents the precision of the Gaussian noise to be added to the curves. Set 0 to have smooth curve, no noise is added in that case.
#' @param rate If \code{G} is \code{NULL}, only the diagonal elements of \code{K} are generated. They are sampled from a \mjseqn{Gamma(Shape = 5, Rate = rate)}.
#' @param spline_order order of the Bsplines. Set four for cubic splines.
#' @param n_picks The number of desired picks in the simulated curves. It may be one for a single, Gaussian like, shape or two for a double Gaussian shape.
#' @param height1 Heigth of the pick of the first Gaussian.
#' @param height2 Heigth of the pick of the second Gaussian.
#' @param width1 Set how large the first pick has to be. This parameter acts like a standard deviation of a Normal distribution, so the larger it is, the thinner is the pick.
#' @param width2 Set how large the second pick has to be. This parameter acts like a standard deviation of a Normal distribution, so the larger it is, the thinner is the pick.
#' @param position1 Set where the first pick has to be located. Position is then computed as \mjseqn{range\_x\[1\] + (range\_x\[2\]-range\_x\[1\])/position1}.
#' @param position2 Set where the second pick has to be located. Position is then computed as \mjseqn{range\_x\[2\] - (range\_x\[2\]-range\_x\[1\])/position2}. Set 2 to place it in the middle.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param highlight_band2 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#' @param seed seed used to simulate data. Set 0 for random seed.
#'
#' @return It returns a list composed of: all simulated curves in \code{data}, all underlying parameters, called \code{Beta}, \code{mu}, \code{K} and the true graph \code{G}.
#' It also return the design matrix \code{basemat}, generated by \code{\link{Generate_Basis}} and the vector of the internal knots generating the splines, see \code{\link{Generate_Basis}}.
#' @export
simulate_curves = function( p = 10, n = 300, r = 235,range_x = c(100,200), G = NULL, K = NULL, b = 3, D = NULL, tau_eps = 0, rate = 0.01/2,
                            spline_order = 3, n_picks = 1, height1 = 1, height2 = 1, width1 = 36, width2 = 24, position1 = 10, position2 = 2,
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
  basis = BGSL:::Generate_Basis(n_basis = p, range = range_x, n_points = r, order = spline_order )
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
    tauK = stats::rgamma(n = p, shape = 10/2, rate = rate)
    K = diag(tauK)
  }
  if(is.null(D)){
    D = 0.01*diag(p)
  }
  if(is.null(K)){
    K = BGSL:::rGwish(G = G, b = b, D = D, threshold_conv = 1e-16, seed = 0)$Matrix
    #K = BDgraph::rgwish(adj = G, b = b, D = D)
  }
  beta <- matrix(0, nrow = n, ncol = p)
  data <- matrix(0, nrow = n, ncol = r)
  for (i in 1:n) {
    seed = 0
    beta[i, ] <- BGSL:::rmvnormal(mean = mu, Mat = K, isPrec = T, seed = seed)
    if(tau_eps == 0){
      data[i, ] <- basemat %*% beta[i, ]
    }else{
      data[i, ] <- basemat %*% beta[i, ] + BGSL:::rmvnormal(mean = rep(0,r), Mat = tau_eps*diag(r), isPrec = T, seed = seed)
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
#' \loadmathjax Given the plinks matrix, this utility computes the False Discovery Rate Index, forcing the false discovery rates to be less than \code{min_rate.}
#' @param plinks matrix containing the posterior inclusion probability for each link. It has to be upper triangular. Its dimension depends on the type of graph it represents.
#' It is indeed possible to pass a \mjseqn{p \times p} matrix, or a \mjseqn{n\_groups \times n\_groups}.
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
  if(FDR[1] < min_rate) {
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
#' \loadmathjax Function to compute the area under ROC curve exploiting trapezoidal rule.
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
#' \loadmathjax Given the true graph and a \code{plinks} matrix, this utility computes the confusion matrices, plot the ROC curve (if requested) and the AUC index,
#' selects the best threshold according to the number of misclassified links and also return the best graph according to the abovementioned analysis.
#' @param PL matrix of size \eqn{p x p} containing the posterior inclusion probability for each link. It has to be upper triangular.
#' @param G_true matrix of size \eqn{p x p} containing the true graph. It has to be upper triangular.
#' @param tol sequence of tolerances to be tested trying to select a graph truncating \code{plinks} at that value.
#' @param ROC boolean. If \code{TRUE}, the plot the ROC curve is showed.
#' @param diag boolean, if the diagonal of \code{plinks} has to be included in the computations. Set \code{FALSE} if the graph is in complete form, set \code{TRUE} for block graphs.
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
  result[[5]] = Compute_AUC(one_minus_specificity, sensitivity)
  names(result) = c("all_confusion", "best_threshold", "best_confusion",
                    "best_truncated_graph", "AUC")
  return(result)
}


#' Simulate data from GGM
#'
#' \loadmathjax This function generates a dataset from a Gaussian Graphical Model where the precision matrix is distributes according to GWishart(3,\mjseqn{I_{p}}).
#' @param p integer, the dimension of the underlying graph.
#' @param n integer, number of observations.
#' @param n_groups number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as \code{group} parameter.
#' @param form string that states if the true graph has to be in \code{"Block"} or \code{"Complete"} form. Only possibilities are \code{"Complete"} and \code{"Block"}.
#' @param groups list representing the groups of block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. Leave \code{NULL} if the graph is not in block form.
#' @param adj_mat matrix representing the desired graph. It has to be a \mjseqn{p \times p} matrix if form is \code{"Complete"} and has to be coherent with the number of groups if form is \code{"Block"}.
#' Only the upper triangular part is needed. If \code{NULL}, a random graph is generated.
#' @param seed integer, seeding value. Set 0 for random seed.
#' @param mean_null boolean, set \code{TRUE} if data has to be zero mean.
#' @param sparsity desired sparsity in the graph. It has to be in the range \mjseqn{(0,1)}. Not used if true graph is provided.
#'
#' @return a list composed of: the covariance matrix of the generated data, it is called \code{U} and contains \mjseqn{\sum_{i=1}^{n}(y_{i}y_{i}^{T})},
#' the true precision matrix called \code{Prec_true}, and the true graph called \code{G_true}.
#' If \code{form} is \code{"Block"}, then \code{G_true} is in block form, i.e a \mjseqn{n\_groups \times n\_groups} matrix and it complete form is also returned, it is called \code{G_complete}.
#' @export
SimulateData_GGM = function(p, n, n_groups = 0, form = "Complete", groups = NULL, adj_mat = NULL,
							              seed = 0, mean_null = TRUE, sparsity = 0.3 )
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
	return (BGSL:::SimulateData_GGM_c(p,n,n_groups,form,graph,adj_mat,seed,mean_null,sparsity,groups))
}

#' Sampler for Gaussian Graphical Models
#'
#' \loadmathjax This function draws samples a posteriori from a Gaussian Graphical Models.
#' \mjtdeqn{$$\begin{eqnarray*}y_{1},\dots,y_{n}|K \sim& N_{p}\left(0,K\right) \cr K| G \sim& GWish\left(d,D\right) \cr G \sim&\pi\left(G\right)\end{eqnarray*}$$}{\begin{eqnarray*}y_{1},\dots,y_{n}|K \sim& N_{p}\left(0,K\right) \cr K| G \sim& GWish\left(d,D\right) \cr G \sim&\pi\left(G\right)\end{eqnarray*}}{\begin{eqnarray*}y_{1},\dots,y_{n}|K \sim& N_{p}\left(0,K\right) \cr K| G \sim& GWish\left(d,D\right) \cr G \sim&\pi\left(G\right)\end{eqnarray*}}
#' The prior chosen for the precision matrix is a GWishart, whose parameters can be fixed in \code{HyParam}.
#' Diffefent priors are available for the graph, they can be set via \code{prior} input.
#' @param data two possibilities are available. (1) a \mjseqn{p \times n} matrix corresponding to \code{n} observation of \code{p}-dimensional variables or (2)
#' a \mjseqn{p \times p} matrix representing \mjseqn{\sum_{i=1}^{n}(y_{i}y_{i}^{T})}, where \mjseqn{y_{i}} is the \code{i}-th observation of a \code{p}-dimensional variale.
#' Data are assumed to be zero mean.
#' @param n integer, number of observed data.
#' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be \mjseqn{(niter - burnin)/thin}.
#' @param burnin number of discarded iterations.
#' @param thin the thining value, it means that only one out of \code{thin} itarations is saved.
#' @param Param list containing parameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{sampler_parameters}} function. It is indeed recommended to build it through that particular function. \code{BaseMat} field is not needed.
#' @param HyParam list containing hyperparameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{GM_hyperparameters}} function. It is indeed recommended to build it through that particular function.
#' @param Init list containig initial values for Markov chain. It has to follow the same notation of the one generated by \code{\link{GM_init}} function. It is indeed recommended to build it through that particular function.
#' @param file_name string, name of the file where the sampled values will be saved.
#' @param form string that states if the true graph has to be in \code{"Block"} or \code{"Complete"} form. Only possibilities are \code{"Complete"} and \code{"Block"}.
#' @param prior string with the desidered prior for the graph. Possibilities are \code{"Uniform"}, \code{"Bernoulli"} and for \code{"Block"} graphs only \code{"TruncatedBernoulli"} and \code{"TruncatedUniform"} are also available.
#' @param algo string with the desidered algorithm for sampling from a GGM. Possibilities are \code{"MH"}, \code{"RJ"} and \code{"DRJ"}.
#' @param groups a list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. If \code{"NULL"}, \code{"n_groups"} are automatically generated. Not needed if form is set to \code{"Complete"}.
#' @param n_groups number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as \code{groups} parameter.
#' @param seed integer, seeding value. Set 0 for random seed.
#' @param print_info boolean, if true progress bar and execution time are displayed.
#' @return It returns a list composed of: \code{MeanK}, the posterior mean of all sampled precision matrix, \code{plinks} which contains the posterior probability of inclusion of each possible link.
#' It is a \mjseqn{p \times p} matrix if \code{form} is \code{"Complete"}, or a \mjseqn{n\_groups \times n\_groups} matrix if \code{form} is \code{"Block"}. \code{AcceptedMoves} contains the number of
#' Metropolis-Hastings moves that were accepted in the sampling, \code{VisitedGraphs} the number of graph that were visited at least once, \code{TracePlot_Gsize} is a vector of length \mjseqn{(niter - burnin)/thin}
#' such that each element is equal to the size of the visited graph in that particular iteration and finally \code{SampledGraphs} is a list containing all the visited graphs and their absolute frequence of visit.
#' To save memory, the graphs are represented only by the upper triangular part, stored row-wise.
#' A binary \code{".h5"} file named \code{file_name} is also generated, it contains all the sampled values.
#' @export
GGM_sampling = function( data, n, niter = 100000, burnin = niter/2, thin = 1, Param = NULL, HyParam = NULL, Init = NULL,
                         file_name = "GGMresults", form = "Complete", prior = "Uniform", algo = "RJ", groups = NULL, n_groups = 0, seed = 0, print_info = TRUE )
{
	p = dim(data)[1]
	if( (dim(data)[1] != dim(data)[2]) && (dim(data)[2] != n) )
		stop("Number of observations in data matrix is not coherent with parameter n")
	if(!(form == "Complete" || form == "Block"))
		stop("Only possible forms are Complete and Block")
	if(form == "Block" && is.null(groups) && n_groups <= 0)
		stop("Groups has to be available if Block form is selected.")
  #Checks Param / HyParam / Init structures
	if(is.null(Param))
		Param = BGSL:::sampler_parameters()
  else if(is.null(Param$MCprior) || is.null(Param$MCpost) || is.null(Param$threshold))
    stop("Param list is incorrectly set. Please use sampler_parameters() function to create it, or just leave NULL for default values.")

	if(is.null(HyParam))
		HyParam = BGSL:::GM_hyperparameters(p = p)
  else if(is.null(HyParam$D_K) || is.null(HyParam$b_K) || is.null(HyParam$Gprior) || is.null(HyParam$sigmaG) || is.null(HyParam$p_addrm))
    stop("HyParam list is incorrectly set. Please use GM_hyperparameters() function to create it, or just leave NULL for default values.")
 	if(dim(HyParam$D_K)[1]!= p){
		stop("Prior inverse scale matrix D is not coherent with data. It has to be a (p x p) matrix. Note that data has to be a (p x p) or a (p x n) matrix.")
	}

  if(is.null(Init))
    Init = BGSL:::GM_init(p = p, n = n, empty = TRUE , form = form, groups = groups, n_groups = n_groups)
  else if(is.null(Init$G0) || is.null(Init$K0))
    stop("Init list is incorrectly set. Please use GM_init() function to create it, or just leave NULL for default values.")

	if(form == "Block" && is.null(groups)){
		if(n_groups <= 1 || n_groups > p)
		  stop("Error, invalid number of groups inserted");
		groups = BGSL:::CreateGroups(p,n_groups)
	}

	if( dim(data)[1] != dim(data)[2] ){
		U = data%*%t(data)
		return (BGSL:::GGM_sampling_c( U, p, n, niter, burnin, thin, file_name,
                            HyParam$D_K, HyParam$b_K,
                            Init$G0, Init$K0,
								            Param$MCprior,Param$MCpost,Param$threshold,
                            form, prior, algo,
                            groups, seed, HyParam$Gprior,
                            HyParam$sigmaG,HyParam$p_addrm, print_info ) )
	}else{
		return (BGSL:::GGM_sampling_c( data, p, n, niter, burnin, thin, file_name,
                            HyParam$D_K, HyParam$b_K,
                            Init$G0, Init$K0,
								            Param$MCprior,Param$MCpost,Param$threshold,
                            form, prior, algo,
                            groups, seed, HyParam$Gprior,
                            HyParam$sigmaG,HyParam$p_addrm, print_info ) )
	}
}

#' Skeleton for Hyperparameters in Graphical Models
#'
#' \loadmathjax This function simply creates a skeleton for all the hyperparameters that has to be fixed in graphical samplers, that are \code{\link{GGM_sampling}} and \code{\link{FGM_sampling}}.
#' Prefer to use \code{\link{LM_hyperparameters}} for \code{\link{FLM_sampling}}.
#' All the quantities has a default value, moreover some of them may not be needed in some algorithm. The goal of this function is to fix a precise notation that will be used in all the code.
#' @param p integer, the dimension of the underlying graph.
#' @param a_tau_eps Shape parameter for gamma prior distribution of \code{tau_eps} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param b_tau_eps Rate parameter for gamma prior distribution of \code{tau_eps} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param sigma_mu  covariance for normal multivariate distribution of \code{mu} parameter. Needed in Functional Models, i.e \code{FGM} and \code{FLM} samplers.
#' @param b_K  prior GWishart Shape parameter. It has to be larger than 2 in order to have a well defined distribution. Needed in all graphical model, i.e \code{FGM} and \code{GGM} sampler and in \code{FLM} with fixed graph.
#' @param D_K matrix of size \mjseqn{p \times p}, it is prior GWishart Inverse-Scale parameter. It has to be symmetric and positive definite. Default is identity matrix. Needed in all graphical model, i.e \code{FGM} and \code{GGM} sampler and in \code{FLM} with fixed graph.
#' @param p_addrm the probability of proposing a new graph by adding one link. It has to be in the range \mjseqn{(0,1)}. Needed in all graphical model where the graph is random, i.e \code{FGM} and \code{GGM} sampler.
#' @param sigmaG  the standard deviation used to perturb elements of precision matrix when constructing the new proposed matrix.
#' Needed in all graphical model where the graph is random, i.e \code{FGM} and \code{GGM} sampler. If algorithm is \code{"MH"} it is not used.
#' @param Gprior represents the prior probability of inclusion of each link in case \code{"Bernoulli"} prior is selected for the graph. It has to be in the range \mjseqn{(0,1)}.
#' Set 0.5 for \code{"Uniform"} prior. Needed in all graphical model where the graph is random, i.e \code{FGM} and \code{GGM} sampler.
#' @return A list with all the hyperparameters described as possible inputs. All values that are not explicitely provided are defaulted.
#' @export
GM_hyperparameters = function(p, a_tau_eps = 20, b_tau_eps = 0.002, sigma_mu = 100, b_K = 3,
                              D_K = NULL, p_addrm = 0.5, sigmaG = 0.5, Gprior = 0.5)
{
	if(is.null(D_K)){
			D_K = diag(p)
	}else{
		if(dim(D_K)[1] != dim(D_K)[2] || p != dim(D_K)[2])
			stop("Inverse scale matrix D_K has to be (p x p) squared symmetric positive definite matrix.")
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
#' \loadmathjax This function simply creates a skeleton for all the hyperparameters that has to be fixed in linear samplers, i.e \code{\link{FLM_sampling}}. All the quantities has a default value,
#' moreover some of them may not be needed in some algorithm. The goal of this function is to fix a precise notation that will be used in all the code.
#' This function is meant to set only the skeleton for \code{\link{FLM_sampling}}, use \code{\link{GM_hyperparameters}} for \code{\link{GGM_sampling}} and \code{\link{FGM_sampling}}.
#' @param p integer, the dimension of the underlying graph.
#' @param a_tau_eps Shape parameter for gamma prior distribution of \code{tau_eps} parameter.
#' @param b_tau_eps Rate parameter for gamma prior distribution of \code{tau_eps} parameter.
#' @param sigma_mu  covariance for normal multivariate distribution of mu parameter.
#' @param b_K  prior GWishart Shape parameter. It has to be larger than 2 in order to have a well defined distribution. Needed only when using \code{\link{FLM_sampling}} with fixed graph.
#' @param D_K  matrix of size \mjseqn{p \times p}. It is prior GWishart inverse scale parameter. It has to be symmetric and positive definite.
#' Default is identity matrix but \code{p} has to be provided in that case. Needed only when using \code{\link{FLM_sampling}} with fixed graph.
#' @param a_tauK  Shape parameter for gamma prior distribution for each \code{tau_K} parameters. Needed only when using \code{\link{FLM_sampling}} with diagonal graph.
#' @param b_tauK  Rate parameter for gamma prior distribution for each \code{tau_K} parameters. Needed only when using \code{\link{FLM_sampling}} with diagonal graph.
#' @return A list with all the hyperparameters described as possible inputs. All values that are not explicitely provided are defaulted.
#' @export
LM_hyperparameters = function( p, a_tau_eps = 20, b_tau_eps = 0.002, sigma_mu = 100, b_K = 3,
                               D_K = NULL, a_tauK = 20, b_tauK = 0.002 )
{
  if(is.null(D_K)){
      D_K = diag(p)
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
#' \loadmathjax This function simply creates a skeleton for some parameters that has to be fixed in all samplers. All the quantities has a default value,
#' moreover some of them may not be needed in some algorithm. The goal of this function is to fix a precise notation that will be used in all the code.
#' @param MCprior integer, the number of iteration for the MonteCarlo approximation of prior normalizing constant of GWishart distribution. Needed in \code{"MH"} and \code{"RJ"} algorithms.
#' @param MCpost integer, the number of iteration for the MonteCarlo approximation of posterior normalizing constant of GWishart distribution. Needed only in \code{"MH"} algorithms.
#' @param BaseMat matrix of dimension \mjseqn{n\_grid\_points \times n\_basis} containing the evaluation of all the \mjseqn{n\_basis} B-spine basis function in all the grid points.
#' May be defaulted as \code{NULL} but note this case it is not automatically generated.
#' Note that it is not needed by \code{\link{GGM_sampling}} but is mandatory in \code{\link{FGM_sampling}} and \code{\link{FLM_sampling}}. For those cases,
#' use \code{\link{Generate_Basis}} to generete it.
#' @param Grid vector of length equal to the number of curves, it is used to specify on what points each curve has been measured. The numbering of the grid points is zero based.
#' @param threshold threshold for convergence in GWishart sampler. It is not needed only in \code{FLM} sampler with diagonal graph.
#' @return A list with all parameters described as possible inputs.
#' @export
sampler_parameters = function(MCprior = 500, MCpost = 750, BaseMat = NULL, Grid = NULL, threshold = 1e-14)
{

	param = list( "MCprior"   = MCprior,
				  "MCpost"    = MCpost,
				  "BaseMat"   = BaseMat,
          "Grid"      = Grid,
				  "threshold" = threshold )
	return (param)
}


#' Skeleton for initial values in Graphical model
#'
#' \loadmathjax This function simply creates a structure for setting initial values for Markov chians in all graphical samplers, that are \code{\link{GGM_sampling}} and \code{\link{FGM_sampling}}.
#' All the quantities has a default value, that can be set to empty chains or random initial points. Some quantities are not needed in \code{\link{GGM_sampling}}, but they are all initialized.
#' The goal of this function is to fix a precise notation that will be used in all the code.
#' Prefer to use \code{\link{LM_hyperparameters}} for \code{\link{FLM_sampling}}.
#' @param p integer, the dimension of the underlying graph.
#' @param n integer, the number of observed data.
#' @param empty boolean, if \code{TRUE} all chians start from zero. If not, it is possible to set the desidered value explicitely by means of the other function parameters or just leave them \code{NULL} and random initial points are generated.
#' @param G0 matrix representing the initial graph. If \code{form} is set to \code{"Complete"} it has to be a \mjseqn{p \times p} matrix,
#' if form is \code{"Block"} then \code{groups} or \code{n_groups} has to be provided and the graph should respect the structure given by those groups. Only the upper triangular part is needed. If \code{NULL}, a random graph is selected.
#' @param K0 numeric \mjseqn{p \times p} matrix representing the initial precision matrix. It has to be consistent with structure of graph \code{G0}. If \code{NULL}, a random matrix is selected.
#' @param Beta0 numeric \mjseqn{p \times n} matrix representing the initial beta coefficients. If \code{NULL}, they are randomly generated. Used only in \code{FGM_sampling}.
#' @param mu0 numeric \mjseqn{p}-dimensional vector representing the initial mu coefficients. If \code{NULL}, they are randomly generated. Used only in \code{FGM_sampling}.
#' @param tau_eps0 scalar number representing the initial \code{tau_eps} coefficient. It has to be strictly positive. If \code{NULL}, it is randomly generated. Used only in \code{FGM_sampling}.
#' @param form string that may take as values only \code{"Complete"} of \code{"Block"} . It states if the algorithm has to run with \code{"Block"} or \code{"Complete"} graphs.
#' @param groups a list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. If \code{"NULL"}, \code{"n_groups"} are automatically generated. Not needed if form is set to \code{"Complete"}.
#' @param n_groups integer, number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as group parameter.
#' @param seed integer, seeding value. Set 0 for random seed.
#'
#' @return A list with all the previously described initial values.
#' @export
GM_init = function(p ,n, empty = TRUE, G0 = NULL, K0 = NULL, Beta0 = NULL, mu0 = NULL, tau_eps0 = NULL, form = "Complete", groups = NULL, n_groups = 0, seed = 0)
{
	if(!(form == "Complete" || form == "Block"))
		stop("Only possible forms are Complete and Block")
	if(form == "Block" && is.null(groups) && n_groups <= 0)
		stop("Groups has to be available if Block form is selected.")
  if(!is.null(tau_eps0) && tau_eps0 <= 0)
    stop("tau_eps0 has to be strictly positive.")
  if(empty){
    if(form == "Complete"){
      G0 = diag(p)
    }else if(form == "Block"){
      if(is.null(groups)){
        if(n_groups <= 1 || n_groups > p)
          stop("Error, invalid number of groups inserted")
        groups = BGSL:::CreateGroups(p,n_groups)
      }
      G0 = diag(length(groups))
    }
    else
      stop("Only possible forms are Complete and Block")
    K0 = diag(p)
    tau_eps0 = 1
    mu0 = rep(0,p)
    Beta0 = matrix(0,nrow = p, ncol = n)
  }else{
    if(is.null(G0)){
      if(form == "Complete"){
        G0 = BGSL:::Create_RandomGraph(p = p, form = "Complete",  groups = NULL, sparsity = 0.5, seed = seed  )$G
      }
      else if(form == "Block"){
        if(is.null(groups)){
          if(n_groups <= 1 || n_groups > p)
            stop("Error, invalid number of groups inserted")

          groups = BGSL:::CreateGroups(p,n_groups)
        }
        G0 = BGSL:::Create_RandomGraph(p = p, form = "Block",  groups = groups, sparsity = 0.5, seed = seed  )$G
        Gcomp = BGSL:::Block2Complete(G0, groups)
      }
      else
        stop("Only possible forms are Complete and Block")
    }
    if(is.null(K0)){
      Dtemp = diag(p)
      K0 = BGSL:::rGwish(G0, 3, Dtemp, groups = groups, threshold_conv = 1e-16, seed = seed)$Matrix
    }

    #Check correctness
    if(form == "Block"){
        if(!(BGSL:::check_structure(Gcomp, K0)))
          stop("Matrix K does not respect structure of graph G0")
    }
    if(form == "Complete"){
      if(!(BGSL:::check_structure(G0, K0)))
        stop("Matrix K does not respect structure of graph G0")
    }

    #Quantities needed only in FGM
    if(is.null(tau_eps0)){
      tau_eps0 = stats::rgamma(n = 1, shape = 50, rate = 2)
    }
    if(is.null(mu0)){
      mu0 = BGSL:::rmvnormal(mean = rep(0,p), Mat = diag(p), isPrec = FALSE)
    }
    if(is.null(Beta0)){
      Beta0 = matrix(0,nrow = p, ncol = n)
      for(i in 1:n){
        Beta0[,i] = BGSL:::rmvnormal(mean = mu0, Mat = K0, isPrec = TRUE, isChol = FALSE)
      }
    }
  }

  init = list( "G0"   = G0,
               "K0"    = K0,
               "Beta0"   = Beta0,
               "mu0"   = mu0,
               "tau_eps0"   = tau_eps0 )
  return (init)
}



#' Skeleton for initial values in Linear model
#'
#' \loadmathjax This function simply creates a structure for setting initial values for Markov chians in \code{\link{FLM_sampling}}. All the quantities has a default value,
#' that can be set to empty chains or random initial points. Some quantities are not needed, it depends on what version of \code{\link{FLM_sampling}} is called. However they are all initialized.
#' The goal of this function is to fix a precise notation that will be used in all the code.
#' This function is meant to set only the initial values for \code{\link{FLM_sampling}}, use \code{\link{GM_init}} for \code{\link{GGM_sampling}} and \code{\link{FGM_sampling}}.
#' @param p integer, the dimension of the underlying graph.
#' @param n integer, the number of observed data.
#' @param empty boolean, if \code{TRUE} all chians start from zero. If not, it is possible to set the desidered value explicitely by means of the other function parameters or just leave them \code{NULL} and random initial points will be used.
#' @param K0 numeric \mjseqn{p \times p} matrix representing the initial precision matrix. It will not actually be used if \code{\link{FLM_sampling}} is called with \code{diagonal_graph} set to \code{TRUE}.
#' If \code{NULL}, a random matrix is selected.
#' @param Beta0 numeric \mjseqn{p \times n} matrix representing the initial beta coefficients. If \code{NULL}, they are randomly generated.
#' @param mu0 numeric \mjseqn{p}-dimensional vector representing the initial mu coefficients. If \code{NULL}, they are randomly generated.
#' @param tau_eps0 scalar number representing the initial \code{tau_eps} coefficient. If \code{NULL}, it is randomly generated.
#' @param tauK0 numeric \mjseqn{p}-dimensional vector representing the initial values for the diagonal of precision matrix. It will not actually be used if \code{\link{FLM_sampling}} is called with \code{diagonal_graph} set to \code{FALSE}.
#' If \code{NULL}, they are randomly generated.
#' @param seed integer, seeding value. Set 0 for random seed.
#'
#' @return A list with all the previously described initial values.
#' @export
#'
LM_init = function(p ,n, empty = TRUE, K0 = NULL, Beta0 = NULL, mu0 = NULL, tau_eps0 = NULL, tauK0 = NULL, seed = 0)
{

  if(empty){
    K0 = diag(p)
    tau_eps0 = 1
    mu0 = rep(0,p)
    tauK0 = rep(1,p)
    Beta0 = matrix(0,nrow = p, ncol = n)
  }else{
    #Not checking if it respect the true graphs structure
    if(is.null(K0)){
      I = diag(p)
      K0 = BGSL:::rGwish(I, 3, I, groups = NULL, threshold_conv = 1e-16, seed = seed)$Matrix
    }
    if(is.null(tau_eps0)){
      tau_eps0 = stats::rgamma(n = 1, shape = 50, rate = 2)
    }
    if(is.null(mu0)){
      mu0 = BGSL:::rmvnormal(mean = rep(0,p), Mat = diag(p), isPrec = FALSE)
    }
    if(is.null(tauK0)){
      tauK0 = rep(0,p)
      for(i in 1:p){
        tauK0[i] = stats::rgamma(n = 1, shape = 50, rate = 2)
      }
    }
    if(is.null(Beta0)){
      Beta0 = matrix(0,nrow = p, ncol = n)
      for(i in 1:n){
        Beta0[,i] = BGSL:::rmvnormal(mean = mu0, Mat = K0, isPrec = TRUE, isChol = FALSE)
      }
    }
  }

  init = list( "K0"    = K0,
               "Beta0"   = Beta0,
               "mu0"   = mu0,
               "tauK0"   = tauK0,
               "tau_eps0"   = tau_eps0 )
  return (init)
}


#' Tuning \code{sigmaG} parameter
#'
#' \loadmathjax The choice of \code{sigmaG} parameter is a crucial part of \code{"RJ"} and \code{"DRJ"} algorithm. It indeed represents the standard deviation for the proposal of new elements in the
#' precision matrix. This function provides an utility for tuning this parameter by simply repeting \code{Nrep} times the first \code{niter} iteration for every proposed
#' \code{sigmaG} in \code{sigmaG_list}.
#' @param data two possibilities are available. (1) a \mjseqn{p \times n} matrix corresponding to \code{n} observation of \code{p}-dimensional variables or (2)
#' a \mjseqn{p \times p} matrix representing \mjseqn{\sum_{i=1}^{n}(y_{i}y_{i}^{T})}, where \mjseqn{y_{i}} is the \code{i}-th observation of a \code{p}-dimensional variale.
#' Data are assumed to be zero mean.
#' @param n number of observed values.
#' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be \mjseqn{(niter - burnin)/thin}.
#' @param sigmaG_list list containing the values of \code{sigmaG} that has to be tested.
#' @param Nrep how many times each \code{sigmaG} has to be tested.
#' @param Param list containing parameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{sampler_parameters}} function. It is indeed recommended to build it through that particular function. \code{BaseMat} field is not needed.
#' @param HyParam list containing hyperparameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{GM_hyperparameters}} function. It is indeed recommended to build it through that particular function.
#' @param Init list containig initial values for Markov chain. It has to follow the same notation of the one generated by \code{\link{GM_init}} function. It is indeed recommended to build it through that particular function.
#' @param form string that states if the true graph has to be in \code{"Block"} or \code{"Complete"} form. Only possibilities are \code{"Complete"} and \code{"Block"}.
#' @param prior string with the desidered prior for the graph. Possibilities are \code{"Uniform"}, \code{"Bernoulli"} and for \code{"Block"} graphs only \code{"TruncatedBernoulli"} and \code{"TruncatedUniform"} are also available.
#' @param algo string with the desidered algorithm for sampling from a GGM. Possibilities are \code{"MH"}, \code{"RJ"} and \code{"DRJ"}.
#' @param groups a list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. If \code{"NULL"}, \code{"n_groups"} are automatically generated. Not needed if form is set to \code{"Complete"}.
#' @param n_groups number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as \code{groups} parameter.
#' @param seed integer, seeding value. Set 0 for random seed.
#' @return plots the mean acceptance ratio for each \code{sigmaG} and returns the highest one.
#' @export
sigmaG_GGM_tuning = function( data, n, niter = 1000, sigmaG_list = seq(0.05, 0.55, by = 0.05), Nrep = 10,
							                Param = NULL, HyParam = NULL,  Init = NULL, n_groups = 0, groups = NULL,
                              form = "Complete", prior = "Uniform", algo = "RJ", seed = 0  )
{
	nburn = niter-1
	thin  = 1
	accepted = rep(0,length(sigmaG_list))
	counter = 1
  file_name = "temp_sigmaGTuning"
  file_name_ext = "temp_sigmaGTuning.h5"
	if(is.null(Param))
		Param = BGSL:::sampler_parameters()
	if(is.null(HyParam))
		HyParam = BGSL:::GM_hyperparameters(p = p)
  if(is.null(Init))
    Init = BGSL:::GM_init(p = p, n = n, empty = TRUE , form = form, groups = groups, n_groups = n_groups)

	for(sigmaG in sigmaG_list){
		cat('\n Starting sigmaG = ',sigmaG,'\n')
		#pb <- txtProgressBar(min = 1, max = Nrep, initial = 1, style = 3)
	  	tot = 0
	  	HyParam$sigmaG = sigmaG
	  for(i in 1:Nrep){
  	    result = BGSL:::GGM_sampling(	data = data, n = n, niter = niter, burnin = nburn, thin = thin,
                         			        Param = param, HyParam = hy, Init = init, groups = NULL, n_groups = n_groups,
                        			        prior = prior, form = form, algo = algo, file_name = file_name,
                        			        seed = seed, print_info = FALSE  )
        if(length(result) == 0)
          stop("GGM_sampling is returning an empty list. Execution stops")
	    tot = tot + result$AcceptedMoves
	    #setTxtProgressBar(pb, i)
	    cat('\n  i = ',i,', acc = ',result$AcceptedMoves,'\n')
	  }
	  #close(pb)
	  accepted[counter] = tot/Nrep
	  counter = counter + 1
	}
	accepted = accepted/niter
  suppressMessages(file.remove(file_name_ext))
  col_plot = rep("black", length(sigmaG_list))
  col_plot[which.max(accepted)] = "red"
	x11();plot(accepted, pch = 16, col = col_plot, xlab = "sigmaG", ylab = "Acceptance Ratio", xaxt = 'n')
  title(main = "Tuning sigmaG parameter")
	mtext(text = sigmaG_list, side=1, line=0.3, at = 1:length(sigmaG_list) , las=2, cex=1.0)
	return (sigmaG_list[which.max(accepted)])
}


#' Block graph to Complete graph map
#'
#' \loadmathjax Function for mapping from a block matrix to its non-block representation. It is the \mjseqn{\rho} described in the report.
#' It can be used also for matrices whose entries are not just 1 or 0, in particular it is useful to map the block form of the \code{plinks} matrix into its complete form.
#' @param Gblock matrix of size \mjseqn{n\_groups \times n\_groups} to be mapped.
#' @param groups list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. It is possible to generete it through CreateGroups function.
#' @return It returns a \mjseqn{p \times p} matrix containing the complete form of \code{Gblock}.
#' @export
Block2Complete = function(Gblock, groups)
{
  m = length(groups)
  if(dim(Gblock)[1] != dim(Gblock)[2])
    stop("The inserted Gblock graph has to the a squared matrix.")
  if(dim(Gblock)[1]!=m)
    stop("The dimension of the inserted graph is not coherent with the number of provided groups.")
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


#' Functional Graphical model for smoothing
#'
#' \loadmathjax This function implements an hybrid Gibbs Sampler strategy to draw samples from the posterior distribution of a Functional Graphical Model, reported below.
#' \mjtdeqn{$$\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} \sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i = 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K\sim~&N_{p}\left(\mu,K\right) \cr \mu\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} \sim~& Gamma\left(a,b\right) \cr K~|~ G ~\sim~& GWish\left(d,D\right) \cr G ~\sim~&\pi\left(G\right) \end{eqnarray*}$$}{\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} &\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i &= 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K&\sim~&N_{p}\left(\mu,K\right) \cr \mu&\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} &\sim~& Gamma\left(a,b\right) \cr K~|~ G &\sim~& GWish\left(d,D\right) \cr G &\sim~&\pi\left(G\right) \end{eqnarray*}}{\begin{eqnarray*}Y_{i}~|~\beta_{i},~\tau_{\epsilon} &\sim~&N_{r}\left(\Phi\beta_{i},\tau_{\epsilon}I_{r}\right), &\forall i &= 1:n \cr \beta_{1},\dots,\beta_{n}~|~\mu,~K&\sim~&N_{p}\left(\mu,K\right) \cr \mu&\sim~&N_{p}\left(0,\sigma_{\mu}I_{p}\right) \cr \tau_{\epsilon} &\sim~& Gamma\left(a,b\right) \cr K~|~ G &\sim~& GWish\left(d,D\right) \cr G &\sim~&\pi\left(G\right) \end{eqnarray*}}
#' It has a double goal, performing a smoothing of the inserted noisy curves and estimating the graph which describes the relationship among the regression coefficients.
#' @param p integer, the number of basis functions. It also represents the dimension of the true underlying graph.
#' @param data matrix of dimension \mjseqn{n\_grid\_points \times n} containing the evaluation of \code{n} functional data over a grid of \code{n_grid_points} nodes.
#' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be \mjseqn{(niter - burnin)/thin}.
#' @param burnin the number of discarded iterations.
#' @param thin the thining value, it means that only one out of \code{thin} itarations is saved.
#' @param thinG the thining value for graphical quantities, it means that Graph and Precision are only saved one every \code{thinG} itarations.
#' @param Param list containing parameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{sampler_parameters}} function.
#' It is indeed recommended to build it through that particular function. It is important to remember that \code{BaseMat} field is needed and cannot be defaulted. Use \code{\link{Generate_Basis}} to create it.
#' It has to be a matrix of dimension \mjseqn{n\_grid\_points \times p} containing the evalutation of \code{p} Bspline basis over a grid of \code{n_grid_points} nodes.
#' @param HyParam list containing hyperparameters needed by the sampler. It has to follow the same notation of the one generated by \code{\link{GM_hyperparameters}} function. It is indeed recommended to build it through that particular function.
#' @param Init list containig initial values for Markov chain. It has to follow the same notation of the one generated by \code{\link{GM_init}} function. It is indeed recommended to build it through that particular function.
#' @param file_name string, name of the file where the sampled values will be saved.
#' @param form string that may take as values only \code{"Complete"} of \code{"Block"} . It states if the algorithm has to run with \code{"Block"} or \code{"Complete"} graphs.
#' @param prior string with the desidered prior for the graph. Possibilities are \code{"Uniform"}, \code{"Bernoulli"} and for \code{"Block"} graphs only \code{"TruncatedBernoulli"} and \code{"TruncatedUniform"} are also available.
#' @param algo string with the desidered algorithm for sampling from a GGM. Possibilities are \code{"MH"}, \code{"RJ"} and \code{"DRJ"}.
#' @param groups a list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
#' i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. If \code{"NULL"}, \code{"n_groups"} are automatically generated. Not needed if form is set to \code{"Complete"}.
#' @param n_groups int, number of desired groups. Not used if form is \code{"Complete"} or if the groups are directly insered as group parameter.
#' @param print_info boolean, if true progress bar and execution time are displayed.
#' @param seed integer, seeding value. Set 0 for random seed.
#'
#' @return Two lists are returned, the first one, called \code{PosteriorMeans}, is composed of the posterior mean of those parameters related to the functional smoothing. The second one,
#' called \code{GraphAnalysis} is a summary of the sampled graphs. It is the same output of \code{\link{GGM_sampling}}, see its documentation for further details.
#' @export
FGM_sampling = function( p, data, niter = 100000, burnin = niter/2, thin = 1, thinG = 1, Param = NULL, HyParam = NULL, Init = NULL,
                         file_name = "FGMresults", form = "Complete", prior = "Uniform", algo = "RJ", groups = NULL, n_groups = 0,
                         print_info = TRUE, seed = 0  )
{
  n = dim(data)[2]
  if(!(form == "Complete" || form == "Block"))
    stop("Only possible forms are Complete and Block")
  if(form == "Block" && is.null(groups) && n_groups <= 0)
    stop("Groups has to be available if Block form is selected.")

  #Checks Param / HyParam / Init structures
  if(is.null(Param))
    Param = BGSL:::sampler_parameters()
  else if(is.null(Param$BaseMat) || is.null(Param$threshold) || is.null(Param$MCprior) || is.null(Param$MCpost))
    stop("Param list is incorrectly set. Please use sampler_parameters() function to create it. Hint: in Functional Models, BaseMat field cannot be defaulted. Use Generate_Basis() to create it.")
  if(is.null(HyParam))
    HyParam = BGSL:::GM_hyperparameters(p = p)
  else if( is.null(HyParam$D_K) || is.null(HyParam$b_K) || is.null(HyParam$a_tau_eps) || is.null(HyParam$b_tau_eps) ||
           is.null(HyParam$sigma_mu) || is.null(HyParam$p_addrm) || is.null(HyParam$sigmaG) || is.null(HyParam$Gprior) )
    stop("HyParam list is incorrectly set. Please use GM_hyperparameters() function to create it, or just leave NULL for default values.")
  if(dim(HyParam$D_K)[1]!= p){
    stop("Prior inverse scale matrix D is not coherent with the number of basis function. It has to be a (p x p) matrix.")
  }
  if(is.null(Init))
    Init = BGSL:::GM_init(p = p, n = n, empty = TRUE, groups = groups, n_groups = n_groups )
  else if(is.null(Init$Beta0) || is.null(Init$K0) || is.null(Init$mu0) || is.null(Init$tau_eps0) || is.null(Init$G0) )
    stop("Init list is incorrectly set. Please use GM_init() function to create it, or just leave NULL for default values.")

  #Create groups if deafulted
  if(form == "Block" && is.null(groups)){
    if(n_groups <= 1 || n_groups > p)
      stop("Error, invalid number of groups inserted");
    groups = BGSL:::CreateGroups(p,n_groups)
  }

  #Launch sampler
  return (
          BGSL:::FGM_sampling_c( data, niter, burnin, thin, thinG,  #data and iterations
                          Param$BaseMat, file_name,  #Basemat and name of file
                          Init$Beta0, Init$mu0, Init$tau_eps0, Init$G0, Init$K0,  #initial values
                          HyParam$a_tau_eps, HyParam$b_tau_eps,  HyParam$sigma_mu, HyParam$b_K, HyParam$D_K, HyParam$sigmaG,  HyParam$p_addrm , HyParam$Gprior, #hyperparam
                          Param$MCprior,  Param$MCpost, Param$threshold,  #GGM_parameters
                          form , prior, algo , groups , seed, print_info
                        )

    )
}




#' Heatmap of a matrix.
#'
#' It plots the heatmap of a given matrix. It is usefull because it allows the user to use custom, non-symmetric palettes, centered around a desired value.
#' @param Mat the matrix to be plotted.
#' @param col.upper the color for the highest value in the matrix.
#' @param col.center the color for the centered value (if desired) or for the middle value in the matrix (oterhwise).
#' @param col.lower the color for the lowest value in the matrix.
#' @param center_value the value around which the palette has to be centered. Set NULL for symmetric palette.
#' @param col.n_breaks has to be odd. The refinement of the palette.
#' @param use_x11_device boolean, if x11() device has to be activated or not.
#' @param main the title of the plot.
#' @param x_label the name of the x-axis in the plot.
#' @param y_label the name of the y-axis in the plot.
#' @param remove_diag boolean, if the diagonal has to be removed or not.
#' @param horizontal boolean, if the heatmap bar has to be horizontal or not.
#' @return this function does not return anything
#'
#' @export
ACheatmap = function(Mat, center_value = 0, col.upper = "darkred", col.center = "grey95", col.lower = "#3B9AB2",
                     col.n_breaks = 59, use_x11_device = TRUE, remove_diag = FALSE, main = " ", x_label = " ",
                     y_label = " ", horizontal=TRUE )
{

  #library(fields)
  #Check for NA
  if(any(is.na(Mat))){
    cat('\n NA values have been removed from Matrix  \n')
  }

  #Create color palette
  if(col.n_breaks %% 2 == 0){
    warning( 'col.n_breaks is even but it has to be odd. Adding 1' )
    col.n_breaks = col.n_breaks + 1
  }
  colorTable = fields::designer.colors(col.n_breaks, c( col.lower, col.center, col.upper) )
  col_length = (col.n_breaks + 1) / 2

  #Check Matrix
  if(remove_diag){
    diag(Mat) = NA
  }
  min_val = min(Mat,na.rm=T)
  max_val = max(Mat,na.rm=T)
  p_row = dim(Mat)[1]
  p_col = dim(Mat)[2]


  #Plot
  if(!is.null(center_value)){

    if(!(min_val < center_value & max_val > center_value)){
      stop('\n The lowest value has to be smaller than center_value and the highest value has to be larger. \n')
    }
    brks = c(seq( min_val, center_value-0.0001,l=col_length), seq( center_value+0.0001, max_val, l=col_length))

  }else{
    brks = seq( min_val, max_val, l=2*col_length)
  }

  colnames(Mat) = 1:p_col
  rownames(Mat) = 1:p_row

  if(use_x11_device){
    x11()
  }

  par(mar=c(5.1, 4.1, 4.1, 2.1),mgp=c(3,1,0))
  if(horizontal){

    fields::image.plot(Mat, axes=F, horizontal=T, main = main,
                       col=colorTable,breaks=brks,xlab=x_label,ylab=y_label)
  }else{

    fields::image.plot(Mat, axes=F, horizontal=FALSE, main = main,
                       col=colorTable,breaks=brks,xlab=x_label,ylab=y_label)
  }
  box()

}



#' KL_dist
#'
#' @param Ktr true precision matrix
#' @param K estimated precision
#'
#' @export
KL_dist = function(Ktr,K){
  p = dim(K)[1]
  inv_Ktr = solve(Ktr)
  A = sum( diag(inv_Ktr%*%K) )
  B = log( det(K)/det(Ktr) )
  return (0.5*(A - p - B))
}
#' upper2complete
#'
#' @param A matrix
#'
#' @export
upper2complete = function(A){
  diag_A = diag(A)
  A = A + t(A)
  diag(A) = diag_A
  return(A)
}
#' Graph_distance
#'
#' @param G1 first graph
#' @param G2 second graph
#' @param diag if diagonal should be included or not
#' @export
Graph_distance = function(G1,G2,diag = F){

  G1_vett = G1[upper.tri(G1,diag = diag)]
  G2_vett = G2[upper.tri(G1,diag = diag)]
  Table   = table(G1_vett, G2_vett)

  return ( Table[1,2] + Table[2,1] )
}
