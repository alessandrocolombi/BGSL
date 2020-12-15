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
#' @param beta matrix of dimension n_basis x n_curves containing the values of regression coefficients
#' @param BaseMat matrix of dimension n_grid_points x n_basis containing the evaluation of all the spline in all the grid points
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param range the range where the curves has to be plotted. Not needed if n_plot is 0.
#' @param grid_points vector of size n_grid_points with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if n_plot is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function Generate_Basis. Default is null.
#' but if provided, n_basis are displayed in the plot. The k-th interval represents the segment where the k-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis]
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis]
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
#' It does not computes the quantiles nor the mean starting from all the sampled values, they have to be previously computed using Compute_QuantileBeta function.
#' @param beta matrix of dimension n_basis x n_curves containing the mean values of regression coefficients
#' @param betaLower matrix of dimension n_basis x n_curves containing the lower quantiles values of regression coefficients. Can be obtained by Compute_QuantileBeta function.
#' @param betaUpper matrix of dimension n_basis x n_curves containing the upper quantiles values of regression coefficients. Can be obtained by Compute_QuantileBeta function.
#' @param BaseMat matrix of dimension n_grid_points x n_basis containing the evaluation of all the spline in all the grid points
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param range the range where the curves has to be plotted. Not needed if n_plot is 0.
#' @param grid_points vector of size n_grid_points with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if n_plot is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function Generate_Basis. Default is null.
#' but if provided, n_basis are displayed in the plot. The k-th interval represents the segment where the k-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis]
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis]
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
#' the regression coefficients. Use smooth_curves function for that.
#' @param data1 matrix of dimension n_curves x n_grid_points representing the first dataset to be plotted.
#' @param data2 matrix of dimension n_curves x n_grid_points representing the secondo dataset to be plotted, if needed.
#' @param range the range where the curves has to be plotted. Not needed if n_plot is 0.
#' @param n_plot the number of curves to be plotted. Set 0 for no plot.
#' @param grid_points vector of size n_grid_points with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if n_plot is 0.
#' @param internal_knots vector with the internal knots used to construct the splines. Can be obtained as return value of function Generate_Basis. Default is null.
#' but if provided, n_basis are displayed in the plot. The k-th interval represents the segment where the k-th spline dominates the others.
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis]
#' @param highlight_band1 a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis]
#' @param title_plot the title of the plot.
#' @param xtitle the title of the x-axis.
#' @param ytitle the title of the x-axis.
#' @param legend_name1 the name for data1 to be printed in the legend. Used only is two datasets are actually plotted.
#' @param legend_name2 the name for data2 to be printed in the legend.
#'
#' @export
plot_curves = function( data1, data2 = NULL, range, n_plot = 1, grid_points = NULL,
                        internal_knots = NULL, highlight_band1 = NULL, highlight_band2 = NULL,
                        title_plot = "Curves", xtitle = " ", ytitle = " ", legend_name1 = "data1", legend_name2 = "data2"){
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
#' @param data matrix of dimension r x n containing the evaluation of n functional data over a grid of r nodes.
#' @param niter the number of total iterations to be performed in the sampling. The number of saved iteration will be (niter - burnin)/thin.
#' @param burnin the number of discarded iterations.
#' @param thin the thining value, it means that only one out of thin itarations is saved.
#' @param BaseMat matrix of dimension r x p containing the evalutation of p Bspline basis over a grid of r nodes
#' @param diagonal_graph boolean, set true if the graph has to be diagonal. Set false otherwise and pass as input the desired graph through the G parameter.
#' @param threshold_GWish double, stop algorithm for GWishart samling if the difference between two subsequent iterations is less than threshold_conv. Used only if diagonal_graph is set to false.
#' @return Two lists are returned, one with all the sampler values and the other with the posterior means.
#' @export
FLM_sampling = function(data, niter, burnin, thin, BaseMat, diagonal_graph = T, G = NULL ,threshold_GWish = 1e-8)
{
  if(diagonal_graph)
  {
    G = matrix(0,2,2)
    return (FLM_sampling_c(data, niter,burnin,thin,BaseMat,G,diagonal_graph,threshold_GWish))
  }
  else
    return (FLM_sampling_c(data, niter,burnin,thin,BaseMat,G,diagonal_graph,threshold_GWish));
}



#' Simulate curves 
#'
#' This function genrates a dataset of n functional curves. Data are generated trying to simulate the shape of one or two Gaussian distributions.
#' @param p Dimension of the true underlying graph.
#' @param n Number of curves to be generated.
#' @param n_grid_points dimension of the grid where the spline are evaluated.
#' @param range_x vector of length two defining the interval for the curves.
#' @param G (p x p) matrix representing the true underlying graph. Default is NULL that corresponds to a diagonal graph.
#' @param K (p x p) matrix representing the true underlying precision matrix. If NULL, a sample from a GWishart is drawn.
#' @param b GWishart shape parameter used if the matrix K has to be drawn. It has to be larger than 2 in order to have a well defined distribution. Default is three.
#' @param D GWishart inverse scale matrix parameter used if the matrix K has to be drawn. It has to be symmetric and positive definite. Default is 0.01*diag(p).
#' @param spline_order order of the Bsplines. Set four for cubic splines.
#' @param n_picks The number of desired picks in the simulated curves. It may be one for a single, Gaussian like, shape or two for a double Gaussian shape.
#' @param height1 Heigth of the pick of the first Gaussian.
#' @param height2 Heigth of the pick of the second Gaussian.
#' @param width1 Set how large the first pick has to be. This parameter acts like a standard deviation of a Normal distribution, so the larger it is, the thinner is the pick.
#' @param width2 Set how large the second pick has to be. This parameter acts like a standard deviation of a Normal distribution, so the larger it is, the thinner is the pick.
#' @param position1 Set where the first pick has to be located. Position is then computed as range_x[1] + (range_x[2]-range_x[1])/position1. Set 2 to place it in the middle.
#' @param position2 Set where the second pick has to be located. Position is then computed as range_x[2] - (range_x[2]-range_x[1])/position2. Set 2 to place it in the middle.
#' @param n_plot The number of curves to be plotted. If it is set to 0, no curves are displayed.
#' @param highlight_band1 Vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis].
#' @param highlight_band1 Vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range [1,n_basis].
#' @param title_plot Title of the plot.
#' @param xtitle Title of the x-axis.
#' @param ytitle Title of the x-axis.
#' @param seed seed used to simulate data.
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
