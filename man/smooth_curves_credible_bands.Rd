% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/BGSL_export.R
\name{smooth_curves_credible_bands}
\alias{smooth_curves_credible_bands}
\title{Computes and plot smoothed curves with credible bands}
\usage{
smooth_curves_credible_bands(
  beta,
  betaLower,
  betaUpper,
  BaseMat,
  n_plot = 0,
  range = NULL,
  grid_points = NULL,
  internal_knots = NULL,
  highlight_band1 = NULL,
  highlight_band2 = NULL,
  title_plot = "Smoothed Curves",
  xtitle = " ",
  ytitle = " ",
  data = NULL
)
}
\arguments{
\item{beta}{matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the values of regression coefficients.}

\item{betaLower}{matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the lower quantiles values of regression coefficients. Can be obtained by \code{\link{Compute_Quantiles}} function.}

\item{betaUpper}{matrix of dimension \mjseqn{n\_basis \times n\_curves} containing the upper quantiles values of regression coefficients. Can be obtained by \code{\link{Compute_Quantiles}} function.}

\item{BaseMat}{matrix of dimension \mjseqn{n\_grid\_points \times n\_basis} containing the evaluation of all the \mjseqn{n\_basis} B-spine basis function in all the grid points.}

\item{n_plot}{the number of curves to be plotted. Set 0 for no plot.}

\item{range}{the range where the curves has to be plotted. Not needed if \code{n_plot} is 0.}

\item{grid_points}{vector of size \mjseqn{n\_grid\_points} with the points where the splines are evaluated. If defaulted they are uniformly generated. Not needed if \code{n_plot} is 0.}

\item{internal_knots}{vector with the internal knots used to construct the splines. Can be obtained as return value of function \code{\link{Generate_Basis}}. Default is null.
If provided, \code{n_basis} are displayed in the plot. The \code{k}-th interval represents the segment where the \code{k}-th spline dominates the others.}

\item{highlight_band1}{a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.}

\item{highlight_band2}{a vector that states if a particular band of the plot has to be highlighted. It has to be a vector within the range \mjseqn{\[1,n\_basis\]}.}

\item{title_plot}{the title of the plot.}

\item{xtitle}{the title of the x-axis.}

\item{ytitle}{the title of the x-axis.}

\item{data}{a \mjseqn{n\_curves \times n\_grid\_points} matrix containing the values of the true curves that are added to the plot if this parameter is not \code{NULL}.}
}
\value{
It returns a list composed of three \mjseqn{n\_curves \times n\_grid\_points} matrices called \code{MeanCurves}, \code{LowerBands} and \code{UpperBands}
containing the smoothed curves and their credible bands.
}
\description{
\loadmathjax This function gets the mean values of the regression coefficients as well as the lower and upper quantiles and builds the smooted curves with their credible bands.
It does not computes the quantiles nor the mean starting from all the sampled values, they have to be previously computed using \code{\link{Compute_Quantiles}} function.
}
