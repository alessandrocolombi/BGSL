% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{Compute_PosteriorMeans}
\alias{Compute_PosteriorMeans}
\title{Compute Posterior means of sampled values}
\usage{
Compute_PosteriorMeans(
  file_name,
  Beta = FALSE,
  Mu = FALSE,
  TauEps = FALSE,
  Precision = FALSE
)
}
\arguments{
\item{file_name, }{string with the name of the file to be open. It has to include the extension, usually \code{.h5}.}

\item{Beta}{boolean, set \code{TRUE} to compute the mean for all \code{p*n} \mjseqn{\beta} coefficients. It may require long time.}

\item{Mu}{boolean, set \code{TRUE} to compute the mean for all \mjseqn{p} parameters.}

\item{TauEps}{boolean, set \code{TRUE} to compute the mean of \mjseqn{\tau_{\epsilon}} parameter.}

\item{Precision}{boolean, set \code{TRUE} to compute the mean for all the elements of the precision matrix
or the \mjseqn{\tau_{j}} coefficients if the file contains the output of a \code{\link{FLM_sampling}}, diagonal version.}
}
\value{
It returns a list containig the mean of the requested quantities.
}
\description{
\loadmathjax This function reads the sampled values saved in a binary file and computes the mean of the requested quantities.
}
