% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{CreateGroups}
\alias{CreateGroups}
\title{Create Groups}
\usage{
CreateGroups(p, n_groups)
}
\arguments{
\item{p}{integer, the dimension of the underlying graph.}

\item{n_groups}{number of desired groups. Has to be greater than \code{p}.}
}
\value{
list representing the groups of the block form.
}
\description{
\loadmathjax This function creates a list with the groups. If possible, groups of equal size are created. The goal of this function is to fix a precise notation that will be used in all the code.
It is indeed recommended to use this function to create them as they need to follow a precise notation.
}
