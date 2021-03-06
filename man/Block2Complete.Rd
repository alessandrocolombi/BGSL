% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/BGSL_export.R
\name{Block2Complete}
\alias{Block2Complete}
\title{Block graph to Complete graph map}
\usage{
Block2Complete(Gblock, groups)
}
\arguments{
\item{Gblock}{matrix of size \mjseqn{n\_groups \times n\_groups} to be mapped.}

\item{groups}{list representing the groups of the block form. Numerations starts from 0 and vertrices has to be contiguous from group to group,
i.e ((0,1,2),(3,4)) is fine but ((1,2,3), (4,5)) and ((1,3,5), (2,4)) are not. It is possible to generete it through CreateGroups function.}
}
\value{
It returns a \mjseqn{p \times p} matrix containing the complete form of \code{Gblock}.
}
\description{
\loadmathjax Function for mapping from a block matrix to its non-block representation. It is the \mjseqn{\rho} described in the report.
It can be used also for matrices whose entries are not just 1 or 0, in particular it is useful to map the block form of the \code{plinks} matrix into its complete form.
}
