% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/0-backend.R
\name{to_loss}
\alias{to_loss}
\title{Get Loss for Output}
\usage{
to_loss(y)
}
\arguments{
\item{y}{A matrix or vector. The output data.}
}
\value{
A loss function.
}
\description{
This function guesses the loss function based on the
 type of outcome provided to \code{y}. It chooses from
 binary cross-entropy, categorical cross-entropy,
 or mean squared error.
}
