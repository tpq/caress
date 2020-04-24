#' Constrain Rows to Unit Sum
#'
#' This weights constraint function ensures that each row
#'  sums to 1. The interpretation here is that each edge from
#'  a node in the previous layer emits a part of the whole.
#'
#' @param w A weights matrix.
#' @export
constraint_rows_to_unit_sum <- function(w){

  w <- w * keras::k_cast(keras::k_greater_equal(w, 0), keras::k_floatx()) # if less than zero, make zero
  w <- keras::k_permute_dimensions(w, c(2, 1, (1:length(dim(w)))[-1][-1])) # swap row to column
  w <- w / (keras::k_sum(w, 1) + 1e-16) # add a bit of noise to prevent 0/0 error
  keras::k_permute_dimensions(w, c(2, 1, (1:length(dim(w)))[-1][-1])) # swap column to row
}

#' Constrain Columns to Unit Sum
#'
#' This weights constraint function ensures that each column
#'  sums to 1. The interpretation here is that each edge from
#'  a node in the new layer receives a part of the whole.
#'
#' @param w A weights matrix.
#' @export
constraint_cols_to_unit_sum <- function(w){

  w <- w * keras::k_cast(keras::k_greater_equal(w, 0), keras::k_floatx()) # if less than zero, make zero
  w / (keras::k_sum(w, 1) + 1e-16) # add a bit of noise to prevent 0/0 error
}

#' Constrain All Weights to Zero
#'
#' This weights constraint function forces all weights to zero.
#'
#' @param w A weights matrix.
#' @export
constraint_all_zeros <- function(w){

  w * keras::k_cast(0, keras::k_floatx())
}

#' Constrain All Weights to One
#'
#' This weights constraint function forces all weights to one.
#'
#' @param w A weights matrix.
#' @export
constraint_all_ones <- function(w){

  w * keras::k_cast(0, keras::k_floatx()) + 1
}

#' Constrain All Weights to Randomize
#'
#' This weights constraint function forces all weights to randomize each cycle.
#'
#' @param w A weights matrix.
#' @export
constraint_runif <- function(w){

  keras::k_random_uniform(dim(w))
}
