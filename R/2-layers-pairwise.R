#' Create Pairwise Layer
#'
#' This function computes all pairwise residuals in the final dimension
#'  of a tensor. For example, if the tensor has the dimensions (2, 3, 4),
#'  this function will return a tensor with the dimensions (2, 3, 4, 4).
#'  The element (2, 3, i, j) would contain the value
#'  ((2, 3, i) - (2, 3, j))^2.
#'
#' @param object The incoming layer.
#' @param name A string. The prefix label for all layers.
#' @return A layer of the pairwise residuals.
#' @importFrom keras layer_reshape layer_lambda layer_subtract
#' @export
layer_pairwise_residual <- function(object, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(object)
  }

  obj_dims <- unlist(dim(object)[-1])

  out <-
    layer_subtract(list(
      layer_reshape(object,
                    c(obj_dims[-length(obj_dims)], 1, obj_dims[length(obj_dims)]),
                    name = paste0(name, "_reshape1")),
      layer_reshape(object,
                    c(obj_dims, 1),
                    name = paste0(name, "_reshape2"))),
      name = paste0(name, "_subtract")
    ) %>%
    layer_lambda(f = function(x) x^2,
                 name = paste0(name, "_square"))

  return(out)
}

#' Create Pairwise Layer
#'
#' This function computes all pairwise residuals in the final dimension
#'  of a tensor. For example, if the tensor has the dimensions (2, 3, 4),
#'  this function will return a tensor with the dimensions (2, 3, 1).
#'  The element (2, 3, 1) would contain the RMSE from the input
#'  SQRT(MEAN(((2, 3, i) - (2, 3, j))^2)).
#'
#' @param object The incoming layer.
#' @param name A string. The prefix label for all layers.
#' @return A layer of the root mean square error.
#' @importFrom keras layer_dense k_sqrt
#' @export
layer_pairwise_rmse <- function(object, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(object)
  }

  obj_dims <- unlist(dim(object)[-1])

  out <- object %>%
    layer_pairwise_residual(name = name) %>%
    layer_reshape(c(obj_dims[-length(obj_dims)], obj_dims[length(obj_dims)]^2),
                  name = paste0(name, "_flatten")) %>%
    layer_dense(1,
                kernel_constraint = constraint_all_ones,
                bias_constraint = constraint_all_zeros,
                name = paste0(name, "_sum")) %>%
    layer_lambda(f = function(x) k_sqrt(x/obj_dims[length(obj_dims)]^2),
                 name = paste0(name, "_square_root"))

  return(out)
}
