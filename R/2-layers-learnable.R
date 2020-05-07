#' Initialize Learnable Layer
#'
#' This function creates a layer of learnable weights. It requires connection
#'  to an input layer, but does not use the input. Instead, the values of the
#'  layer are the manifestation of learnable weights.
#'
#' @param input The incoming layer.
#' @param array_dim The dimensions of the learnable kernel.
#' @param name A string. The prefix label for all layers.
#' @return A layer of learnable weights.
#' @examples
#' library(keras)
#' library(caress)
#' x <- as.matrix(iris[,1:4])
#' y <- to_categorical(as.numeric(iris[,5])-1)
#' k_clear_session()
#' input <- from_input(x)
#' weight <- input %>%
#'   layer_learnable_array(4) %>%
#'   layer_reshape(c(4,1))
#' target <- layer_kernel_dot(input, weight) %>%
#'   layer_flatten() %>%
#'   to_output(y)
#' m <- prepare(input, target)
#' build(m, x, y)
#' @export
layer_learnable_array <- function(input, array_dim, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(input)
  }

  input %>%
    layer_lambda(function(x) k_cast(matrix(1, nrow = 1, ncol = 1), k_floatx()),
                 name = paste0(name, "_dummy")) %>%
    layer_dense(prod(array_dim), name = paste0(name, "_learned_weights")) %>%
    layer_reshape(array_dim, name = paste0(name, "_shaped_weights"))
}

#' Perform Kernel Dot Product
#'
#' This function calls \code{k_dot} as a "lambda" layer.
#'
#' @param layer The incoming layer.
#' @param kernel A layer that represents a kernel.
#' @param name A string. The prefix label for all layers.
#' @return The result of \code{k_dot}.
#' @examples
#' library(keras)
#' library(caress)
#' x <- as.matrix(iris[,1:4])
#' y <- to_categorical(as.numeric(iris[,5])-1)
#' k_clear_session()
#' input <- from_input(x)
#' weight <- input %>%
#'   layer_learnable_array(4) %>%
#'   layer_reshape(c(4,1))
#' target <- layer_kernel_dot(input, weight) %>%
#'   layer_flatten() %>%
#'   to_output(y)
#' m <- prepare(input, target)
#' build(m, x, y)
#' @export
layer_kernel_dot <- function(layer, kernel, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(layer)
  }

  layer_lambda(object = list(layer, kernel), f = function(x) k_dot(x[[1]], x[[2]]),
               name = paste0(name, "_k_dot"))
}

#' Perform Kernel Convolution
#'
#' This function calls \code{k_conv2d} as a "lambda" layer.
#'
#' @param layer The incoming layer.
#' @param kernel A layer that represents a kernel.
#' @param name A string. The prefix label for all layers.
#' @return The result of \code{k_conv2d}.
#' @examples
#' library(keras)
#' library(caress)
#' x <- as.matrix(iris[,1:4])
#' y <- to_categorical(as.numeric(iris[,5])-1)
#' k_clear_session()
#' input <- from_input(x)
#' reshape <- input %>%
#'   layer_dense(4*20) %>%
#'   layer_reshape(c(4,20,1))
#' weight <- input %>%
#'   layer_learnable_array(c(1, 20, 4)) %>%
#'   layer_lambda(f = function(x) k_transpose(x))
#' target <- layer_kernel_conv2d(reshape, weight) %>%
#'   layer_flatten() %>%
#'   to_output(y)
#' m <- prepare(input, target)
#' build(m, x, y)
#' @export
layer_kernel_conv2d <- function(layer, kernel, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(layer)
  }

  layer_lambda(object = list(layer, kernel), f = function(x) k_conv2d(x[[1]], x[[2]]),
               name = paste0(name, "_k_conv2d"))
}
