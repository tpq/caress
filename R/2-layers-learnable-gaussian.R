#' Initialize Learnable Layer
#'
#' This function creates a layer of an arbitrary size, such that the
#'  activity of its nodes are spatially organized to follow a Gaussian
#'  distribution (based on a learnable mu and sigma).
#'
#' This function is probably not useful in practice, but is included
#'  as a pre-cursor to other learnable Gaussian layers.
#'
#' @param input The incoming layer.
#' @param size The total number of nodes in the Gaussian kernel.
#' @param name A string. The prefix label for all layers.
#' @return A layer of learnable weights.
#' @examples
#' library(keras)
#' library(caress)
#' x <- as.matrix(iris[,1:4])
#' y <- to_categorical(as.numeric(iris[,5])-1)
#' k_clear_session()
#' input <- from_input(x)
#' input2dense <- input %>%
#'   layer_dense(1)
#' weight <- input %>%
#'   layer_learnable_gaussian(100)
#' target <- layer_lambda(list(input2dense, weight), function(x) x[[1]]+x[[2]]) %>%
#'   layer_flatten() %>%
#'   to_output(y)
#' m <- prepare(input, target)
#' build(m, x, y)
#' plot(as.numeric(get_layer_output(m, x, "input_gaussian")))
#' @export
layer_learnable_gaussian <- function(input, size, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(input)
  }

  input %>%
    layer_learnable_array(2, name = name) %>%
    layer_lambda(function(x){
      range <- seq(-3, 3, length.out = size)
      range_cast <- k_cast(t(range), k_floatx())
      mu = x[1,1]
      sigma = x[1,2]
      1 / ( sigma * sqrt(2*pi) ) * exp(-.5 * ( (range_cast-mu)/sigma )^2 )
    }, name = paste0(name, "_gaussian"))
}

#' Initialize Learnable Layer
#'
#' This function creates a layer as a matrix, such that the
#'  activity of one row is spatially organized to follow a Gaussian
#'  distribution (based on a learnable mu and sigma).
#'
#' Example output:
#' [1,] 1.000000 0.0000000 0.00000000
#' [2,] 0.072769 0.2522599 0.00235219
#' [3,] 0.000000 0.0000000 0.00000000
#' [4,] 0.000000 0.0000000 0.00000000
#'
#' @param input The incoming layer.
#' @param kernel_size Exactly 2 integers. The dimensionality
#'  of the learned filter. The second number refers to the total
#'  number of nodes in the Gaussian kernel.
#' @param receptor_row The row in the filter kernel that should
#'  follow a Gaussian distribution.
#' @param target_row The row in the filter kernel that should
#'  have 1 in the first (or last) column.
#' @param target_mirror Toggles whether the \code{target_row}
#'  should have 1 in the first or last column. When \code{target_mirror = FALSE},
#'  the first column is used.
#' @param name A string. The prefix label for all layers.
#' @return A layer of learnable weights.
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
#' filter <- input %>%
#'   layer_learnable_gaussian_conv2d_pair(c(4, 20), target_row = 1, receptor_row = 2)
#' target <- layer_kernel_conv2d(reshape, filter) %>%
#'   layer_flatten() %>%
#'   to_output(y)
#' m <- prepare(input, target)
#' build(m, x, y)
#' plot(as.numeric(get_layer_output(m, x, "input_gaussian_conv2d_pair")[2,,1,1]))
#' @export
layer_learnable_gaussian_conv2d_pair <- function(input, kernel_size, receptor_row = 1,
                                                 target_row = receptor_row, target_mirror = FALSE,
                                                 name = NULL){

  if(!length(kernel_size) == 2){
    stop("Please provide a 'kernel_size' of 2.")
  }

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(input)
  }

  input %>%
    layer_learnable_array(2, name = name) %>%
    layer_lambda(function(x){

      mu = x[1,1]
      sigma = x[1,2]

      # Define receptive field in the i-th row of the filter
      filter <- array(0, c(kernel_size, 1, 1))
      filter[receptor_row,,1,1] <- seq(-3, 3, length.out = kernel_size[2])
      filter_cast <- k_cast(filter, k_floatx())

      # Transform receptive field based on learned mu and sigma...
      filter_as_pdf <-       1 / ( sigma * sqrt(2*pi) ) * exp(-.5 * ( (filter_cast-mu)/sigma )^2 )

      # Define MASK for the i-th row of the filter -- set non-receptor cells equal to zero
      mask <- array(0, c(kernel_size, 1, 1))
      mask[receptor_row,,1,1] <- 1
      mask_cast <- k_cast(mask, k_floatx())

      # Define target pair in the j-th row of the filter -- set target cell equal to one
      target_col <- ifelse(target_mirror, kernel_size[2], 1)
      target <- array(0, c(kernel_size, 1, 1))
      target[target_row,target_col,1,1] <- 1
      target_cast <- k_cast(target, k_floatx())

      (filter_as_pdf * mask_cast) + target_cast
    }, name = paste0(name, "_gaussian_conv2d_pair"))
}
