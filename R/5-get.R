#' Get Layer Output
#'
#' This function returns the output of a layer for any input data.
#'
#' @param model A \code{keras} model.
#' @param data A matrix or list of matrices. The input data.
#' @param layer The layer name or index.
#' @return An R array.
#' @export
get_layer_output <- function(model, data, layer){

  layer.i <- layer2index(model, layer)

  # create a Keras function to get i-th layer
  get_output <-
    keras::k_function(inputs = model$input,
                      outputs = model$layers[[layer.i]]$output)

  # extract output
  get_output(data)
}

#' Get Layer Weights
#'
#' This function returns the weights of a layer.
#'
#' @inheritParams get_layer_output
#' @return An R array.
#' @export
get_layer_weights <- function(model, layer){

  layer.i <- layer2index(model, layer)
  weights.obj <- model$layers[[layer.i]]$weights
  lapply(weights.obj, keras::k_eval)
}
