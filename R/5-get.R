#' Get Layer Names
#'
#' This function returns the names of all layers in a model.
#'
#' @param model A keras model.
#' @return A character vector. The layer names.
#' @export
get_layer_names <- function(model){

  sapply(model$layers, function(x) x$name)
}

#' Get Incoming Layer Name
#'
#' This function returns the name of the incoming layer.
#'
#' @param object A keras layer.
#' @return A string. The layer name.
#' @export
get_incoming_layer_name <- function(object){

  unlist(strsplit(object$name, "/|:"))[1]
}

#' Find Index for Layer Name
#'
#' This function returns the numeric index that corresponds
#'  to a requested layer. If the request is a string, this function
#'  looks up the index. If the request is an integer, this function
#'  returns that integer.
#'
#' @param model A keras model.
#' @param layer A string or integer. The requested layer.
#' @return An integer. The requested layer's index.
#' @export
layer2index <- function(model, layer){

  if(class(layer) == "character"){
    layer.i <- which(layer == get_layer_names(model))
    if(length(layer.i) == 0) stop("Provided 'layer' name not found.")
  }else if(class(layer) == "numeric"){
    layer.i <- layer
  }else{
    stop("Provide 'layer' argument as character or numeric.")
  }

  return(layer.i)
}

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
