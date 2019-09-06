#' Create Input Layer
#'
#' This function automatically creates an input layer for
#'  a provided matrix object.
#'
#' @param x A matrix or list of matrices. The input data.
#' @param name A string or character vector. The name(s) of the layer.
#' @return A layer object (or list of layer objects).
#' @importFrom keras layer_input
#' @export
from_input <- function(x, name = NULL){

  # Name layer based as input
  if(is.null(name)){
    name <- "input"
  }

  if(identical(class(x), "list")){

    out <- vector("list", length(x))
    for(i in 1:length(out)){

      name_i <- name
      if(length(name) > 1) name_i <- name[i]
      out[[i]] <- from_input(x = x[[i]], name = name_i)
    }

    layer_names <- lapply(out, function(layer) layer$name)
    names(out) <- lapply(layer_names, function(name) unlist(strsplit(name, "/|:"))[1])
    return(out)

  }else{

    x <- as.matrix(x)
    layer_input(shape = dim(x)[-1], name = name)
  }
}

#' Create Output Layer
#'
#' This function automatically creates an output layer for
#'  a provided matrix object.
#'
#' @param object A \code{keras} model.
#' @param y A matrix or list of matrices. The output data.
#' @param name A string or character vector. The name(s) of the layer.
#' @param ... Arguments passed to \code{keras::layer_dense}.
#' @return A layer object (or list of layer objects).
#' @importFrom keras layer_dense
#' @export
to_output <- function(object, y, name = NULL, ...){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- paste0(get_incoming_layer_name(object), "_to_output")
  }

  if(identical(class(y), "list")){

    out <- vector("list", length(y))
    for(i in 1:length(out)){

      name_i <- paste0(name, "_", i)
      if(length(name) > 1) name_i <- name[i]
      out[[i]] <- to_output(object = object, y = y[[i]], name = name_i)
    }

    layer_names <- lapply(out, function(layer) layer$name)
    names(out) <- lapply(layer_names, function(name) unlist(strsplit(name, "/|:"))[1])
    return(out)

  }else{

    y <- as.matrix(y)
    type <- type_of_y(y)

    if(type == "one-hot-encoded"){

      message("Alert: Preparing model for binary or multi-class classification.")
      return(object %>% layer_dense(units = dim(y)[-1], activation = 'softmax', name = name, ...))

    }else if(type == "multi-label"){

      message("Alert: Preparing model for multi-label classification.")
      return(object %>% layer_dense(units = dim(y)[-1], activation = 'sigmoid', name = name, ...))

    }else if(type == "continuous"){

      message("Alert: Preparing model for univariate or multivariate regression.")
      return(object %>% layer_dense(units = dim(y)[-1], activation = 'linear', name = name, ...))

    }else{

      stop("Type not recognized!")
    }
  }
}
