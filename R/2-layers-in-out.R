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

  if(identical(class(x), "list")){

    out <- vector("list", length(x))
    for(i in 1:length(out)){

      name_i <- name
      if(is.null(name)) name_i <- "input"
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
#' @param model A \code{keras} model.
#' @param y A matrix or list of matrices. The output data.
#' @param name A string or character vector. The name(s) of the layer.
#' @return A layer object (or list of layer objects).
#' @importFrom keras layer_dense
#' @export
to_output <- function(model, y, name = NULL){

  if(identical(class(y), "list")){

    out <- vector("list", length(y))
    for(i in 1:length(out)){

      name_i <- name
      if(is.null(name)) name_i <- "output"
      if(length(name) > 1) name_i <- name[i]
      out[[i]] <- to_output(model = model, y = y[[i]], name = name_i)
    }

    layer_names <- lapply(out, function(layer) layer$name)
    names(out) <- lapply(layer_names, function(name) unlist(strsplit(name, "/|:"))[1])
    return(out)

  }else{

    y <- as.matrix(y)
    type <- type_of_y(y)

    if(type == "one-hot-encoded"){

      message("Alert: Preparing model for binary or multi-class classification.")
      return(model %>% layer_dense(units = dim(y)[-1], activation = 'softmax', name = name))

    }else if(type == "multi-label"){

      message("Alert: Preparing model for multi-label classification.")
      return(model %>% layer_dense(units = dim(y)[-1], activation = 'sigmoid', name = name))

    }else if(type == "continuous"){

      message("Alert: Preparing model for univariate or multivariate regression.")
      return(model %>% layer_dense(units = dim(y)[-1], activation = 'linear', name = name))

    }else{

      stop("Type not recognized!")
    }
  }
}
