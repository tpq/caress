#' Get Type for Y
#'
#' This function guesses the "type" of outcome based on the
#'  object provided. It guesses "one-hot-encoded", "multi-label",
#'  or "continuous". Used by \code{\link{to_loss}} and
#'  \code{\link{to_output}}.
#'
#' @param y A matrix or vector. The output data.
#' @return A string.
#' @export
type_of_y <- function(y){

  y <- as.matrix(y)

  if(all(apply(y, 1, sum) == 1)){ # discrete outcomes -> softmax

    return("one-hot-encoded")

  }else if(all(y %in% c(0, 1))){ # multiple outcomes -> sigmoid

    return("multi-label")

  }else{ # continuous -> linear

    return("continuous")
  }
}

#' Get Loss for Output
#'
#' This function guesses the loss function based on the
#'  type of outcome provided to \code{y}. It chooses from
#'  binary cross-entropy, categorical cross-entropy,
#'  or mean squared error.
#'
#' @param y A matrix or vector. The output data.
#' @return A loss function.
#' @export
to_loss <- function(y){

  if(identical(class(y), "list")){

    lapply(y, to_loss)

  }else{

    y <- as.matrix(y)
    type <- type_of_y(y)

    if(type == "one-hot-encoded"){

      return(keras::loss_categorical_crossentropy)

    }else if(type == "multi-label"){

      return(keras::loss_binary_crossentropy)

    }else if(type == "continuous"){

      return(keras::loss_mean_squared_error)

    }else{

      stop("Type not recognized!")
    }
  }
}

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
