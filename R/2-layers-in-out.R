#' Create Input Layer
#'
#' This function automatically creates an input layer for
#'  a provided matrix object.
#'
#' @param x A matrix or vector. The input data.
#' @param name A string. The name of the layer.
#' @param append_name A boolean. Toggles whether to give each
#'  name a unique appendix. Used for backend only.
#' @return A layer object.
#' @importFrom keras layer_input
#' @export
from_input <- function(x, name = NULL, append_name = FALSE){

  if(identical(class(x), "list")){

    model_index <<- 1
    lapply(x, from_input, append_name = TRUE)

  }else{

    if(append_name){

      if(is.null(name)) name <- "input"
      name <- paste0(name, "_", model_index)
      model_index <<- model_index + 1
    }

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
#' @param y A matrix or vector. The output data.
#' @param name A string. The name of the layer.
#' @param append_name A boolean. Toggles whether to give each
#'  name a unique appendix. Used for backend only.
#' @return A layer object.
#' @importFrom keras layer_dense
#' @export
to_output <- function(model, y, name = NULL, append_name = FALSE){

  if(identical(class(y), "list")){

    model_link <<- model
    model_index <<- 1
    lapply(y, function(output) to_output(model_link, output, name = name, append_name = TRUE))

  }else{

    if(append_name){

      if(is.null(name)) name <- "output"
      name <- paste0(name, "_", model_index)
      model_index <<- model_index + 1
    }

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
