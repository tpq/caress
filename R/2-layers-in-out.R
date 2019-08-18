#' Create Input Layer
#'
#' This function automatically creates an input layer for
#'  a provided matrix object.
#'
#' @param x A matrix. The input data.
#' @param name A string. The name of the layer.
#' @return A layer object.
#' @importFrom keras layer_input
#' @export
from_input <- function(x, name = NULL){

  layer_input(shape = dim(x)[-1], name = name)
}

#' Create Output Layer
#'
#' This function automatically creates an output layer for
#'  a provided matrix object.
#'
#' @param model A \code{keras} model.
#' @param y A matrix or vector. The output data.
#' @param name A string. The name of the layer.
#' @return A layer object.
#' @importFrom keras layer_dense
#' @export
to_output <- function(model, y, name = NULL){

  if(!is.null(dim(y))){ # if y is an array

    if(length(dim(y)) == 2){ # if y is a matrix

      if(ncol(y) == 1) stop("Provide single column as vector.")

      if(all(rowSums(y) == 1)){ # discrete outcomes

        if(ncol(y) == 2){

          message("Alert: Preparing model for binary classification.")
          return(model %>% layer_dense(units = 2, activation = 'softmax', name = name))

        }else{

          message("Alert: Preparing model for multi-class classification.")
          return(model %>% layer_dense(units = ncol(y), activation = 'softmax', name = name))
        }

      }else if(all(y %in% c(0, 1))){

        message("Alert: Preparing model for multi-label classification.")
        return(model %>% layer_dense(units = ncol(y), activation = 'sigmoid', name = name))

      }else{

        message("Alert: Preparing model for multi-variate regression.")
        return(model %>% layer_dense(units = ncol(y), activation = 'linear', name = name))
      }

    }else{

      message("Alert: Preparing model for multi-dimensional output.")
      return(model %>% layer_dense(units = dim(y[-1]), activation = 'linear', name = name))
    }

  }else{ # if y is a vector

    if(all(y %in% c(0, 1))){

      message("Alert: Preparing model for binary classification.")
      return(model %>% layer_dense(units = 1, activation = 'sigmoid', name = name))

    }else{

      message("Alert: Preparing model for uni-variate regression.")
      return(model %>% layer_dense(units = 1, activation = 'linear', name = name))
    }
  }
}
