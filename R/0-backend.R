#' Get Type for Y
#'
#' This function guesses the "type" of outcome based on the
#'  object provided. It guesses "binary", "multiclass",
#'  "multilabel", "multivariate", or "univariate". Used by
#'  \code{type2loss} to choose a loss function.
#'
#' @param y The outcome. A vector or matrix.
#' @return A string.
#' @export
type_of_y <- function(y){

  if(!is.null(dim(y))){ # if y is an array

    if(length(dim(y)) == 2){ # if y is a matrix

      if(ncol(y) == 1) stop("Provided matrix cannot have one column.")

      if(all(rowSums(y) == 1)){ # discrete outcomes

        if(ncol(y) == 2){

          return("binary")

        }else{

          return("multiclass")
        }

      }else if(all(y %in% c(0, 1))){

        return("multilabel")

      }else{

        return("multivariate")
      }

    }else{

      return("multivariate")
    }

  }else{ # if y is a vector

    if(all(y %in% c(0, 1))){

      return("binary")

    }else{

      return("univariate")
    }
  }
}

#' Get Loss for Type
#'
#' This function guesses the loss function based on the
#'  type of outcome returned from \code{type_of_y}. It
#'  chooses from cross-entropy or mean squared error.
#'
#' @param type Choose from "binary", "multiclass",
#'  "multilabel", "multivariate", or "univariate".
#' @return A loss function.
#' @export
type2loss <- function(type){

  if(class(type) == "list"){

    lapply(type, type2loss)

  }else{

    if(type == "binary"){

      return(keras::loss_binary_crossentropy)

    }else if(type == "multiclass"){

      return(keras::loss_categorical_crossentropy)

    }else if(type == "multilabel"){

      return(keras::loss_binary_crossentropy)

    }else if(type == "multivariate"){

      return(keras::loss_mean_squared_error)

    }else if(type == "univariate"){

      return(keras::loss_mean_squared_error)

    }else{

      stop("Type not recognized!")
    }
  }
}
