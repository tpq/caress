#' @import keras
#' @importFrom keras %>%
NULL

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

  if(!class(y) %in% c("array", "matrix")) y <- as.matrix(y)

  if(all(apply(y, 1, sum) == 1)){ # discrete outcomes -> softmax

    return("one-hot-encoded")

  }else if(all(y %in% c(0, 1)) & !all(y == 0)){ # multiple outcomes -> sigmoid

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

#' Get Metric for Output
#'
#' This function guesses the metric to use based on the
#'  type of outcome provided to \code{y}. It chooses from
#'  "accuracy" or "mean absolute percent error".
#'
#' @param y A matrix or vector. The output data.
#' @return A metric.
#' @export
to_metric <- function(y){

  if(identical(class(y), "list")){

    lapply(y, to_metric)

  }else{

    type <- type_of_y(y)

    if(type == "one-hot-encoded"){

      return(keras::metric_categorical_accuracy)

    }else if(type == "multi-label"){

      return(keras::metric_binary_accuracy)

    }else if(type == "continuous"){

      return(keras::metric_mean_absolute_percentage_error)

    }else{

      stop("Type not recognized!")
    }
  }
}
