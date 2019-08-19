#' Split the Training and Test Set
#'
#' This function splits a list of inputs and a list of outputs
#'  by random sampling. All inputs and outputs get split the same.
#'
#' @param x The input data or a list of input data.
#' @param y The output data or a list of output data.
#' @param split An integer. The training-test set split as a percentage.
#' @param normalize A logical. Toggles whether to normalize each
#'  feature by subtracting the training set mean and dividing
#'  by the training set standard deviation.
#' @return A nested list with two slots "train" and "test", each of which
#'  contain another list with two more slots "x" and "y". These provide a
#'  list of the processed inputs and outputs, respectively.
#'  For example, access the training input with $train$x.
#'  You can feed these lists to \code{build}.
#' @export
sample_random <- function(x, y, split = 67, normalize = TRUE){

  # Always treat input as a list
  if(class(x) == "list"){
    message("Alert: Multiple input detected.")
  }else{
    x <- list(x)
  }

  # Always treat output as a list
  if(class(y) == "list"){
    message("Alert: Multiple output detected.")
  }else{
    y <- list(y)
  }

  # Coerce input as matrix
  for(i in 1:length(x)){
    x[[i]] <- as.matrix(x[[i]])
  }

  # Coerce output as matrix
  for(j in 1:length(y)){
    if(class(y[[j]]) == "character" | class(y[[j]]) == "factor"){
      message("Alert: One-hot encoding factor.")
      y[[j]] <- keras::to_categorical(as.numeric(factor(y[[j]]))-1)
    }else{
      y[[j]] <- as.matrix(y[[j]])
    }
  }

  if(!length(unique(sapply(c(x, y), nrow))) == 1){
    stop("All input and output must have the same number of rows!")
  }

  # Subset training data for all input and output
  train.index <- sample(1:nrow(x[[1]]), 67/100*nrow(x[[1]]))
  X.TRAIN <- lapply(x, function(dat) dat[train.index,])
  X.TEST <- lapply(x, function(dat) dat[-train.index,])
  Y.TRAIN <- lapply(y, function(dat) dat[train.index,])
  Y.TEST <- lapply(y, function(dat) dat[-train.index,])

  # Normalize per-feature via (x - mu) / sd
  if(normalize){
    for(i in 1:length(X.TRAIN)){
      m <- apply(X.TRAIN[[i]], 2, mean)
      s <- apply(X.TRAIN[[i]], 2, stats::sd)
      X.TRAIN[[i]] <- sweep(sweep(X.TRAIN[[i]], 2, m, "-"), 2, s, "/")
      X.TEST[[i]] <- sweep(sweep(X.TEST[[i]], 2, m, "-"), 2, s, "/")
    }
  }

  if(length(X.TRAIN) == 1){
    X.TRAIN <- X.TRAIN[[1]]
    X.TEST <- X.TEST[[1]]
  }

  if(length(Y.TRAIN) == 1){
    Y.TRAIN <- Y.TRAIN[[1]]
    Y.TEST <- Y.TEST[[1]]
  }

  list(
    "train" =
      list(
        "x" = X.TRAIN,
        "y" = Y.TRAIN
      ),
    "test" =
      list(
        "x" = X.TEST,
        "y" = Y.TEST
      )
  )
}
