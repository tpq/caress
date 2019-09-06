#' Prepare a Keras Model
#'
#' This wrapper joins the input and output layer(s) using
#'  \code{keras::keras_model}, then plots the model graph and
#'  prints the summary.
#'
#' @param input A layer or list of layers. The input layer(s).
#' @param output A layer or list of layers. The output layer(s).
#' @return A keras model.
#' @export
prepare <- function(input, output){

  # Convert lists of lists into a list
  input <- unlist(input, recursive = TRUE)
  output <- unlist(output, recursive = TRUE)

  model <- keras::keras_model(input, output)
  print(deepviz::plot_model(model))
  print(summary(model))
  return(model)
}

#' Train a Keras Model
#'
#' This wrapper automatically compiles and fits a model on
#'  the provided training data. Loss is selected automatically
#'  for each output using \code{type2loss}.
#'
#' @param model A keras model.
#' @param x_train,y_train A numeric or list of numerics.
#'  Arguments passed to \code{keras::fit}.
#' @param lr
#'  Arguments passed to \code{keras::compile}.
#' @param epochs,batch_size,validation_split
#'  Arguments passed to \code{keras::fit}.
#' @param patience_early_stopping Argument passed to
#'  \code{keras::callback_early_stopping}.
#' @return This function returns the history. The model
#'  is updated in situ.
#' @importFrom keras %>%
#' @export
build <- function(model, x_train, y_train,
                  lr = 0.001, epochs = 30, batch_size = 128,
                  patience_early_stopping = 5,
                  validation_split = 0.2){

  loss <- to_loss(y_train)
  metric <- to_metric(y_train)

  model %>%
    keras::compile(
      loss = loss,
      optimizer = keras::optimizer_rmsprop(lr = lr),
      metrics = metric
    )

  history <- model %>%
    keras::fit(
      x_train, y_train,
      epochs = epochs, batch_size = batch_size,
      callbacks = list(
        keras::callback_early_stopping(patience = patience_early_stopping)),
      validation_split = validation_split
    )

  return(history)
}
