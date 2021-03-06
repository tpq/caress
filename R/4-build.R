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
  tryCatch(print(deepviz::plot_model(model)),
           error = function(e) warning("Model visualization failed."))
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
#' @param loss A list of losses, one for each \code{y_train}. If set to NULL,
#'  program will choose loss automatically based on \code{y_train}.
#'  Argument passed to \code{keras::compile}.
#' @param lr,loss_weights
#'  Arguments passed to \code{keras::compile}.
#' @param epochs,batch_size,validation_split
#'  Arguments passed to \code{keras::fit}.
#' @param early_stopping_monitor,early_stopping_patience
#'  Arguments passed to \code{keras::callback_early_stopping}.
#' @param reduce_lr_monitor,reduce_lr_patience
#'  Arguments passed to \code{keras::callback_reduce_lr_on_plateau}.
#' @return This function returns the history. The model
#'  is updated in situ.
#' @export
build <- function(model, x_train, y_train,
                  lr = 0.001, loss = NULL, loss_weights = NULL,
                  epochs = 30, batch_size = 128,
                  early_stopping_monitor = "val_loss",
                  early_stopping_patience = 5,
                  reduce_lr_monitor = "val_loss",
                  reduce_lr_patience = epochs,
                  validation_split = 0.2){

  if(is.null(loss)){
    loss <- to_loss(y_train)
  }
  metric <- to_metric(y_train)

  if(is.list(y_train)){ # if y is a list
    if(is.null(names(y_train))){ # and there are no names
      names(y_train) <- model$output_names # then name the list
    }
    names(loss) <- names(y_train) # pass names to loss
    names(metric) <- names(y_train) # pass names to metric
  }

  model %>%
    keras::compile(
      optimizer = keras::optimizer_rmsprop(lr = lr),
      loss = loss, metrics = metric,
      loss_weights = loss_weights
    )

  history <- model %>%
    keras::fit(
      x_train, y_train,
      epochs = epochs, batch_size = batch_size,
      callbacks = list(
        keras::callback_early_stopping(monitor = early_stopping_monitor, patience = early_stopping_patience),
        keras::callback_reduce_lr_on_plateau(monitor = reduce_lr_monitor, patience = reduce_lr_patience)
      ),
      validation_split = validation_split
    )

  return(history)
}
