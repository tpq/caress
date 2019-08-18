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

  model <- keras::keras_model(input, output)
  print(deepviz::plot_model(model))
  print(summary(model))
  return(model)
}

#' Build a Keras Model
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
#' @return This function returns the history. The model
#'  is updated in situ.
#' @importFrom keras %>%
#' @export
build <- function(model, x_train, y_train,
                  lr = 0.001, epochs = 30, batch_size = 128,
                  validation_split = 0.2){

  if(class(y_train) == "list"){
    type <- lapply(y_train, type_of_y)
  }else{
    type <- type_of_y(y_train)
  }

  loss <- type2loss(type)

  model %>%
    keras::compile(
      loss = loss,
      optimizer = keras::optimizer_rmsprop(lr = lr),
      metrics = c('accuracy')
    )

  history <- model %>%
    keras::fit(
      x_train, y_train,
      epochs = epochs, batch_size = batch_size,
      validation_split = validation_split
    )

  return(history)
}
