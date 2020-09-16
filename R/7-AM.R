#' Get the Loss Associated with a Layer
#'
#' @param model A keras model.
#' @param layer The name of the layer.
#' @param filter The filter to select. Skip with \code{filter = NA}.
#' @return A scalar loss.
#' @export
get_layer_loss <- function(model, layer, filter = NA){

  activity <- get_layer(model, name = layer)$output

  if(is.na(filter)){

    loss <- k_mean(activity)

  }else{

    dims <- dim(activity)
    slice <- switch(as.character(length(dims)),
                    "1" = activity[filter],
                    "2" = activity[,filter],
                    "3" = activity[,,filter],
                    "4" = activity[,,,filter],
                    "5" = activity[,,,,filter],
                    "6" = activity[,,,,,filter],
                    "7" = activity[,,,,,,filter],
                    "8" = activity[,,,,,,,filter],
                    "9" = activity[,,,,,,,,filter]
    )

    loss <- k_mean(slice)
  }

  return(loss)
}

#' Get the Gradient Associated with a Layer
#'
#' @param model A keras model.
#' @param loss The output from \code{\link{get_layer_loss}}.
#' @return A gradient.
#' @export
get_layer_gradient <- function(model, loss){

  grads <- k_gradients(loss, model$input)[[1]]
  # grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
}

#' Perform A Single Gradient Ascent
#'
#' @param model A keras model.
#' @param input The input to perturb.
#' @param loss The output from \code{\link{get_layer_loss}}.
#' @param gradient The output from \code{\link{get_layer_gradient}}.
#' @param lr The learning rate used to update input.
#' @return The perturbed input.
#' @examples
#' library(keras)
#' library(caress)
#' data(iris)
#' x <- as.matrix(iris[,1:4])
#' y <- to_categorical(as.numeric(iris[,5])-1)
#' input <- from_input(x, name = "input")
#' output <- input %>%
#'   layer_dense(units = 2, activation = "linear", name = "middle") %>%
#'   to_output(y, name = "output")
#' model <- prepare(input, output)
#' build(model, x, y, epochs = 10, batch_size = 8)
#' loss <- get_layer_loss(model, "middle")
#' gradient <- get_layer_gradient(model, loss)
#' x <- matrix(runif(4), 1, 4)
#' for(i in 1:20){
#'   x <- ascend(model, x, loss, gradient, lr = 1e5)
#' }
#' @export
ascend <- function(model, input, loss, gradient, lr = 1e5){

  runit <- k_function(list(model$input), list(loss, gradient))
  AM_out <- runit(input)
  loss_value <- AM_out[[1]]
  grads_value <- AM_out[[2]]
  print("Loss at this step is: ")
  print(loss_value)
  print("Gradient L1-norm at this step is: ")
  print(mean(abs(grads_value)))
  next_input <- input + (grads_value * lr)
}
