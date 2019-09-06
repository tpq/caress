#' Split Model into Two Parallel Layers
#'
#' This function splits the input into two parallel layers.
#'  Each layer has the same architecture, but is influenced
#'  by different weights.
#'
#' @param object A \code{keras} model.
#' @param name A string. The prefix label for all layers.
#' @param ... Arguments passed to \code{keras::layer_dense}.
#' @return A list of two layers: Channel 1 and Channel 2.
#' @export
layer_to_stereo <- function(object, name = NULL, ...){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(object)
  }

  channel1 <- object %>% keras::layer_dense(..., name = paste0(name, "_channel", 1))
  channel2 <- object %>% keras::layer_dense(..., name = paste0(name, "_channel", 2))

  return(list("channel1" = channel1,
              "channel2" = channel2))
}

#' Split Model into Three Parallel Layers
#'
#' This function splits the input into three parallel layers.
#'  The first two layers have the same architecture, but are influenced
#'  by different weights. The third layer is the concatenation of
#'  the other two layers.
#'
#' @inheritParams layer_to_stereo
#' @return A list of three layers: Channel 1, Channel 2, and
#'  Channel 1 & Channel 2.
#' @export
layer_to_stereo_and_mono <- function(object, name = NULL, ...){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(object)
  }

  channel1 <- object %>% keras::layer_dense(..., name = paste0(name, "_channel", 1))
  channel2 <- object %>% keras::layer_dense(..., name = paste0(name, "_channel", 2))
  channel12 <- keras::layer_concatenate(list(channel1, channel2),
                                        name = paste0(name, "_channel", 12))

  return(list("channel1" = channel1, "channel2" = channel2,
              "channel12" = channel12))
}
