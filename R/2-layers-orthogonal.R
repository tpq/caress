#' Create Orthogonal Layer
#'
#' This function makes the "target" layer orthogonal to the "reference"
#'  layer so that it cannot predict the "reference". In other words, the
#'  "target" layer will equal the "reference" layer when the "target"
#'  is orthogonal to the "reference".
#'
#' @param target The incoming layer with predictive power.
#' @param reference The incoming layer you do not want to predict.
#' @param name A string. The name of the layers.
#' @return A layer that equals the "reference" when the "target" is
#'  orthogonal to the "reference".
#' @importFrom keras layer_dot layer_add
#' @export
layer_orthogonal_to <- function(target, reference, name = NULL){

  if(is.null(name)) name <- unlist(strsplit(target$name, "/|:"))[1]
  name_reference <- unlist(strsplit(reference$name, "/|:"))[1]
  name_perpendicular <- paste0(name, "_orthogonal_to_", name_reference)
  name_addback <- paste0(name, "_add_back_", name_reference)

  message("Alert: This layer equals ", name_reference, " when input is orthogonal to it.")
  zero_if_perpendicular <- layer_dot(list(target, reference), axes = 1, name = name_perpendicular)
  layer_add(list(zero_if_perpendicular, reference), name = name_addback)
}
