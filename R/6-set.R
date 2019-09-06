#' Set Layer Weights
#'
#' This function sets the weights of a layer.
#'
#' @param model A \code{keras} model.
#' @param layer The layer name or index.
#' @param weights A matrix or list of matrices. The weights.
#'  For example, the output of \code{\link{get_layer_weights}}.
#' @param freeze A logical. Toggles whether or not to freeze the weights.
#'  If frozen, the weights will not update during re-training.
#' @return Null. This function updates the model in place.
#' @export
set_layer_weights <- function(model, layer, weights, freeze){

  layer.i <- layer2index(model, layer)
  layer <- model$layers[[layer.i]]

  if(freeze){
    layer %>%
      keras::set_weights(weights) %>%
      keras::freeze_weights()
  }else{
    layer %>%
      keras::set_weights(weights)
  }
}

#' Copy Weights from Another Model
#'
#' This function uses a pre-trained "reference" model to automatically
#'  set the weights for all layers that have the same name.
#'
#' @inheritParams set_layer_weights
#' @param reference Another \code{keras} model. The model from which
#'  to copy the weights.
#' @return Null. This function updates the model in place.
#' @export
model_mirror <- function(model, reference, freeze){

  names_model <- caress::get_layer_names(model)
  names_reference <- caress::get_layer_names(reference)
  overlap <- intersect(names_model, names_reference)

  if(length(overlap) == 0){
    stop("The models have no layer names in common.")
  }else{
    print(paste0("Copying weights for layers: ", paste0(overlap, collapse = ", ")))
  }

  for(name in overlap){
    ref_weights <- get_layer_weights(model = reference, layer = name)
    set_layer_weights(model = model, layer = name, weights = ref_weights, freeze = freeze)
  }
}
