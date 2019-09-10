#' Apply a DeepTRIAGE Layer
#'
#' This function applies a variant of the DeepTRIAGE attention
#'  mechanism to the incoming layer (see DOI:10.1101/533406).
#'  This implementation differs slightly from the publication
#'  in that all layers have the same activation function and
#'  the random embedding weights are optionally learnable.
#'
#' @inheritParams layer_pseudo_embed
#' @param result_dim An integer. The size of the final layer.
#' @param hidden_dim An integer. The size of the hidden layers.
#' @param hidden_activation A string. The activation for the hidden layers.
#' @param hidden_dropout A numeric. The dropout for the hidden layers.
#' @importFrom keras layer_dropout layer_flatten layer_activation_softmax
#' @export
layer_to_dense_DeepTRIAGE <- function(object, result_dim,
                                      embed_dim = result_dim*4, random_embedding = FALSE,
                                      hidden_dim = 32, hidden_activation = "tanh", hidden_dropout = .5,
                                      name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(object)
  }

  # Embed the data (N x D x M) [M << D]
  embedded_data <- object %>%
    layer_pseudo_embed(embed_dim, random_embedding = random_embedding, name = name)

  # Compress the embedded data (N x D x P) [P << M]
  compressed_data <- embedded_data %>%
    layer_dense(units = result_dim, activation = hidden_activation, name = paste0(name, "_3D_compressor")) %>%
    layer_dropout(rate = hidden_dropout, name = paste0(name, "_dropout_c"))

  # Calculate importance (N x D x 1)
  input_dim <- unlist(dim(object)[-1])
  importance_scores <- embedded_data %>%
    layer_dense(units = hidden_dim, activation = hidden_activation, name = paste0(name, "_hidden_i")) %>%
    layer_dropout(rate = hidden_dropout, name = paste0(name, "_dropout_i")) %>%
    layer_dense(units = 1, name = paste0(name, "_3D_importance_raw")) %>%
    # totally softmax the entire (N x D x 1) layer -> then, reshape it back to (N x D x 1)
    layer_flatten(name = paste0(name, "_importance_raw")) %>%
    layer_activation_softmax(name = paste0(name, "_importance_scores")) %>%
    layer_reshape(c(input_dim, 1), name = paste0(name, "_3D_importance_scores"))

  # Dot product: (N x D x P) %*% (N x D x 1) = (N x P x 1)
  dot_axis <- length(dim(importance_scores))-2
  outgoing <- layer_dot(list(compressed_data, importance_scores), axes = dot_axis,
                        name = paste0(name, "_3D_latent"))

  # Drop to (N x P)
  out_dim <- dim(outgoing)[c(-1, -length(dim(outgoing)))]
  outgoing %>%
    layer_reshape(out_dim, name = paste0(name, "_latent")) %>%
    # run through one last hidden layer
    layer_dense(units = hidden_dim, activation = hidden_activation, name = paste0(name, "_hidden_l")) %>%
    layer_dropout(rate = hidden_dropout, name = paste0(name, "_dropout_l"))
}
