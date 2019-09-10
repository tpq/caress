#' Create an Embedding Matrix
#'
#' This function creates an embedding matrix in the form of a layer.
#'  This layer can be multiplied with other layers as if it were itself
#'  a weights matrix. This is because all samples within the embedding layer
#'  will always have the same activation.
#'
#' @param object A \code{keras} model.
#' @param embed_dim An integer. The size of the final embedding matrix will
#'  equal the input dimension times the embedding dimension.
#' @param random_embedding A boolean. Toggles whether to freeze the embedding
#'  matrix with random values. Otherwise, the embedding matrix is trainable.
#' @param name A string. The prefix label for all layers.
#' @return A layer that acts like an embedding matrix.
#' @importFrom keras layer_dense layer_reshape layer_multiply
#' @export
layer_pseudo_embed <- function(object, embed_dim, random_embedding = FALSE, name = NULL){

  # Name layer based on incoming data
  if(is.null(name)){
    name <- get_incoming_layer_name(object)
  }

  # Given D input features and M embedding features
  input_dim <- unlist(dim(object)[-1])

  if(random_embedding){

    # Create a random embedding matrix
    rand_weights <- list(t(stats::runif(prod(input_dim)*embed_dim)))

    embedding_matrix <- object %>%
      # Ensure all samples have same embedding matrix by introducing a dummy node
      layer_dense(units = 1,
                  kernel_constraint = constraint_all_zeros,
                  bias_constraint = constraint_all_ones,
                  name = paste0(name, "_erase_tensor_data")) %>%
      # The weights emited from dummy node is the embedding matrix
      layer_dense(units = prod(input_dim)*embed_dim,
                  use_bias = FALSE, weights = rand_weights, trainable = FALSE,
                  name = paste0(name, "_compute_embedding_weights")) %>%
      # Shape embedding matrix as [D, M]
      layer_reshape(c(input_dim, embed_dim),
                    name = paste0(name, "_embedding_matrix"))

  }else{

    embedding_matrix <- object %>%
      # Ensure all samples have same embedding matrix by introducing a dummy node
      layer_dense(units = 1,
                  kernel_constraint = constraint_all_zeros,
                  bias_constraint = constraint_all_ones,
                  name = paste0(name, "_erase_tensor_data")) %>%
      # The weights emited from dummy node is the embedding matrix
      layer_dense(units = prod(input_dim)*embed_dim,
                  use_bias = FALSE,
                  name = paste0(name, "_compute_embedding_weights")) %>%
      # Shape embedding matrix as [D, M]
      layer_reshape(c(input_dim, embed_dim),
                    name = paste0(name, "_embedding_matrix"))
  }

  # Shape input matrix as [D, 1]
  input_matrix <- object %>%
    layer_reshape(c(input_dim, 1),
                  name = paste0(name, "_data_to_embed"))

  # Row-wise multiply (I hope?)
  layer_multiply(list(embedding_matrix, input_matrix),
                 name = paste0(name, "_embedded_data"))
}
