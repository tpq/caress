library(keras)
library(caress)
data(iris)
x <- as.matrix(iris[,1:4])
y <- to_categorical(as.numeric(iris[,5])-1)
embed_dim <- 8

# Test for 2D tensor input
k_clear_session()
input <- from_input(x)
embed <- input %>% layer_pseudo_embed(embed_dim, random_embedding = TRUE, name = "e") %>%
  layer_flatten() %>%
  to_output(y)
model <- prepare(input, embed)

test_that("embedding matrix layer has correct dims and does not train", {

  build(model, x, y, epochs = 10, batch_size = 8)
  a <- get_layer_output(model, x, "e_embedding_matrix")

  expect_equal(
    a[1,,],
    a[2,,]
  )

  # The random embeddings initialize when layer_pseudo_embed is called!
  build(model, x, y, epochs = 10, batch_size = 8)
  b <- get_layer_output(model, x, "e_embedding_matrix")

  expect_equal(
    a,
    b
  )

  expect_equal(
    dim(a),
    c(dim(x), embed_dim)
  )
})

# Test for 3D tensor input
k_clear_session()
A <- array(1:(150*3*5), c(150, 3, 5))
input <- from_input(A)
embed <- input %>% layer_pseudo_embed(embed_dim, random_embedding = TRUE, name = "e") %>%
  layer_flatten() %>%
  to_output(y)
model <- prepare(input, embed)

test_that("embedding matrix layer has correct dims and does not train", {

  build(model, A, y, epochs = 10, batch_size = 8)
  a <- get_layer_output(model, A, "e_embedding_matrix")

  expect_equal(
    a[1,,,],
    a[2,,,]
  )

  expect_equal(
    dim(a),
    c(dim(A), embed_dim)
  )
})
