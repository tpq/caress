library(keras)
library(caress)
data(iris)
x <- as.matrix(iris[,1:4])
y <- to_categorical(as.numeric(iris[,5])-1)
embed_dim <- 8

input <- from_input(x)
embed <- input %>% layer_pseudo_embed(embed_dim, random_embedding = TRUE, name = "e") %>%
  layer_flatten() %>%
  to_output(y)
model <- prepare(input, embed)

test_that("embedding matrix layer has correct dims and does not train", {

  build(model, x, y, epochs = 10, batch_size = 8)
  a <- get_layer_output(model, x, "e_embedding_matrix")

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
