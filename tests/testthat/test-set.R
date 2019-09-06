library(keras)
library(caress)
data(iris)

x <- as.matrix(iris[,1:4])
y <- to_categorical(as.numeric(iris[,5])-1)

input <- from_input(x, name = "input")
output <- input %>%
  layer_dense(units = 2, activation = "linear", name = "middle") %>%
  to_output(y, name = "output")

model <- prepare(input, output)
build(model, x, y, epochs = 10, batch_size = 8)

model2 <- prepare(input, output)
model_mirror(model2, reference = model, freeze = TRUE)

test_that("model_mirror correctly sets weights for each layer", {

  expect_equal(
    get_layer_weights(model, "input"),
    get_layer_weights(model2, "input")
  )
})

build(model2, x, y, epochs = 10, batch_size = 8)

test_that("model_mirror correctly freezes weights for each layer", {

  expect_equal(
    get_layer_weights(model, "input"),
    get_layer_weights(model2, "input")
  )
})
