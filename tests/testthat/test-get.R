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

test_that("getters work as expected", {

  expect_equal(
    get_layer_names(model),
    c("input", "middle", "output")
  )

  OUT <- get_layer_output(model, x, "middle")
  WEIGHT <- get_layer_weights(model, "middle")
  expect_equal(
    round(OUT, 2),
    round(sweep(x %*% WEIGHT[[1]], 2, WEIGHT[[2]], "+"), 2)
  )
})
