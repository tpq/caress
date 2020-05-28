library(keras)
library(caress)
data(iris)

x <- as.matrix(iris[,1:4])
y <- to_categorical(as.numeric(iris[,5])-1)

k_clear_session()
use_session_with_seed(1)

input <- from_input(x, name = "input")
output <- input %>%
  layer_dense(units = 2, activation = "linear", name = "first",
              kernel_constraint = constraint_cols_to_unit_sum) %>%
  layer_dense(units = 2, activation = "linear", name = "second",
              kernel_constraint = constraint_rows_to_unit_sum) %>%
  to_output(y, name = "output")

model <- prepare(input, output)
build(model, x, y, epochs = 10, batch_size = 8)

test_that("constraints work as expected", {

  W <- get_layer_weights(model, "first")[[1]]
  expect_equal(
    round(colSums(W), 0),
    c(1, 1)
  )

  W <- get_layer_weights(model, "second")[[1]]
  expect_equal(
    round(rowSums(W), 0),
    c(1, 1)
  )
})
