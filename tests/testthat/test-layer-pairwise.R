library(keras)
library(caress)
data(iris)
x <- as.matrix(iris[,1:4])
y <- to_categorical(as.numeric(iris[,5])-1)

# Test for 2D tensor input
k_clear_session()
input <- from_input(x)
middle <- input %>%
  layer_dense(12) %>%
  layer_reshape(c(3, 4))
out <- middle %>%
  layer_pairwise_rmse(name = "pairwise") %>%
  layer_flatten() %>%
  to_output(y)

model <- prepare(input, out)
build(model, x, y, epochs = 1)

test_that("the pairwise subtraction layer works as expected", {

  a <- get_layer_output(model, x, "pairwise_reshape1")[1,1,1,]
  b <- get_layer_output(model, x, "pairwise_square")[1,1,,]

  expect_equal(
    round((a[1] - a[2])^2, 3),
    round(b[1,2], 3)
  )

  a <- get_layer_output(model, x, "pairwise_reshape1")[1,2,1,]
  b <- get_layer_output(model, x, "pairwise_square")[1,2,,]

  expect_equal(
    round((a[1] - a[2])^2, 3),
    round(b[1,2], 3)
  )

  a <- get_layer_output(model, x, "pairwise_square")[1,1,,]
  b <- get_layer_output(model, x, "pairwise_sum")[1,1,]

  expect_equal(
    round(sum(a), 3),
    round(b, 3)
  )

  a <- get_layer_output(model, x, "pairwise_square")[1,1,,]
  b <- get_layer_output(model, x, "pairwise_square_root")[1,1,]

  expect_equal(
    round(sqrt(mean(a)), 3),
    round(b, 3)
  )
})
