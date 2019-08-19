library(keras)
library(caress)
data(iris)

x <- list(iris[,1:2], iris[,3:4])
y <- list(iris[,4], iris[,5])

set.seed(1)
sampled <- sample_random(x, y, split = 67)

set.seed(1)
sampled2 <- sample_random(iris[,1:4], iris[,5], split = 67)

test_that("sample_random function handles lists of input correctly", {

  expect_equal(
    cbind(sampled$train$x[[1]], sampled$train$x[[2]]),
    sampled2$train$x
  )

  expect_equal(
    sampled$test$y[[2]],
    sampled2$test$y
  )
})

set.seed(1)
sampled3 <- sample_random(iris[,1:4], iris[,5], split = 67, normalize = FALSE)

set.seed(1)
sampled4 <- iris[sample(1:nrow(iris), (67/100)*nrow(iris)),]

test_that("sample_random $x and $y training sets agree", {

  expect_equal(
    sampled3$train$x[,1],
    sampled4[,1]
  )

  expect_equal(
    apply(sampled3$train$y, 1, which.max),
    as.numeric(sampled4[,5])
  )
})
