
<!-- README.md is generated from README.Rmd. Please edit that file -->
Quick start
-----------

Welcome to the `caress` GitHub page!

Neural networks have lots of applications. A lot of people want to learn how to use them. The keras package makes it easy to design neural network architectures. The caress package makes it even easier.

``` r
library(devtools)
devtools::install_github("tpq/caress")
```

This package includes some helper functions that automate bread-and-butter network building. For example, they one-hot encode factors, normalize the feature input, set the input size, set the output size, and choose the correct loss function. I also wanted to make the functional API easier to use by going `from_input` on `to_output`.

``` r
library(keras)
library(caress)
data(iris)
data <- sample_random(x = iris[,1:4], y = iris[,5], split = 80, normalize = TRUE)
#> Alert: One-hot encoding factor.
x_train <- data$train$x
y_train <- data$train$y
x_test <- data$test$x
y_test <- data$test$y

input <- from_input(x_train)
output <- input %>%
  layer_dense(units = 2, activation = "tanh") %>%
  to_output(y_train)
#> Alert: Preparing model for binary or multi-class classification.

model <- prepare(input, output)
#> Registered S3 methods overwritten by 'ggplot2':
#>   method         from 
#>   [.quosures     rlang
#>   c.quosures     rlang
#>   print.quosures rlang
#> ___________________________________________________________________________
#> Layer (type)                     Output Shape                  Param #     
#> ===========================================================================
#> input_1 (InputLayer)             (None, 4)                     0           
#> ___________________________________________________________________________
#> dense (Dense)                    (None, 2)                     10          
#> ___________________________________________________________________________
#> dense_1 (Dense)                  (None, 3)                     9           
#> ===========================================================================
#> Total params: 19
#> Trainable params: 19
#> Non-trainable params: 0
#> ___________________________________________________________________________
#> NULL
```

Now, we can compile and fit the model with a single function call.

``` r
history <- build(model, x_train, y_train, epochs = 100, batch_size = 8)
evaluate(model, x_test, y_test)
```

See the vignette for more examples.
