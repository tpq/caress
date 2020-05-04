## caress 0.1.1
---------------------
* Add `tryCatch` for `deepviz` call because of problems with lambda layer

## caress 0.1.0
---------------------
* Update constraints
    * Fix 0 / 0 NaN problem with the unit sum constraints

## caress 0.0.9
---------------------
* Update constraints
    * Fix 0 / 0 NaN problem with the unit sum constraints

## caress 0.0.8
---------------------
* Update stereo methods
    * Rename `layer_to_dense_stereo_and_mono` to `layer_to_dense_stereo_and_add`
    * Add `layer_to_dense_stereo_and_cat` to concatenate layers
    * Add `layer_to_dense_stereo_and_diff` to subtract layers
* Improve front-end
    * Fix `to_output` to correctly handle arrays
    * `sample_random` now uses min-max instead of mean-sd normalization
    * New `model_decode` wrapper helps decode latent space
    * Now pass `loss_weights` to `compile()`

## caress 0.0.7
---------------------
* Update stereo methods
    * Rename `layer_to_stereo` to `layer_to_dense_stereo`, etc.
    * Use `layer_add` instead of `layer_concatenate` to make the mono channel
    * Add explicit units and activation arguments
* Improve front-end
    * New `layer_to_dense_DeepTRIAGE` adds a DeepTRIAGE layer
    * `from_input` and `to_output` now correctly accept arrays
    * `layer_pseudo_embed` should now accept arrays
    * Now provide metrics as a named list

## caress 0.0.6
---------------------
* Add more helper functions
    * `set_model_weights` sets weights for a layer
    * `model_mirror` sets many weights based on a pre-trained "reference" model
    * `constraint_runif` uniformly randomizes weights each cycle
* Add new layer functions
    * `layer_to_stereo` splits a model into two parallel layers
    * `layer_to_stereo_and_mono` splits a model into three parallel layers
    * `layer_pseudo_embedding` creates an embedding matrix layer
* Improve back-end
    * Add `callback_early_stopping` callback to `build` by default
    * If all output is 0, treat as a regression model
    * `build` now chooses a suitable metric

## caress 0.0.5
---------------------
* Add more helper functions
    * `layer_orthogonal_to` equals output when orthogonal to output
* Improve front-end
    * Fix `get_layer_output` to handle lists of input
    * `from_input` and `to_output` now accept a vector of names
    * `from_input` and `to_ouput` now return a named list
    * Fix `<<-` WARNINGS

## caress 0.0.4
---------------------
* Improve front-end
    * `from_input` now supports lists of inputs
    * `to_output` now supports lists of outputs
    * `prepare` now supports lists of lists
* Improve back-end
    * Simplify `type_of_y` function to handle 2+D tensors
    * Replace `type2loss` with `to_loss` function
    * Have `to_output` call `type_of_y`

## caress 0.0.3
---------------------
* Add more helper functions
    * `sample_random` bulk prepares a list of training and test sets
    * `constraint_rows_to_unit_sum` constrains row sums of weights to 1
    * `constraint_cols_to_unit_sum` constrains column sums of weights to 1
    * `constraint_all_zeros` constrains all weights to 0
    * `constraint_all_ones` constrains all weights to 1

## caress 0.0.2
---------------------
* Add more helper functions
    * `get_layer_names` returns all layer names
    * `get_layer_output` returns layer output for a given input
    * `get_layer_weights` returns layer weights

## caress 0.0.1
---------------------
* Add some helper functions
    * `from_input` creates an input layer based on a matrix
    * `to_output` creates an output layer based on a matrix or vector
    * `prepare` joins input and output and displays model
    * `build` automatically compiles and fits model
