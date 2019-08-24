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
