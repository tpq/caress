#' Make Categorical Mask
#'
#' This function one-hot encodes an integer vector as a matrix,
#'  except that it masks certain outcomes. For example, consider
#'  the three outcomes {0, 1, 2} where 2 is masked. The value 0
#'  is coded as {1, 0, 0}. The value 1 is coded as {0, 1, 0}.
#'  The value 2 is coded as {.33, .33, .33}. This blinds the
#'  classifier to the value 2.
#'
#' @param y The outcome. An integer vector.
#' @param mask The values to mask. If missing, all values are masked.
#' @return A matrix like \code{keras::to_categorical}.
#' @export
to_categorical_mask <- function(y, mask){

  if(!0 %in% y) warning("No zero provided. Are you sure about this?")

  if(missing(mask)){
    message("Alert: No mask provided. Masking all values.")
    mask <- unique(y)
  }

  y_out <- keras::to_categorical(y)
  row_mask <- y %in% mask
  y_out[row_mask,] <- 1/ncol(y_out)
  return(y_out)
}
