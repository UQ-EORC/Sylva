
test_that("a parent past the last cylinder is refused", {
  rows <- matrix(0, 2, 12)
  rows[, 6] <- 1; rows[, 7] <- 1; rows[, 8] <- 0.1
  rows[1, 9] <- -1; rows[2, 9] <- 5
  colnames(rows) <- QSM_COLUMNS
  expect_error(metrics(qsm(as.data.frame(rows))), "parent 5")
})
