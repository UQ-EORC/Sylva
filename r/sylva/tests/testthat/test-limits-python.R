# The memory budget against the Python package
# (fixtures written by tests/parity/export_r_limits.py).

expected <- load_expected("limits")

test_that("sizes read as in Python", {
  values <- c(0, 1, 999, 999.4, 999.6, 1000, 1049, 1050, 1e6, 12345678, 2.5e9, 7.77e12, 3.2e15, 0.4, -5)
  expect_equal(human_bytes(values), expected$human)
})

test_that("the budget is set, checked and put back as in Python", {
  set_memory_budget(2.5)
  on.exit(set_memory_budget(NULL))
  expect_equal(memory_budget(), expected$budget)
  err <- tryCatch(memory_check(1e6, 5000, "a 100 x 100 x 100 grid", "a larger voxel"), error = conditionMessage)
  expect_equal(err, expected$message)
  expect_silent(memory_check(1000, 8, "a small thing", "nothing"))
  set_memory_budget(NULL)
  a <- memory_available()
  expect_true(is.null(a) || a > 0)
})
