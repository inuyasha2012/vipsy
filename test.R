library(stringr)
library(mirt)
test_dim_1_small <- function(base_path, try_count, method) {
  res_a <- 1:try_count
  res_b <- 1:try_count
  for (i in 1:try_count) {
    dt <- read.table(str_c(base_path, i - 1, ".txt"), quote = "\"", comment.char = "")
    a <- read.table(str_c(base_path, "a_", i - 1, ".txt"), quote = "\"", comment.char = "")
    b <- read.table(str_c(base_path, "b_", i - 1, ".txt"), quote = "\"", comment.char = "")
    x_feature <- 1
    item_size <- length(dt[1,])
    mod <- mirt(dt, x_feature, '2PL', method=method)
    param <- coef(mod, simplify = TRUE)$items
    a_ <- param[, 1:x_feature]
    b_ <- param[, x_feature + 1]
    a_rmse <- sum(((a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    b_rmse <- sum(((b_ - b)^2)^0.5) / item_size
    print(a_rmse)
    print(b_rmse)
    res_a[i] <- a_rmse
    res_b[i] <- b_rmse
  }
  return(list(a=res_a, b=res_b))
}
test_3pl <- function(base_path, try_count, method) {
  res_a <- 1:try_count
  res_b <- 1:try_count
  res_c <- 1:try_count
  for (i in 1:try_count) {
    dt <- read.table(str_c(base_path, i - 1, ".txt"), quote = "\"", comment.char = "")
    a <- read.table(str_c(base_path, "a_", i - 1, ".txt"), quote = "\"", comment.char = "")
    b <- read.table(str_c(base_path, "b_", i - 1, ".txt"), quote = "\"", comment.char = "")
    c <- read.table(str_c(base_path, "c_", i - 1, ".txt"), quote = "\"", comment.char = "")
    x_feature <- 1
    item_size <- length(dt[1,])
    mod <- mirt(dt, x_feature, '2PL', method=method)
    param <- coef(mod, simplify = TRUE)$items
    a_ <- param[, 1:x_feature]
    b_ <- param[, x_feature + 1]
    c_ <- param[, x_feature + 2]
    a_rmse <- sum(((a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    b_rmse <- sum(((b_ - b)^2)^0.5) / item_size
    c_rmse <- sum(((c_ - c)^2)^0.5) / item_size
    print(a_rmse)
    print(b_rmse)
    print(c_rmse)
    res_a[i] <- a_rmse
    res_b[i] <- b_rmse
    res_c[i] <- c_rmse
  }
  return(list(a=res_a, b=res_b, c=res_c))
}
#res <- test_dim_1_small("irt_2pl_100_", 10, 'EM')
#res <- test_dim_1_small("irt_2pl_100_", 10, 'MHRM')
#res <- test_dim_1_small("irt_2pl_200_", 10, 'EM')
#res <- test_dim_1_small("irt_2pl_200_", 10, 'MHRM')
#res <- test_dim_1_small("irt_2pl_500_", 10, 'EM')
#res <- test_dim_1_small("irt_2pl_500_", 10, 'MHRM')
res <- test_3pl("irt_3pl_500_", 10, 'MHRM')
print(mean(res$a))
print(sd(res$a))
print(mean(res$b))
print(sd(res$b))
print(mean(res$c))
print(sd(res$c))
