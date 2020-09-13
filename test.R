library(stringr)
library(mirt)
test_2pl <- function(base_path, try_count, method, technical=list()) {
  res_a <- 1:try_count
  res_b <- 1:try_count
  for (i in 1:try_count) {
    dt <- read.table(str_c(base_path, i - 1, ".txt"), quote = "\"", comment.char = "")
    a <- read.table(str_c(base_path, "a_", i - 1, ".txt"), quote = "\"", comment.char = "")
    b <- read.table(str_c(base_path, "b_", i - 1, ".txt"), quote = "\"", comment.char = "")
    x_feature <- length(a[1,])
    item_size <- length(dt[1,])
    mod <- mirt(dt, x_feature, '2PL', method=method, technical = technical )
    param <- coef(mod, simplify = TRUE)$items
    a_ <- param[, 1:x_feature]
    b_ <- param[, x_feature + 1]
    a_rmse1 <- sum(((-a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    a_rmse2 <- sum(((a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    a_rmse <- min(a_rmse1, a_rmse2)
    b_rmse <- sum(((b_ - b)^2)^0.5) / item_size
    print(str_c('a_rmse1:', a_rmse1))
    print(str_c('a_rmse2:', a_rmse2))
    print(str_c('b_rmse:', b_rmse))
    res_a[i] <- a_rmse
    res_b[i] <- b_rmse
  }
  return(list(a=res_a, b=res_b))
}
test_3pl <- function(base_path, try_count, method, GenRandomPars=FALSE, technical=list()) {
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
    mod <- mirt(dt, x_feature, '3PL', method=method, GenRandomPars=GenRandomPars, technical=technical)
    param <- coef(mod, simplify = TRUE)$items
    a_ <- param[, 1:x_feature]
    b_ <- param[, x_feature + 1]
    c_ <- param[, x_feature + 2]
    a_rmse1 <- sum(((-a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    a_rmse2 <- sum(((a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
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
test_4pl <- function(base_path, try_count, method, technical=list()) {
  res_a <- 1:try_count
  res_b <- 1:try_count
  res_c <- 1:try_count
  res_d <- 1:try_count
  for (i in 1:try_count) {
    dt <- read.table(str_c(base_path, i - 1, ".txt"), quote = "\"", comment.char = "")
    a <- read.table(str_c(base_path, "a_", i - 1, ".txt"), quote = "\"", comment.char = "")
    b <- read.table(str_c(base_path, "b_", i - 1, ".txt"), quote = "\"", comment.char = "")
    c <- read.table(str_c(base_path, "c_", i - 1, ".txt"), quote = "\"", comment.char = "")
    d <- read.table(str_c(base_path, "d_", i - 1, ".txt"), quote = "\"", comment.char = "")
    x_feature <- 1
    item_size <- length(dt[1,])
    mod <- mirt(dt, x_feature, '4PL', method=method, technical = technical)
    param <- coef(mod, simplify = TRUE)$items
    a_ <- param[, 1:x_feature]
    b_ <- param[, x_feature + 1]
    c_ <- param[, x_feature + 2]
    d_ <- param[, x_feature + 3]
    a_rmse1 <- sum(((-a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    a_rmse2 <- sum(((a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
    b_rmse <- sum(((b_ - b)^2)^0.5) / item_size
    c_rmse <- sum(((c_ - c)^2)^0.5) / item_size
    d_rmse <- sum(((d_ - d)^2)^0.5) / item_size
    print(a_rmse)
    print(b_rmse)
    print(c_rmse)
    print(d_rmse)
    res_a[i] <- a_rmse
    res_b[i] <- b_rmse
    res_c[i] <- c_rmse
    res_d[i] <- d_rmse
  }
  return(list(a=res_a, b=res_b, c=res_c, d=res_d))
}
#res <- test_2pl("irt_2pl_100_", 10, 'EM')
#res <- test_2pl("irt_2pl_100_", 10, 'MHRM')
#res <- test_2pl("irt_2pl_200_", 10, 'EM')
#res <- test_2pl("irt_2pl_200_", 10, 'MHRM')
#res <- test_2pl("irt_2pl_500_", 10, 'EM')
#res <- test_2pl("irt_2pl_500_", 10, 'MHRM')
#res <- test_3pl("irt_3pl_500_", 10, 'EM')
#res <- test_4pl("irt_4pl_1000_", 10, 'MHRM', technical = list(NCYCLES=2000))
#res <- test_2pl("irt_2pl_1000_", 10, 'EM')
res <- test_2pl("irt_2pl_1000_", 10, 'EM', technical = list(NCYCLES=2000))
print(str_c('mean_a:', mean(res$a)))
print(str_c('std_a:', sd(res$a)))
print(str_c('mean_b:', mean(res$b)))
print(str_c('std_b:', sd(res$b)))
print(str_c('mean_c:', mean(res$c)))
print(str_c('std_c:', sd(res$c)))
print(str_c('mean_d:', mean(res$d)))
print(str_c('std_d:', sd(res$d)))
