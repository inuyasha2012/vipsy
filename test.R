library(stringr)
library(mirt)
test_irt <- function(base_path, try_count, method, technical = list(), GenRandomPars = TRUE) {
  model_name <- str_split(base_path, '_', simplify = TRUE)[2]
  param_num <- as.numeric(str_split(model_name, "p", simplify = TRUE)[1])
  if (param_num > 1) {
    res_a <- 1:try_count
  }
  res_b <- 1:try_count
  if (param_num > 2) {
    res_c <- 1:try_count
  }
  if (param_num > 3) {
    res_d <- 1:try_count
  }
  for (i in 1:try_count) {
    dt <- read.table(str_c(base_path, "_", i - 1, ".txt"), quote = "\"", comment.char = "")
    if (param_num > 1) {
      a <- read.table(str_c(base_path, "_a_", i - 1, ".txt"), quote = "\"", comment.char = "")
    }
    b <- read.table(str_c(base_path, "_b_", i - 1, ".txt"), quote = "\"", comment.char = "")
    if (param_num > 2) {
      c <- read.table(str_c(base_path, "_c_", i - 1, ".txt"), quote = "\"", comment.char = "")
    }
    if (param_num > 3) {
      d <- read.table(str_c(base_path, "_d_", i - 1, ".txt"), quote = "\"", comment.char = "")
    }
    x_feature <- length(a[1,])
    item_size <- length(dt[1,])
    tryCatch({
               mod <- mirt(dt, x_feature, str_c(param_num, 'PL'), method = method, technical = technical,
                           GenRandomPars = GenRandomPars)
               param <- coef(mod, simplify = TRUE)$items
               a_ <- param[, 1:x_feature]
               b_ <- param[, x_feature + 1]
               c_ <- param[, x_feature + 2]
               d_ <- param[, x_feature + 3]
               if (param_num > 1) {
                 a_rmse1 <- sum(((-a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
                 a_rmse2 <- sum(((a_ - a)^2)^0.5) / (x_feature * item_size - x_feature * (x_feature - 1) / 2)
                 a_rmse <- min(a_rmse1, a_rmse2)
               }
               b_rmse <- sum(((b_ - b)^2)^0.5) / item_size
               if (param_num > 2) {
                 c_rmse <- sum(((c_ - c)^2)^0.5) / item_size
               }
               if (param_num > 3) {
                 d_rmse <- sum(((d_ - d)^2)^0.5) / item_size
               }
               if (param_num > 1) {
                 print(a_rmse)
               }
               print(b_rmse)
               if (param_num > 2) {
                 print(c_rmse)
               }
               if (param_num > 3) {
                 print(d_rmse)
               }
               if (param_num > 1) {
                 res_a[i] <- a_rmse
               }
               res_b[i] <- b_rmse
               if (param_num > 2) {
                 res_c[i] <- c_rmse
               }
               if (param_num > 3) {
                 res_d[i] <- d_rmse
               }
             }, error = function(e) { cat("ERROR :", conditionMessage(e), "\n") }
    )
  }
  rmse <- list(b = res_b)
  if (param_num > 1) {
    rmse$a <- res_a
  }
  if (param_num > 2) {
    rmse$c <- res_c
  }
  if (param_num > 3) {
    rmse$d <- res_d
  }
  return(rmse)
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
#res <- test_irt("irt_3pl_10000_item_50_dim_5", 10, 'MHRM',
##                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
##                GenRandomPars = TRUE)

res <- test_irt("irt_3pl_10000_item_50_dim_5", 10, 'MHRM',
                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
                GenRandomPars = FALSE)
print(str_c('mean_a:', mean(res$a)))
print(str_c('std_a:', sd(res$a)))
print(str_c('mean_b:', mean(res$b)))
print(str_c('std_b:', sd(res$b)))
print(str_c('mean_c:', mean(res$c)))
print(str_c('std_c:', sd(res$c)))
print(str_c('mean_d:', mean(res$d)))
print(str_c('std_d:', sd(res$d)))
