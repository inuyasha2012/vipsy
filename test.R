library(stringr)
library(mirt)
test_irt <- function(base_path, try_count, method, technical = list(), GenRandomPars = FALSE, custom_pars = FALSE) {
  model_name <- str_split(base_path, '_', simplify = TRUE)[2]
  param_num <- as.numeric(str_split(model_name, "p", simplify = TRUE)[1])
  success_ct <- 0
  time_lt <- NULL
  if (param_num > 1) {
    res_a <- NULL
  }
  res_b <- NULL
  if (param_num > 2) {
    res_c <- NULL
  }
  if (param_num > 3) {
    res_d <- NULL
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
    if (custom_pars) {
      pars <- mirt(
        dt,
        x_feature,
        '2PL',
        pars = 'values'
      )
      for (j in 1:length(pars$name)) {
        if (str_detect(pars$name[i], 'a')) {
          pars$value[j] <- 1
        }
        if (pars$name[j] == 'd') {
          pars$value[j] <- 0
        }
        if (pars$name[j] == 'g') {
          pars$value[j] <- 0.1
        }
        if (pars$name[j] == 'u') {
          pars$value[j] <- 0.9
        }
      }
    }
    tryCatch({
               t1 <- proc.time()
               if (custom_pars) {
                 mod <- mirt(dt, x_feature, str_c(param_num, 'PL'), method = method, technical = technical,
                             GenRandomPars = GenRandomPars, pars = pars)
               } else {
                 mod <- mirt(dt, x_feature, str_c(param_num, 'PL'), method = method, technical = technical,
                             GenRandomPars = GenRandomPars)
               }
               t2 <- proc.time()
               t <- t2 - t1
               time_lt <- append(time_lt, t[3][[1]])
               success_ct <- success_ct + 1
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
                 res_a <- append(res_a, a_rmse)
               }
               res_b <- append(res_b, b_rmse)
               if (param_num > 2) {
                 res_c <- append(res_c, c_rmse)
               }
               if (param_num > 3) {
                 res_d <- append(res_d, d_rmse)
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
  rmse$success_ct <- success_ct
  rmse$time_lt <- time_lt
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

res <- test_irt("irt_3pl_sample_5000_item_50_dim_5", 10, 'MHRM',
                technical = list(NCYCLES = 2, info_if_converged = FALSE, logLik_if_converged = FALSE),
                GenRandomPars = FALSE, custom_pars = FALSE)
print(str_c('success count:', res$success_ct))
print(str_c('time_mean:', mean(remse$time_lt)))
print(str_c('mean_a:', mean(res$a)))
print(str_c('std_a:', sd(res$a)))
print(str_c('mean_b:', mean(res$b)))
print(str_c('std_b:', sd(res$b)))
print(str_c('mean_c:', mean(res$c)))
print(str_c('std_c:', sd(res$c)))
print(str_c('mean_d:', mean(res$d)))
print(str_c('std_d:', sd(res$d)))
