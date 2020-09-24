library(stringr)
library(mirt)
library(GDINA)
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
    dt[dt == 'NaN'] = NA
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
  print(str_c('success count:', rmse$success_ct))
  print(str_c('time_mean:', mean(rmse$time_lt)))
  print(str_c('mean_a:', mean(rmse$a)))
  print(str_c('std_a:', sd(rmse$a)))
  print(str_c('mean_b:', mean(rmse$b)))
  print(str_c('std_b:', sd(rmse$b)))
  print(str_c('mean_c:', mean(rmse$c)))
  print(str_c('std_c:', sd(rmse$c)))
  print(str_c('mean_d:', mean(rmse$d)))
  print(str_c('std_d:', sd(rmse$d)))
  return(rmse)
}
test_cdm <- function(base_path, try_count, method) {
  model_name <- str_split(base_path, '_', simplify = TRUE)[2]
  if (model_name == 'ho') {
    model_name <- 'ho-dina'
  }
  success_ct <- 0
  res_g <- NULL
  res_s <- NULL
  if (model_name == 'ho-dina') {
    res_lam0 <- NULL
    res_lam1 <- NULL
  }
  time_lt <- NULL
  for (i in 1:try_count) {
    dt <- read.table(str_c(base_path, "_", i - 1, ".txt"), quote = "\"", comment.char = "")
    g <- read.table(str_c(base_path, "_g_", i - 1, ".txt"), quote = "\"", comment.char = "")
    s <- read.table(str_c(base_path, "_s_", i - 1, ".txt"), quote = "\"", comment.char = "")
    Q <- read.table(str_c(base_path, "_q_", i - 1, ".txt"), quote = "\"", comment.char = "")
    if (model_name == 'ho-dina') {
      lam0 <- read.table(str_c(base_path, "_lam0_", i - 1, ".txt"), quote = "\"", comment.char = "")
      lam1 <- read.table(str_c(base_path, "_lam1_", i - 1, ".txt"), quote = "\"", comment.char = "")
    }
    item_size <- length(dt[1,])
    t1 <- proc.time()
    if (model_name == 'ho-dina') {
      mod <- GDINA(dat = dt, Q = t(Q), model = 'dina',
                   att.dist = "higher.order", higher.order = list(model = "2PL", Prior = TRUE))
    } else {
      mod <- GDINA(dat = dt, Q = t(Q), model = model_name)
    }
    t2 <- proc.time()
    t <- t2 - t1
    time_lt <- append(time_lt, t[3][[1]])
    success_ct <- success_ct + 1
    param <- coef(mod, what = "gs")
    g_ <- param[, 1]
    s_ <- param[, 2]
    g_rmse <- sum(((g_ - g)^2)^0.5) / item_size
    s_rmse <- sum(((s_ - s)^2)^0.5) / item_size
    print(g_rmse)
    print(s_rmse)
    res_g <- append(res_g, g_rmse)
    res_s <- append(res_s, s_rmse)
    if (model_name == 'ho-dina') {
      param <- coef(mod, what = "lambda")
      lam0_ <- param[, 2]
      lam1_ <- param[, 1]
      lam0_rmse <- sum(((lam0_ - lam0)^2)^0.5) / length(Q[, 1])
      lam1_rmse <- sum(((lam1_ - lam1)^2)^0.5) / length(Q[, 1])
      print(lam0_rmse)
      print(lam1_rmse)
      res_lam0 <- append(res_lam0, lam0_rmse)
      res_lam1 <- append(res_lam1, lam1_rmse)
    }
  }
  rmse <- list(g = res_g, s = res_s, lam0 = res_lam0, lam1 = res_lam1)
    rmse$success_ct <- success_ct
    rmse$time_lt <- time_lt
    print(str_c('success count:', rmse$success_ct))
    print(str_c('time_mean:', mean(rmse$time_lt)))
    print(str_c('mean_g:', mean(rmse$g)))
    print(str_c('std_g:', sd(rmse$g)))
    print(str_c('mean_s:', mean(rmse$s)))
    print(str_c('std_s:', sd(rmse$s)))
    if (model_name == 'ho-dina') {
      print(str_c('mean_lam0:', mean(rmse$lam0)))
      print(str_c('std_lam0:', sd(rmse$lam0)))
      print(str_c('mean_lam1:', mean(rmse$lam1)))
      print(str_c('std_lam1:', sd(rmse$lam1)))
    }
  return(rmse)
}
#print('100样本，50题，1维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_100_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('100样本，50题，1维，2参数，MHRM算法')
#res <- test_irt("dt/irt_2pl_sample_100_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('200样本，50题，1维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_200_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('200样本，50题，1维，2参数，MHRM算法')
#res <- test_irt("dt/irt_2pl_sample_200_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('500样本，50题，1维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_500_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('500样本，50题，1维，2参数，MHRM算法')
#res <- test_irt("dt/irt_2pl_sample_500_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('500样本，50题，1维，3参数，EM算法')
#res <- test_irt("dt/irt_3pl_sample_500_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('500样本，50题，1维，3参数，MHRM算法')
#res <- test_irt("dt/irt_3pl_sample_500_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('1000样本，50题，1维，3参数，EM算法')
#res <- test_irt("dt/irt_3pl_sample_1000_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('1000样本，50题，1维，3参数，MHRM算法')
#res <- test_irt("dt/irt_3pl_sample_1000_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('500样本，50题，1维，4参数，EM算法')
#res <- test_irt("dt/irt_4pl_sample_500_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('500样本，50题，1维，4参数，MHRM算法')
#res <- test_irt("dt/irt_4pl_sample_500_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = TRUE)
#print('1000样本，50题，1维，4参数，EM算法')
#res <- test_irt("dt/irt_4pl_sample_1000_item_50_dim_1", 10, 'EM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('1000样本，50题，1维，4参数，MHRM算法')
#res <- test_irt("dt/irt_4pl_sample_1000_item_50_dim_1", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = TRUE)
#print('1000样本，50题，2维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_1000_item_50_dim_2", 10, 'EM',
#                technical = list(NCYCLES = 500, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('1000样本，50题，2维，2参数，MHRM算法')
#res <- test_irt("dt/irt_2pl_sample_1000_item_50_dim_2", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('1000样本，50题，3维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_1000_item_50_dim_3", 5, 'EM',
#                technical = list(NCYCLES = 500, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = TRUE)
#print('1000样本，50题，3维，2参数，MHRM算法')
#res <- test_irt("dt/irt_2pl_sample_1000_item_50_dim_3", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('5000样本，50题，3维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_5000_item_50_dim_3", 1, 'EM',
#                technical = list(NCYCLES = 500, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = FALSE)
#print('10000样本，50题，5维，2参数，EM算法')
#res <- test_irt("dt/irt_2pl_sample_5000_item_50_dim_3", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('10000样本，50题，5维，2参数，MHRM算法')
#res <- test_irt("dt/irt_2pl_sample_10000_item_50_dim_5", 4, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('10000样本，50题，5维，3参数，MHRM算法')
#res <- test_irt("dt/irt_3pl_sample_10000_item_50_dim_5", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = FALSE, custom_pars = TRUE)
#print('mil,5000样本，50题，5维，3参数，MHRM算法')
#res <- test_irt("mil/irt_3pl_sample_5000_item_50_dim_5", 10, 'MHRM',
#                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
#                GenRandomPars = TRUE, custom_pars = FALSE)
#print('mil,5000样本，50题，5维，4参数，MHRM算法')
##res <- test_irt("mil/irt_4pl_sample_5000_item_50_dim_5", 10, 'MHRM',
##                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
##                GenRandomPars = FALSE, custom_pars = TRUE)
#print('dina, 100000样本')
#res <- test_cdm("cdm/cdm_ho_dina_sample_10000_item_100", 4)
print('irt, 缺失数据')
res <- test_irt("miss/irt_2pl_sample_10000_item_500_dim_1", 4, 'EM',
                technical = list(NCYCLES = 2000, info_if_converged = FALSE, logLik_if_converged = FALSE),
                GenRandomPars = FALSE, custom_pars = FALSE)