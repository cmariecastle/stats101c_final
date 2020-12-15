library(boot)
library(caret)
library(glmnet)
library(pls)
source("clean_training.R")

# Linear combinations ----------------------------------------------------------

# Initialize objects
yt_train_trimmed <- yt_train
yt_train_scaled_trimmed <- yt_train_scaled
# Find first linear combinations
linear_combos = yt_train_trimmed %>%
  select(where(is.numeric), -id, -growth_2_6) %>%
  findLinearCombos()
# Remove variables until no linear combinations remain
while (!is.null(linear_combos$remove)) {
  # Remove linear combinations
  yt_train_trimmed <- yt_train_trimmed[, -linear_combos$remove]
  yt_train_scaled_trimmed <- yt_train_scaled_trimmed[, -linear_combos$remove]
  # Check for additional linear combinations
  linear_combos <- yt_train_trimmed %>%
    select(where(is.numeric), -id, -growth_2_6) %>%
    findLinearCombos()
}

# Highly correlated variables --------------------------------------------------

# punc_num_( and punc_num_) are obviously correlated so let us remove one
yt_train_trimmed <- select(yt_train_trimmed, -`punc_num_)`)
yt_train_scaled_trimmed <- select(yt_train_scaled_trimmed, -`punc_num_)`)

# Inspect correlated variables
hi_cor <- yt_train_trimmed %>%
  select(-id, -PublishedDate, -growth_2_6) %>%
  as.matrix() %>%
  cor() %>%
  findCorrelation(cutoff = 0.5, names = TRUE, exact = TRUE)
# It appears that many of the hog_* and cnn_* variables are highly correlated
# Let us create sets that do not contain the hog_* variables
hi_cor_hog <- hi_cor[grepl(hi_cor, pattern = "hog")]
yt_train_hog <- select(yt_train_trimmed, !all_of(hi_cor_hog))
yt_train_scaled_hog <- select(yt_train_scaled_trimmed, !all_of(hi_cor_hog))
# Let us create sets that do not contain the cnn_* variables
hi_cor_cnn <- hi_cor[grepl(hi_cor, pattern = "cnn")]
yt_train_hog_cnn <- select(yt_train_hog, !all_of(hi_cor_cnn))
yt_train_scaled_hog_cnn <- select(yt_train_scaled_hog, !all_of(hi_cor_cnn))

# Multiple linear regression ---------------------------------------------------

set.seed(101)

# Control parameters for train() to perform 10-fold cross-validation
tr_ctrl <- trainControl(method = "cv", number = 10)

# yt_train_trimmed
lm_trimmed <- train(
  growth_2_6 ~ ., data = yt_train_trimmed[, -1],
  method = "lm",
  trControl = tr_ctrl
)
metric <- c(lm_trimmed = lm_trimmed$results$RMSE)
# yt_train_scaled_trimmed
lm_scaled_trimmed <- train(
  growth_2_6 ~ ., data = yt_train_scaled_trimmed[, -1],
  method = "lm",
  trControl = tr_ctrl
)
metric <- c(metric, lm_sc_trimmed = lm_scaled_trimmed$results$RMSE)
# yt_train_hog
lm_hog <- train(
  growth_2_6 ~ ., data = yt_train_hog[, -1],
  method = "lm",
  trControl = tr_ctrl
)
metric <- c(metric, lm_hog = lm_hog$results$RMSE)
# yt_train_scaled_hog
lm_scaled_hog <- train(
  growth_2_6 ~ ., data = yt_train_scaled_hog[, -1],
  method = "lm",
  trControl = tr_ctrl
)
metric <- c(metric, lm_sc_hog = lm_scaled_hog$results$RMSE)
# yt_train_hog_cnn
lm_hog_cnn <- train(
  growth_2_6 ~ ., data = yt_train_hog_cnn[, -1],
  method = "lm",
  trControl = tr_ctrl
)
metric <- c(metric, lm_hog_cnn = lm_hog_cnn$results$RMSE)
# yt_train_scaled_hog
lm_scaled_hog_cnn <- train(
  growth_2_6 ~ ., data = yt_train_scaled_hog_cnn[, -1],
  method = "lm",
  trControl = tr_ctrl
)
metric <- c(metric, lm_sc_hog_cnn = lm_scaled_hog_cnn$results$RMSE)

# RMSE
metric

# Shrinkage/Regularization -----------------------------------------------------

set.seed(3)

# Parameters to try
alphas <- c(0, 0.25, 0.5, 0.75, 1)
lambdas <- 10^seq(10, -10, length.out = 100)
tg <- data.frame(
  alpha = rep(alphas, each = length(lambdas)),
  lambda = rep(lambdas, length(alphas))
)

# yt_train_trimmed
shr_trimmed <- train(
  growth_2_6 ~ ., data = yt_train_trimmed[, -1],
  method = "glmnet",
  tuneGrid = tg,
  trControl = tr_ctrl
)
metric <- c(metric, shr_trimmed = min(shr_trimmed$results$RMSE))
# yt_train_scaled_trimmed
shr_scaled_trimmed <- train(
  growth_2_6 ~ ., data = yt_train_scaled_trimmed[, -1],
  method = "glmnet",
  tuneGrid = tg,
  trControl = tr_ctrl
)
metric <- c(metric, shr_sc_trimmed = min(shr_scaled_trimmed$results$RMSE))
# yt_train_hog
shr_hog <- train(
  growth_2_6 ~ ., data = yt_train_hog[, -1],
  method = "glmnet",
  tuneGrid = tg,
  trControl = tr_ctrl
)
metric <- c(metric, shr_hog = min(shr_hog$results$RMSE))
# yt_train_scaled_hog
shr_scaled_hog <- train(
  growth_2_6 ~ ., data = yt_train_scaled_hog[, -1],
  method = "glmnet",
  tuneGrid = tg,
  trControl = tr_ctrl
)
metric <- c(metric, shr_sc_hog = min(shr_scaled_hog$results$RMSE))
# yt_train_hog_cnn
shr_hog_cnn <- train(
  growth_2_6 ~ ., data = yt_train_hog_cnn[, -1],
  method = "glmnet",
  tuneGrid = tg,
  trControl = tr_ctrl
)
metric <- c(metric, shr_hog_cnn = min(shr_hog_cnn$results$RMSE))
# yt_train_scaled_hog_cnn
shr_scaled_hog_cnn <- train(
  growth_2_6 ~ ., data = yt_train_scaled_hog_cnn[, -1],
  method = "glmnet",
  tuneGrid = tg,
  trControl = tr_ctrl
)
metric <- c(metric, shr_sc_hog_cnn = min(shr_scaled_hog_cnn$results$RMSE))

# Dimension reduction ----------------------------------------------------------

set.seed(1013)

# Principal Components Regression (PCR)

# yt_train_scaled_trimmed
pcr_sc_trimmed <- pcr(
  growth_2_6 ~ ., data = yt_train_scaled_trimmed[, -1],
  center = FALSE, scale = FALSE,
  validation = "CV", segments = 10
)
metric <- c(metric, pcr_sc_trimmed = min(RMSEP(pcr_sc_trimmed)$val["CV",,]))
# yt_train_scaled_hog
pcr_sc_hog <- pcr(
  growth_2_6 ~ ., data = yt_train_scaled_hog[, -1],
  center = FALSE, scale = FALSE,
  validation = "CV", segments = 10
)
metric <- c(metric, pcr_sc_hog = min(RMSEP(pcr_sc_hog)$val["CV",,]))
# yt_train_scaled_hog_cnn
pcr_sc_hog_cnn <- pcr(
  growth_2_6 ~ ., data = yt_train_scaled_hog_cnn[, -1],
  center = FALSE, scale = FALSE,
  validation = "CV", segments = 10
)
metric <- c(metric, pcr_sc_hog_cnn = min(RMSEP(pcr_sc_hog_cnn)$val["CV",,]))

# Partial Least Squares (PLS) regression

# yt_train_scaled_trimmed
pls_sc_trimmed <- plsr(
  growth_2_6 ~ ., data = yt_train_scaled_trimmed[, -1],
  center = FALSE, scale = FALSE,
  validation = "CV", segments = 10
)
metric <- c(metric, pls_sc_trimmed = min(RMSEP(pls_sc_trimmed)$val["CV",,]))
# yt_train_scaled_hog
pls_sc_hog <- plsr(
  growth_2_6 ~ ., data = yt_train_scaled_hog[, -1],
  center = FALSE, scale = FALSE,
  validation = "CV", segments = 10
)
metric <- c(metric, pls_sc_hog = min(RMSEP(pls_sc_hog)$val["CV",,]))
# yt_train_scaled_hog_cnn
pls_sc_hog_cnn <- plsr(
  growth_2_6 ~ ., data = yt_train_scaled_hog_cnn[, -1],
  center = FALSE, scale = FALSE,
  validation = "CV", segments = 10
)
metric <- c(metric, pls_sc_hog_cnn = min(RMSEP(pls_sc_hog_cnn)$val["CV",,]))
