# Packages used ----------------------------------------------------------------
library(boot)
library(caret)
library(glmnet)
library(pls)
library(randomForest)
library(tidyverse)

### FINAL MODEL ################################################################

# Step 0: Importing data -------------------------------------------------------

yt_train_uncleaned <- read.csv("training.csv")
test <- read.csv("test.csv")

# Step 1: Cleaning the data ----------------------------------------------------

# Any NA values? No
any(apply(yt_train_uncleaned, 2, function(x) sum(is.na(x))) > 0)
# Any infinite values? No
any(apply(yt_train_uncleaned, 2, function(x) sum(is.infinite(x))) > 0)
# Any NaN values? No
any(apply(yt_train_uncleaned, 2, function(x) sum(is.nan(x))) > 0)
# Any duplicated rows (excluding id)? No
sum(duplicated(yt_train_uncleaned[, -1]))
# Are there any duplicated id's? No
sum(duplicated(yt_train_uncleaned$id))

# Are there unusual duration values? No
summary(yt_train_uncleaned$Duration)
# No 0 second videos
# A video of 42895 seconds (or ~12 hours) isn't unusual on YouTube

# Are there unusual views_2_hours values? No
summary(yt_train_uncleaned$views_2_hours)
# No negative values
# Nothing too unusual here either unless you count the outliers

# Indices of constant variables (i.e. all same value)
useless_variables <- which(
  apply(yt_train_uncleaned, 2, function(x) sum(length(unique(x))) == 1)
)
# Remove constant variables
training <- yt_train_uncleaned[, -c(useless_variables)]
test <- test[,-c(useless_variables)]

# write.csv(training, "training_clean.csv", row.names = FALSE)
# write.csv(test, "test_clean.csv", row.names = FALSE)
# 
# #Loading in the cleaned data
# training <- read.csv("training_clean.csv")
# test <- read.csv("test_clean.csv")

# Step 2: Optimizing correlation cutoff ----------------------------------------

set.seed(1)

#Removing PublishedDate
training <- training[,-2]
test <- test[,-2]

#Saving the test id's for later
testid <- cbind(test$id)

#Removing the id's
training <- training[,-1]
test <- test[,-1]

#Creating a correlation matrix for use in findCorrelation
corrmx <- cor(training)

#Running through the different thresholds

#Testing 0.5
correlated_vars <- findCorrelation(corrmx, cutoff = 0.5)
temp_training <- training[,-correlated_vars]
temp_test <- test[,-correlated_vars]

ncol(temp_training)
ncol(temp_test)

forest_uncorr05 <- tuneRF(
  x = temp_training[,-153], y = temp_training$growth_2_6,
  plot = TRUE, doBest = TRUE
)
print(forest_uncorr05)

#Testing 0.6
correlated_vars <- findCorrelation(corrmx, cutoff = 0.6)
temp_training <- training[,-correlated_vars]
temp_test <- test[,-correlated_vars]

ncol(temp_training)
ncol(temp_test)

forest_uncorr06 <- tuneRF(
  x = temp_training[,-165], y = temp_training$growth_2_6,
  plot = TRUE, doBest = TRUE
)
print(forest_uncorr06)

#Testing 0.7
correlated_vars <- findCorrelation(corrmx, cutoff = 0.7)
temp_training <- training[,-correlated_vars]
temp_test <- test[,-correlated_vars]

ncol(temp_training)
ncol(temp_test)

forest_uncorr07 <- tuneRF(
  x = temp_training[,-167], y = temp_training$growth_2_6,
  plot = TRUE, doBest = TRUE
)
print(forest_uncorr07)

#Testing 0.8
correlated_vars <- findCorrelation(corrmx, cutoff = 0.8)
temp_training <- training[,-correlated_vars]
temp_test <- test[,-correlated_vars]

ncol(temp_training)
ncol(temp_test)

forest_uncorr08 <- tuneRF(
  x = temp_training[,-172], y = temp_training$growth_2_6,
  plot = TRUE, doBest = TRUE
)
print(forest_uncorr08)

#Testing 0.9
correlated_vars <- findCorrelation(corrmx, cutoff = 0.9)
temp_training <- training[,-correlated_vars]
temp_test <- test[,-correlated_vars]

ncol(temp_training)
ncol(temp_test)

forest_uncorr09 <- tuneRF(
  x = temp_training[,-176], y = temp_training$growth_2_6,
  plot = TRUE, doBest = TRUE
)
print(forest_uncorr09)

# Step 3: Variable selection ---------------------------------------------------

set.seed(1)

#Finding and removing variables with correlations > 0.8
correlated_vars <- findCorrelation(corrmx, cutoff = 0.8)
training_uncorr <- training[,-correlated_vars]
test_uncorr <- test[,-correlated_vars]

#Finding the optimal mtry Random Forest model
#using the new data without the correlated variables
forest_uncorr <- tuneRF(
  x = training_uncorr[,-172], y = training_uncorr$growth_2_6,
  plot = TRUE, doBest = TRUE
)
# OOB RMSE
sqrt(min(forest_uncorr$mse))

#Plotting to see what variables are unimportant
varImpPlot(forest_uncorr, sort = FALSE, n.var = 40)

#Removing all hog variables up until hog_485
training_uncorr <- training_uncorr[,-(3:40)]
test_uncorr <- test_uncorr[,-(3:40)]

#Fitting a new RF to this new dataset
without_hog485 <- tuneRF(
  x = training_uncorr[,-134], y = training_uncorr$growth_2_6,
  plot = TRUE, doBest = TRUE
)
# OOB RMSE
sqrt(min(without_hog485$mse))

#Plotting to see what variables are unimportant
varImpPlot(without_hog485, sort = FALSE, n.var = 40)

#Remove the rest of the hog variables
training_uncorr <- training_uncorr[,-(3:53)]
test_uncorr <- test_uncorr[,-(3:53)]

ncol(training_uncorr)
ncol(test_uncorr)

#Fitting a new RF to this new dataset
without_allhog <- tuneRF(
  x = training_uncorr[,-83], y = training_uncorr$growth_2_6,
  plot = TRUE, doBest = TRUE
)

# OOB RMSE
sqrt(min(without_allhog$mse))

#Remove more variables deemed unimportant from varImpPlot
varImpPlot(without_allhog, sort = FALSE, n.var = 40)

#The cnn and max variables are found to be unimportant
#Removing max_blue,green,red plus cnn_0,9,20,36,39,65
unnecessary <- c(3, 4, 6, 8, 9, 10, 16, 19, 20)

training_uncorr <- training_uncorr[,-unnecessary]
test_uncorr <- test_uncorr[,-unnecessary]

ncol(training_uncorr)
ncol(test_uncorr)

# Step 4: Final tuning ---------------------------------------------------------

#Fitting a new RF to this new dataset
without_cnnmax <- tuneRF(
  x = training_uncorr[,-74], y = training_uncorr$growth_2_6,
  plot = TRUE, doBest = TRUE
)

# OOB RMSE
sqrt(min(without_allhog$mse))

final <- cbind(testid, predict(without_cnnmax, newdata = test_uncorr))
colnames(final) <- c("id", "growth_2_6")
# write.csv(final, "rmcnn_max_final.csv", row.names = FALSE)

### SHRINKAGE METHODS ##########################################################

# training <- read.csv("training_clean.csv")
# test <- read.csv("test_clean.csv")

set.seed(1)
# Values to try for lambda
grid <- 10^seq(10, -3, length = 100)

# Separating predictors and response
uncorr_x <- model.matrix(growth_2_6 ~ ., data = training_uncorr)[, -1]
uncorr_y <- training_uncorr$growth_2_6
uncorr_test_x <- data.matrix(test_uncorr)

# LASSO regression
uncorr_lasso <- glmnet(uncorr_x, uncorr_y, alpha = 1, family = "gaussian",
                       lambda = grid, standardize = TRUE)
uncorr_cv <- cv.glmnet(uncorr_x, uncorr_y, alpha = 1, family = "gaussian",
                       lambda = grid, standardize = TRUE,
                       nfolds = 10, type.measure = "mse")
# Lowest RMSE (square root of mean cross-validated error)
sqrt(min(uncorr_cv$cvm))
lambda <- uncorr_cv$lambda.min
pred_lasso <- predict(uncorr_lasso, newx = uncorr_test_x, s = lambda)

# Ridge regression
uncorr_ridge <- glmnet(uncorr_x, uncorr_y, alpha = 0, family = "gaussian", 
                       lambda = grid, standardize = TRUE)
uncorr_cv2 <- cv.glmnet(uncorr_x, uncorr_y, alpha = 0, family = "gaussian", 
                        lambda = grid, standardize = TRUE,
                        nfolds = 10, type.measure = "mse")
# Lowest RMSE (square root of mean cross-validated error)
sqrt(min(uncorr_cv2$cvm))
lambda <- uncorr_cv$lambda.min
pred_ridge <- predict(uncorr_ridge, newx = uncorr_test_x, s = lambda)

lasso_data <- cbind(id, pred_lasso)
colnames(lasso_data) <- c("id", "growth_2_6")
# write.csv(lasso_data, "lasso_uncorr.csv", row.names = FALSE)

ridge_data <- cbind(id, pred_ridge)
colnames(ridge_data) <- c("id", "growth_2_6")
# write.csv(ridge_data, "ridge_uncorr.csv", row.names = FALSE)

### OTHER COMPARED METHODS #####################################################

# Including PublishedDate ------------------------------------------------------

yt_test_original <- read_csv("test.csv")
yt_train_original <- read_csv("training.csv")

# Convert PublishedDate to number of seconds since 1970-01-01 00:00:00 UTC
yt_test <- mutate(
  yt_test_original,
  PublishedDate = as.numeric(
    as.POSIXct(PublishedDate, "%m/%d/%Y %H:%M", tz = "UTC")
  )
)
yt_train <- mutate(
  yt_train_original,
  PublishedDate = as.numeric(
    as.POSIXct(PublishedDate, "%m/%d/%Y %H:%M", tz = "UTC")
  )
)

# Remove variables w/same value for all observations
yt_train <- yt_train[, -useless_variables]
yt_test <- yt_test[, -useless_variables]

# Scaling predictors -----------------------------------------------------------

# Separate id variable
yt_train_id <- yt_train$id
yt_train_scaled <- select(yt_train, -id)
yt_test_id <- yt_test$id
yt_test_scaled <- select(yt_test, -id)
# Separate binary variables and response
yt_train_end <- select(yt_train, Num_Subscribers_Base_low:last_col())
yt_train_scaled <- select(yt_train_scaled, !Num_Subscribers_Base_low:last_col())
yt_test_end <- select(yt_test, Num_Subscribers_Base_low:last_col())
yt_test_scaled <- select(yt_test_scaled, !Num_Subscribers_Base_low:last_col())
# Scale remaining numeric variables
yt_train_scaled <- scale(yt_train_scaled)
yt_test_scaled <- scale(
  yt_test_scaled,
  # Use the same centering and scaling factors as training
  center = attr(yt_train_scaled, "scaled:center"),
  scale = attr(yt_train_scaled, "scaled:scale")
)
# Rejoin separated variables
yt_train_scaled <- bind_cols(id = yt_train_id, yt_train_scaled, yt_train_end)
yt_test_scaled <- bind_cols(id = yt_test_id, yt_test_scaled, yt_test_end)

# Removing linear combinations -------------------------------------------------

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

# Removing highly correlated variables -----------------------------------------

# punc_num_( and punc_num_) are obviously correlated so remove one
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

# Dimension reduction methods --------------------------------------------------

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

sort(metric)
