library(tidyverse)

yt_test_original <- read_csv("test.csv")
yt_train_uncleaned <- read_csv("training.csv")

# Convert PublishedDate to number of seconds since 1970-01-01 00:00:00 UTC
yt_test <- mutate(
  yt_test_original,
  PublishedDate = as.numeric(
    as.POSIXct(PublishedDate, "%m/%d/%Y %H:%M", tz = "UTC")
  )
)
yt_train <- mutate(
  yt_train_uncleaned,
  PublishedDate = as.numeric(
    as.POSIXct(PublishedDate, "%m/%d/%Y %H:%M", tz = "UTC")
  )
)

# Function that tests if all elements in a vector are the same value
all_same <- function(x) {
  all(x == rep_along(x, head(x, 1)))
}

# Remove variables w/same value for all observations
useless <- names(select(yt_train, where(all_same)))
yt_train <- select(yt_train, !all_of(useless))
yt_test <- select(yt_test, !all_of(useless))

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
