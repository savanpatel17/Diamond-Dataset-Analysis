# Name: Savankumar Patel 
#        Vraj Shah
# Date: 2024-05-05  
# Class: ALY6015 Intermediate Analytics
# Initial Report on Diamonds Dataset

# Setup Environment
cat("\014") # Clears the console
rm(list = ls()) # Clears the global environment
options(scipen = 100) # Disables scientific notation for entire R session

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(readr)
library(table1)
library(tidyr)
library(caret)
library(reshape2) 
library(tidyverse)
library(car)
library(MASS)
library(glmnet)
# Load Data
# setwd("C:/Users/VRAJ/OneDrive/Desktop/studies/aly 6015")
diamonds <- read.csv("diamonds.csv")

# Initial Data Inspection
head(diamonds)
str(diamonds)
names(diamonds)

# Check for missing values
print(sum(is.na(diamonds)))

# Check for zero or negative values in dimensions which might be considered as anomalies
anomalies <- sum(diamonds$x <= 0 | diamonds$y <= 0 | diamonds$z <= 0)
print(paste("Anomalous entries found:", anomalies))

# Removing anomalies and missing data if any
diamonds <- diamonds %>% 
  filter(x > 0 & y > 0 & z > 0) %>%
  drop_na()

# Sort the dataframe by 'price' in ascending order
diamonds <- diamonds %>% arrange(price)

#add column name volume by using formula
diamonds <- diamonds %>% 
  mutate(volume = x * y * z)

# Generate descriptive statistics using table1
diamonds_stats <- table1(~ carat  + color + clarity + depth + table + price + volume | cut, data = diamonds)
diamonds_stats



# Univariate Analysis: Visualizing distributions and summaries
# Plot histogram for carat
ggplot(diamonds, aes(x = carat)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black")


# Calculate the correlation matrix for relevant numerical attributes
cor_matrix <- cor(diamonds[, c("carat", "depth", "table", "price", "volume")], use = "complete.obs")

# Melt the correlation matrix for visualization
melted_cor_matrix <- melt(cor_matrix)

# Plot the correlation matrix as a heatmap
ggplot(melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1)) +
  labs(title = "Correlation Matrix of Diamond Attributes",
       x = "Attributes", y = "Attributes")


#Question 1

# Calculate volume
diamonds$volume <- diamonds$x * diamonds$y * diamonds$z

# Summary of volume
summary(diamonds$volume)

# Calculate the IQR for volume
Q1 <- quantile(diamonds$volume, 0.25)
Q3 <- quantile(diamonds$volume, 0.75)
IQR <- Q3 - Q1

# Define the lower and upper bound for outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Filter out the outliers based on IQR method
diamonds_clean <- diamonds %>%
  filter(volume >= lower_bound & volume <= upper_bound)

# Scatter Plot after removing outliers using IQR method
ggplot(diamonds_clean, aes(x = volume, y = price)) +
  geom_point(alpha = 0.3) +
  labs(title = "Scatter Plot of Volume vs. Price (Outliers Removed using IQR Method)",
       x = "Volume (x * y * z)",
       y = "Price") +
  theme_minimal()

# Fit the initial linear model without log transformation
lm_model <- lm(price ~ volume, data = diamonds_clean)
summary(lm_model)

# Plot residuals vs. fitted values to check for homoscedasticity
plot(lm_model$fitted.values, lm_model$residuals)
abline(h = 0, col = "red")

# QQ plot to check for normality of residuals
qqnorm(lm_model$residuals)
qqline(lm_model$residuals, col = "red")

# Histogram of residuals
hist(lm_model$residuals, breaks = 30, main = "Histogram of Residuals", xlab = "Residuals")


# Data Transformation: Log Transformation of Volume
diamonds_clean <- diamonds_clean %>%
  mutate(log_volume = log(volume))

# Data Splitting
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(diamonds_clean$price, p = 0.7, 
                                  list = FALSE, 
                                  times = 1)
diamondsTrain <- diamonds_clean[trainIndex, ]
diamondsTest <- diamonds_clean[-trainIndex, ]

# Linear Regression Model with Log Volume
lm_log_model <- lm(price ~ log_volume, data = diamondsTrain)
summary(lm_log_model)

# Polynomial Regression Model with Log Volume
poly_log_model <- lm(price ~ poly(log_volume, 2), data = diamondsTrain)
summary(poly_log_model)

# Model Performance Evaluation
lm_pred <- predict(lm_log_model, diamondsTest)
poly_pred <- predict(poly_log_model, diamondsTest)

# Calculate R-squared and RMSE
lm_log_r2 <- R2(lm_pred, diamondsTest$price)
lm_log_rmse <- RMSE(lm_pred, diamondsTest$price)

poly_log_r2 <- R2(poly_pred, diamondsTest$price)
poly_log_rmse <- RMSE(poly_pred, diamondsTest$price)

lm_log_r2
lm_log_rmse

poly_log_r2
poly_log_rmse

# Calculate Adjusted R^2, AIC, and BIC for Linear Model
adj_r2_lm_log <- summary(lm_log_model)$adj.r.squared
aic_lm_log <- AIC(lm_log_model)
bic_lm_log <- BIC(lm_log_model)

# Calculate Adjusted R^2, AIC, and BIC for Polynomial Model
adj_r2_poly_log <- summary(poly_log_model)$adj.r.squared
aic_poly_log <- AIC(poly_log_model)
bic_poly_log <- BIC(poly_log_model)

# Print the statistics for comparison
cat("Linear Model with Log-Transformed Volume:\n")
cat("Adjusted R^2:", adj_r2_lm_log, "\n")
cat("AIC:", aic_lm_log, "\n")
cat("BIC:", bic_lm_log, "\n")
cat("R-squared:", lm_log_r2, "\n")
cat("RMSE:", lm_log_rmse, "\n")

cat("\nPolynomial Model with Log-Transformed Volume:\n")
cat("Adjusted R^2:", adj_r2_poly_log, "\n")
cat("AIC:", aic_poly_log, "\n")
cat("BIC:", bic_poly_log, "\n")
cat("R-squared:", poly_log_r2, "\n")
cat("RMSE:", poly_log_rmse, "\n")

# Diagnostic Plots for Polynomial Model with Log-Transformed Volume
# Plot residuals vs. fitted values to check for homoscedasticity
plot(poly_log_model$fitted.values, poly_log_model$residuals)
abline(h = 0, col = "red")

# QQ plot to check for normality of residuals
qqnorm(poly_log_model$residuals)
qqline(poly_log_model$residuals, col = "red")

# Histogram of residuals
hist(poly_log_model$residuals, breaks = 30, main = "Histogram of Residuals", xlab = "Residuals")


# Cross-Validation
set.seed(123)
cv_results_lm <- train(price ~ log_volume, data = diamonds_clean, method = "lm",
                       trControl = trainControl(method = "cv", number = 10))
print(cv_results_lm)

cv_results_poly <- train(price ~ poly(log_volume, 2), data = diamonds_clean, method = "lm",
                         trControl = trainControl(method = "cv", number = 10))
print(cv_results_poly)

# Final Model Evaluation
best_model <- ifelse(lm_log_rmse < poly_log_rmse, lm_log_model, poly_log_model)
best_pred <- ifelse(lm_log_rmse < poly_log_rmse, lm_pred, poly_pred)

# Predicting prices using the polynomial model with log-transformed volume
best_pred <- predict(poly_log_model, diamondsTest)

# Creating a data frame for actual and predicted values
prediction_data <- data.frame(log_volume = diamondsTest$log_volume, Actual = diamondsTest$price, Predicted = best_pred)

# Plotting the predictions vs. actual values
ggplot(prediction_data, aes(x = log_volume)) +
  geom_point(aes(y = Actual), alpha = 0.3) +
  geom_line(aes(y = Predicted), color = "red") +
  labs(title = "Final Model: Predictions vs. Actual",
       x = "Log Volume (log(x * y * z))",
       y = "Price") +
  theme_minimal()

#Question 2

# Create boxplots to visualize the relationship between cut, color, clarity, and price
# Create boxplot for price by cut quality, removing the legend and sorting the x-axis
ggplot(diamonds, aes(x = cut, y = price, fill = cut)) +
  geom_boxplot() +
  labs(title = "Price by Cut Quality", x = "Cut Quality", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

# Create boxplot for price by color, removing the legend
ggplot(diamonds, aes(x = color, y = price, fill = color)) +
  geom_boxplot() +
  labs(title = "Price by Color Grade", x = "Color Grade", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

# Create boxplot for price by clarity, removing the legend
ggplot(diamonds, aes(x = clarity, y = price, fill = clarity)) +
  geom_boxplot() +
  labs(title = "Price by Clarity Grade", x = "Clarity Grade", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

# Ensure the plot is rendered before proceeding
dev.off()

#Question 3


# Ensure categorical variables are factors
diamonds$cut <- as.factor(diamonds$cut)
diamonds$color <- as.factor(diamonds$color)
diamonds$clarity <- as.factor(diamonds$clarity)

# Fit the multiple linear regression model
mlr_model <- lm(price ~ carat + cut + color + clarity + depth + table + volume, data = diamonds)

# Summarize the model
summary(mlr_model)

vif(mlr_model)

# Prepare the data for Ridge regression
x <- model.matrix(price ~ carat + cut + color + clarity + depth + table + volume, data = diamonds)[, -1]
y <- diamonds$price

# Standardize the predictors
x <- scale(x)


# Set the seed for reproducibility
set.seed(123)

# Perform cross-validation to find the optimal lambda
cv_ridge <- cv.glmnet(x, y, alpha = 0)

# Get the optimal lambda
best_lambda <- cv_ridge$lambda.min
best_lambda

# Fit the Ridge regression model with the optimal lambda
ridge_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)

# Display the coefficients of the Ridge model
ridge_coefficients <- coef(ridge_model)
print(ridge_coefficients)

# Predict on the standardized data
ridge_predictions <- predict(ridge_model, s = best_lambda, newx = x)

# Calculate RMSE and R-squared for the Ridge model
ridge_rmse <- sqrt(mean((y - ridge_predictions)^2))
ridge_r2 <- 1 - sum((y - ridge_predictions)^2) / sum((y - mean(y))^2)

# Print the evaluation metrics
cat("Ridge Regression Model:\n")
cat("RMSE:", ridge_rmse, "\n")
cat("R-squared:", ridge_r2, "\n")

# Residual Analysis
residuals <- as.numeric(y - ridge_predictions)  # Ensure residuals are numeric

# Residual Plot
residual_data <- data.frame(Fitted = as.numeric(ridge_predictions), Residuals = residuals)
ggplot(residual_data, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot", x = "Fitted Values", y = "Residuals") +
  theme_minimal()

# QQ Plot
qqnorm(residuals, main = "QQ Plot of Residuals")
qqline(residuals, col = "red")

# Histogram of Residuals
ggplot(residual_data, aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Residuals", x = "Residuals", y = "Frequency") +
  theme_minimal()

# Summary Statistics of Residuals
summary(residuals)


#Question 4

# Apply log transformation to skewed variables (excluding price)
diamonds <- diamonds %>%
  mutate(log_carat = log(carat),
         log_volume = log(volume))

# Fit the multiple linear regression model with log-transformed predictors
mlr_model <- lm(price ~ log_carat + cut + color + clarity + depth + table + log_volume, data = diamonds)

# Summarize the model
summary(mlr_model)

# Check for multicollinearity in the transformed model
vif(mlr_model)

# Prepare the data for Lasso regression
x <- model.matrix(price ~ log_carat + cut + color + clarity + depth + table + log_volume, data = diamonds)[, -1]
y <- diamonds$price

# Standardize the predictors
x <- scale(x)

# Perform cross-validation to find the optimal lambda for Lasso
set.seed(123)
cv_lasso <- cv.glmnet(x, y, alpha = 1)

# Get the optimal lambda
best_lambda_lasso <- cv_lasso$lambda.min
best_lambda_lasso

# Fit the Lasso regression model with the optimal lambda
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda_lasso)
lasso_model

# Display the coefficients of the Lasso model
lasso_coefficients <- coef(lasso_model)
print(lasso_coefficients)

# Cross-validation for the MLR model with log-transformed predictors
set.seed(123)
cv_mlr <- train(price ~ log_carat + cut + color + clarity + depth + table + log_volume, 
                data = diamonds, 
                method = "lm", 
                trControl = trainControl(method = "cv", number = 10))

print(cv_mlr)

# Cross-validation for Lasso regression
set.seed(123)
cv_lasso_model <- train(price ~ log_carat + cut + color + clarity + depth + table + log_volume, 
                        data = diamonds, 
                        method = "glmnet", 
                        trControl = trainControl(method = "cv", number = 10), 
                        tuneGrid = expand.grid(alpha = 1, lambda = best_lambda_lasso))

print(cv_lasso_model)

# Compare RMSE and R-squared for all models
models <- list(MLR = cv_mlr, Lasso = cv_lasso_model)
results <- resamples(models)
summary(results)

# Predict on the standardized data for Lasso
lasso_predictions <- predict(lasso_model, s = best_lambda_lasso, newx = x)
residuals_lasso <- y - lasso_predictions

# Residual Plot
residual_data_lasso <- data.frame(Fitted = as.numeric(lasso_predictions), Residuals = as.numeric(residuals_lasso))
ggplot(residual_data_lasso, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.3) +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Lasso Regression: Residual Plot", x = "Fitted Values", y = "Residuals") +
  theme_minimal()

# QQ Plot
qqnorm(residuals_lasso, main = "Lasso Regression: QQ Plot of Residuals")
qqline(residuals_lasso, col = "red")

# Histogram of Residuals
ggplot(residual_data_lasso, aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Lasso Regression: Histogram of Residuals", x = "Residuals", y = "Frequency") +
  theme_minimal()

