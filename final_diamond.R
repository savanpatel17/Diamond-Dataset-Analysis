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


# Load Data
diamonds <- read.csv("diamond_modified.csv")

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

# Generate descriptive statistics using table1
diamonds_stats <- table1(~ carat + cut + color + clarity + depth + table + price + x + y + z | cut, data = diamonds)
diamonds_stats

# Univariate Analysis: Visualizing distributions and summaries
# Plot histogram for carat
ggplot(diamonds, aes(x = carat)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black")

# Create boxplots to visualize the relationship between cut, color, clarity, and price
# Boxplot for cut
ggplot(diamonds, aes(x = cut, y = price, fill = cut)) +
  geom_boxplot() +
  labs(title = "Price by Cut Quality", x = "Cut Quality", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Boxplot for color
ggplot(diamonds, aes(x = color, y = price, fill = color)) +
  geom_boxplot() +
  labs(title = "Price by Color Grade", x = "Color Grade", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Boxplot for clarity
ggplot(diamonds, aes(x = clarity, y = price, fill = clarity)) +
  geom_boxplot() +
  labs(title = "Price by Clarity Grade", x = "Clarity Grade", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Convert factors to appropriate format
diamonds$cut <- factor(diamonds$cut, levels = c("Fair", "Good", "Very Good", "Premium", "Ideal"))
diamonds$color <- factor(diamonds$color, levels = c("J", "I", "H", "G", "F", "E", "D"))
diamonds$clarity <- factor(diamonds$clarity, levels = c("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"))

# Calculate the correlation matrix for relevant numerical attributes
cor_matrix <- cor(diamonds[, c("carat", "depth", "table", "price", "x", "y", "z")], use = "complete.obs")

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

# Log transformation of price and volume
diamonds$log_price <- log(diamonds$price + 1)  # Adding 1 to avoid log(0)
diamonds$log_volume <- log(diamonds$volume + 1)


# Basic scatter plot with a regression line
ggplot(diamonds, aes(x=log_volume, y=log_price)) +
  geom_point(alpha=0.5) +  # Adjust alpha for point transparency if points are dense
  geom_smooth(method="lm", se=FALSE, color="blue") +
  labs(title="Log Volume vs Log Price with Regression Line",
       x="Log Volume (cubic mm)", y="Log Price (USD)") +
  theme_minimal()


# Model to see if volume predicts price
model <- lm(price ~ volume, data = diamonds)
summary(model)




#Question 2


# Fitting the linear regression model
lm_model <- lm(price ~ cut + color + clarity, data = diamonds)
summary(lm_model)

vif(lm_model)  # Variance inflation factors should be less than 5 ideally

# ANOVA to check the impact of each factor
anova_model <- aov(price ~ cut + color + clarity, data = diamonds)
summary(anova_model)

# Plotting residuals
par(mfrow=c(2,2))
plot(lm_model)

# Ensure that the diamonds dataset has no missing values in the relevant columns
diamonds <- na.omit(diamonds)  # Remove all rows with any NA values

# Confirm that there are no missing values left
sum(is.na(diamonds$price))
sum(is.na(diamonds$cut))
sum(is.na(diamonds$color))
sum(is.na(diamonds$clarity))

# Redefine x and y for the Ridge Regression
x <- model.matrix(price ~ cut + color + clarity - 1, data = diamonds)  # -1 to omit intercept
y <- diamonds$price

# Check to ensure that dimensions match
print(paste("Rows in x:", nrow(x)))
print(paste("Length of y:", length(y)))


ridge_model <- glmnet(x, y, alpha = 0)
summary(ridge_model)
plot(ridge_model)

# Coefficients from the best model
coef(ridge_model, s = s_lambda_min)  # Extract coefficients at the lambda that gives minimum cross-validation error

#Question3

# Split the data into training and testing sets (70% training, 30% testing)
set.seed(123) # Ensure reproducibility
trainIndex <- createDataPartition(diamonds$price, p = 0.7, list = FALSE)
trainData <- diamonds[trainIndex, ]
testData <- diamonds[-trainIndex, ]

# Ensure that log_price and log_volume are not included in the dataset
trainData$log_price <- NULL
trainData$log_volume <- NULL
testData$log_price <- NULL
testData$log_volume <- NULL

# Train the Generalized Linear Model
glm_model <- glm(price ~ ., data = trainData, family = gaussian())

# Print the model summary
summary(glm_model)

# Make predictions on the test data
predictions_glm <- predict(glm_model, newdata = testData, type = "response")

# Evaluate the model
mse_glm <- mean((testData$price - predictions_glm)^2)
mae_glm <- mean(abs(testData$price - predictions_glm))
rsquared_glm <- cor(predictions_glm, testData$price)^2

# Print evaluation metrics
cat("Mean Squared Error (MSE):", mse_glm, "\n")
cat("Mean Absolute Error (MAE):", mae_glm, "\n")
cat("R-squared:", rsquared_glm, "\n")

##########################lasso#####################################

# Load the necessary library for Ridge regression
library(glmnet)

# Prepare the data: Convert categorical variables to numeric using dummy variables
# Ensure categorical variables maintain consistent factor levels
diamonds$cut <- factor(diamonds$cut, levels = c("Fair", "Good", "Ideal", "Premium", "Very Good", "verygood"))
diamonds$color <- factor(diamonds$color, levels = c("D", "E", "F", "G", "H", "I", "J"))
diamonds$clarity <- factor(diamonds$clarity, levels = c("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"))

# Create the model matrix excluding the intercept
x <- model.matrix(price ~ . - 1, data = diamonds)  
y <- diamonds$price

# Fit a Ridge regression model to the data
ridge_model <- glmnet(x, y, alpha = 0)

# Perform cross-validation to determine the best lambda value for regularization
cv_ridge <- cv.glmnet(x, y, alpha = 0)
plot(cv_ridge)  # Visualize the cross-validation results

# Extract the best lambda value
best_lambda <- cv_ridge$lambda.min

# Refit the Ridge model using the optimal lambda
ridge_model_best <- glmnet(x, y, alpha = 0, lambda = best_lambda)

# View the coefficients
print(coef(ridge_model_best))

# To ensure newx matches, use the same model.matrix approach
predictions <- predict(ridge_model_best, s = best_lambda, newx = x)  # Use x as it is the same data

# Print predictions
print(predictions)

