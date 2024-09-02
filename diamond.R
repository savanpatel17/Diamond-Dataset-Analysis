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

# Load Data
diamonds_old <- read_csv("diamonds.csv")

# Initial Data Inspection
print(head(diamonds))
str(diamonds)

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
print(diamonds_stats)

# Univariate Analysis: Visualizing distributions and summaries
# Histogram for carat
ggplot(diamonds, aes(x = carat)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Carat", x = "Carat", y = "Frequency")

# Boxplot for price grouped by cut
ggplot(diamonds, aes(x = cut, y = price, fill = cut)) +
  geom_boxplot() +
  labs(title = "Price by Cut", x = "Cut", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Bivariate Analysis: Exploring relationships between variables
# Scatter plot of price vs. carat, colored by cut
grouped_data <- diamonds %>% group_by(cut)
ggplot(grouped_data, aes(x = carat, y = price, color = cut)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +  # Linear trend line
  labs(title = "Price vs. Carat by Cut", x = "Carat", y = "Price") +
  scale_color_manual(values = c("Fair" = "red", "Good" = "orange", "Very Good" = "yellow", "Premium" = "green", "Ideal" = "blue"))

# Load necessary libraries
library(reshape2) 

# Calculate the correlation matrix for relevant numerical attributes
cor_matrix <- cor(diamonds[, c("carat", "depth", "table", "price", "x", "y", "z")], use = "complete.obs")

# Melt the correlation matrix
melted_cor_matrix <- melt(cor_matrix, varnames = c("Variable1", "Variable2"))

# Plot the correlation matrix
ggplot(data = melted_cor_matrix, aes(x = Variable1, y = Variable2, fill = value)) +
  geom_tile(color = "white") +  # Use white to separate the tiles
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +
  theme_minimal() +
  labs(title = "Correlation Matrix of Diamond Attributes", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text(angle = 45, hjust = 1))
