Machine Learning for Parkinson's Disease Detection
Overview:
This project utilizes machine learning techniques to create an effective model for the detection of Parkinson's disease. The primary focus is on employing the Random Forest algorithm, a robust and accurate approach. The entire implementation is done using the R programming language.

Project Contents:
1. Load Required Libraries:

library(readr)
library(tidyverse)
library(mlbench)
library(e1071)
library(caret)
library(DataExplorer)
library(MVN)
library(MASS)
options(warn=-1)
2. Loading Data:

Parkinson1 <- read_csv("Parkinson1.csv")
Parkinson3 <- read_csv("Parkinson3.csv")
Train <- as.data.frame(Parkinson1)
Test <- as.data.frame(Parkinson3)
3. Explanatory Data Analysis:
Check datatypes, missing values, and summary statistics.
Explore the balance between populations and examine variables like Gender.
Visualize variable distributions and correlations.
Check for normality using multivariate normality tests.
4. Data Preparation, Preprocessing, and Modeling:
Data Cleaning:

# Remove columns with near zero variance
varParams <- preProcess(Train[, -2], method=c("nzv"))
print(varParams)
Feature Selection:

# Dropping id column
Train <- Train[,2:47]
# Remove highly correlated independent variables
Train<- Train[, -c(findCorrelation(cor(Train), cutoff=0.9))]
Data Transforms:

# Convert dependent variable to factor
Train$Status = as.factor(Train$Status)
# Scale dataset
normParams <- preProcess(Train[, -1], method=c("range"))
Train[, -1] <- predict(normParams, Train[, -1])
Modeling:
# Prepare resampling method
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
# Logistic Regression
fit.lr <- train(Status ~ ., data=Train, method="glm", metric=metric, trControl=trainControl)
# Random Forest
fit.rf <- train(Status ~ ., data=Train, method="rf", metric=metric, trControl=trainControl)
# CART
fit.cart <- train(Status ~ ., data=Train, method="rpart", metric=metric, trControl=trainControl)
# Collect resamples
results <- resamples(list(LR=fit.lr, CART=fit.cart, RF=fit.rf))
Results:
The Random Forest model demonstrates the highest mean accuracy among the models evaluated.
For more details on data exploration, preprocessing, and code implementation, refer to the provided R script. Feel free to explore, replicate, and contribute to the project. If you have any questions or need assistance, contact [Your Name/Email].





