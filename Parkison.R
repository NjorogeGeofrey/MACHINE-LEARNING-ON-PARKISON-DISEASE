#Load required libraries
library(readr)
library(tidyverse) # metapackage of all tidyverse packages
library(mlbench) # Data modelling
library(e1071) # Naive Bayes
library(caret) # Data preprocessing, modelling
library(DataExplorer) # Visualization
library(MVN)
library(MASS)
# Hide warnings
options(warn=-1) 


#Loading the data
Parkinson1 <- read_csv("Parkinson1.csv")
Parkinson3 <- read_csv("Parkinson3.csv")
#Naming the data
Train <- as.data.frame(Parkinson1)
Test <- as.data.frame(Parkinson3)

######################Explanatory Data Analysis#################
#Explanatory data analysis
# Check datatypes
sapply(Train, class)
#The output variable is of data type integer, which needs to be converted to factor, 
#considering this is a classification problem.

#Checking for missing values
sum(is.na(Train))


# Summary statistics
summary(Train)

# Explore balance between populations
y <- Train$Status
cbind(freq=table(y), percentage=prop.table(table(y))*100)
#The output shows that the populations are complettly balanced

# Explore status among Gender.
prop.table(table(Train$Gender, Train$Status))*100
#More men were invloved compared to females

# Check standard deviations
sapply(Train, sd)


#Correlation test
cor(Train[,2:47])


#Multiple predictor pairs appear to have very high positive correlation.
#Namely Jitter_rel & Jitter_RAP,  Jitter_PPQ & Shim_loc , etc.
#It might be a good idea to limit autocorrelating features during feature selectio

#Data visualization.
# Visualize distribution of variables
# Histograms of each attribute from 4 to 47

for(i in 4:47) {
  print(ggplot(Train, aes(x=Train[,i])) + geom_histogram(bins=30, aes(y=..density..), colour="black", fill="white")
        + geom_density(alpha=.2, fill="#FF6666"))
}

# Boxplots for each attribute
par(mfrow=c(3,3))
for(i in 4:47) {
  boxplot(Train[,i], main=names(Train)[i])
}

#Correlation  Plot
plot_correlation(Train)

############CHECkING FOR NORMALITY############

# Check multivariate normality
mvn(Train[, 4:47], mvnTest = "hz", multivariateOutlierMethod = "quan")
#From the output we see that some variables do not have normal distribution.

###############DATA PREPARATION , PREPROCESSING AND MODELLING##########

#Data Cleaning
#We know that there are no missing values in our dataset.
#We will check variables with very low variance and consider removing 
# Remove columns with near zero variance
# Calculate preprocessing parameters from the dataset
varParams <- preProcess(Train[, -2], method=c("nzv"))
# Summarize preprocess parameters
print(varParams)
#Since all variables were ignored, we can conclude that all variables have acceptable variance.

#Feature Selection
# Dropping id column, as it holds no weight as a predictor
Train <- Train[,2:47]

# Check if id column has been dropped
head(Train, 2)

# Find highly correlated independent variables
print(findCorrelation(cor(Train), cutoff=0.9))

# Remove highly correlated independent variable
Train<- Train[, -c(findCorrelation(cor(Train), cutoff=0.9))]

# Dataset without high correlation between independent variables
head(Train, 5)

#We saw in the data summarization step, that multiple predictors were highly correlated.
#Above function finds those predictors and we promptly remove them.

###Data Transforms
# For classification, the dependent variable should be of class factor
Train$Status = as.factor(Train$Status)

# Recheck classes of all variables
sapply(Train, class)

####
# Scale dataset or Nomalizing

# Calculate preprocessing parameters from the dataset
normParams <- preProcess(Train[, -1], method=c("range"))
# Summarize preprocess parameters
print(normParams)
# Transform the dataset using above parameters
Train[, -1] <- predict(normParams, Train[, -1])
summary(Train)
#During our EDA, we had found that all variables had a different scale.
#Rescaling them to a common scale might increase performance 
#for instance based and weight based algorithms.

##Modellind
# Prepare resampling method
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
#
# Discriminant analysis
lda_model <- lda(Status ~ ., data = Train)
lda_model
#Classification
# Logistic Regression for classification
set.seed(7)
fit.lr <- train(Status ~ ., data=Train, method="glm", metric=metric, trControl=trainControl)
# Random Forest for classification
set.seed(7)
fit.rf <- train(Status ~ ., data=Train, method="rf", metric=metric, trControl=trainControl)
## CART
set.seed(7)
fit.cart <- train(Status ~ ., data=Train, method="rpart", metric=metric, trControl=trainControl)
# Collect resamples
results <- resamples(list(LR=fit.lr, CART=fit.cart, RF=fit.rf))
summary(results)
dotplot(results)
##
#From the summary and dotplot, we can see that Random Forest  has the best mean accuracy



