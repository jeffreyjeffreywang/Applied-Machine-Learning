setwd('/Users/jeffreywang/Desktop/AML/HW7')
library(gdata)
data <- read.csv('blogData_train.csv')
library(glmnet)
library("rlist")
library(data.table)
xmat = as.matrix(data[,-281])
ymat = as.matrix(data[,281])
# Poisson generalized model with lasso
model <- cv.glmnet(xmat, ymat, family='poisson',nfolds=10, alpha=1)
# A plot of cross-validated deviance against the regularization variable
plot(model)
# A scatter plot of true values vs predicted values for training data
predicted = predict(model, newx=xmat, s="lambda.1se")
plot(predicted, ymat, main="Training data\nPredicted v.s. True", xlab="Predicted value", ylab="True value")
reg <- lm(ymat~predicted) # R squared = 0.1398
abline(reg, col='red')
# A scatter plot of true values vs predicted values for testing data
test_data_files = list.files(path="BlogFeedback")
# Load test data from csv files
test_data <- read.csv(paste("BlogFeedback/", test_data_files[1], sep=""))
for (i in 2:60) {
  test_data = rbindlist(list(test_data, read.csv(paste("BlogFeedback/", test_data_files[i], sep=""))))
}
# Split test data to x_test and y_test
par(mfrow=c(1,1))
x_test = as.matrix(test_data[,-281])
y_test = as.matrix(test_data[,281])
predicted_test = predict(model, x_test, s="lambda.1se")
plot(predicted_test, y_test, main="Testing data\nPredicted v.s. True", xlab="Predicted value", ylab="True value")
reg_test <- lm(y_test~predicted_test)
abline(reg_test, col='red')

