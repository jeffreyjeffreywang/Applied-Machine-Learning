# Part 1d
library(caret)
library(e1071)
library(klaR)
# Read CSV into R
data = read.csv(file="pima-indians-diabetes.csv", header=TRUE, sep=",")
attributes = c(3, 4, 6, 8) #blood pressure, skin thickness, BMI, age
accuracy_vec = seq(1, 10, by=1)
for(seed in seq(1, 10)) {
  # Test train split
  set.seed(seed)
  in_train = createDataPartition(data[,9], p=0.8, list=FALSE)
  train_data = data[in_train, ]
  test_data = data[-in_train, ]
  test_label = test_data[,9]
  # Train the model
  feature = train_data[, attributes]
  label = as.factor(train_data[,9])
  model = svmlight(feature, label, pathsvm="/Users/jeffreywang/Desktop/AML/svm_light")
  prediction = predict(model, test_data[, attributes])
  predict_label = prediction$class
  correct = test_label == predict_label
  
  accuracy = length(correct[correct==TRUE])/length(correct)
  accuracy_vec[seed] = accuracy
}
print(accuracy_vec)
print(mean(accuracy_vec))
