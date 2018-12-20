library(caret)
library(e1071)
library(klaR)
# Read CSV into R
data = read.csv(file="Desktop/AML/pima-indians-diabetes.csv", header=TRUE, sep=",")
accuracy_vec = seq(1, 10, by=1)
for(seed in seq(1, 10)) {
  # Test train split
  set.seed(seed)
  in_train = createDataPartition(data[,9], p=0.8, list=FALSE)
  train_data = data[in_train, ]
  test_data = data[-in_train, ]
  # Train the model
  feature = train_data[,-9]
  label = as.factor(train_data[,9])
  model = train(feature, label, 'nb', trControl=trainControl(method='cv', number=10))
  prediction = predict(model$finalModel, test_data[,-9])
  label_prediction = prediction$class
  # Calculate confusion matrix
  true_positive = 0
  false_positive = 0
  true_negative = 0
  false_negative = 0
  for(i in seq(1, nrow(test_data))) {
    if (label_prediction[i] == "Yes") {
      if (test_data[i,9] == 1) {
        true_positive = true_positive + 1
      } else {
        false_positive = false_positive + 1
      }
    } else {
      if (test_data[i, 9] == 0) {
        true_negative = true_negative + 1
      } else {
        false_negative = false_negative + 1
      }
    } 
  }
  confusion_matrix = matrix(
    c(true_positive, false_positive, false_negative, true_negative),
    nrow = 2,
    ncol = 2,
    byrow = TRUE
  )
  # print(confusion_matrix)
  n_correct = true_positive + true_negative
  accuracy = n_correct/nrow(test_data)
  accuracy_vec[seed] = accuracy
}
print(accuracy_vec)
print(mean(accuracy_vec))
