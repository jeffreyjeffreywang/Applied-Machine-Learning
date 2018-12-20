# Read CSV into R
# Part 1A
data = read.csv(file="Desktop/pima-indians-diabetes.csv")
accuracy_vec = seq(1, 10, by=1)
train_ratio = 0.8
attribute = 1
for (iter in seq(1, 10, by=1)) {
  # Test-train spilts
  train_test_list = test_train_split(data, train_ratio, iter)
  train_data = train_test_list[[1]]
  test_data = train_test_list[[2]]
  # Calculate prior probabilities
  outcome = train_data[,9]
  n_positive = length(outcome[outcome==1])
  n_negative = length(outcome[outcome==0])
  P_positive = n_positive/length(outcome)
  P_negative = n_negative/length(outcome)
  # Calculate likelihood
  feature = train_data[,1]
  positive_feature_mean = mean(feature[outcome==1])
  positive_feature_var = var(feature[outcome==1])
  negative_feature_mean = mean(feature[outcome==0])
  negative_feature_var = var(feature[outcome==0])
  confusion_matrix = estimate(test_data, positive_feature_mean, positive_feature_var, 
                              negative_feature_mean, negative_feature_var, P_positive, P_negative)
  n_correct = confusion_matrix[1,1] + confusion_matrix[2,2]
  accuracy = n_correct / nrow(test_data)
  accuracy_vec[iter] = accuracy
}
print(accuracy_vec)

# Evaluate on test data and return the confusion matrix
estimate = function(test_data, positive_mean, positive_var, negative_mean, negative_var, P_positive, P_negative) {
  true_positive = 0
  false_positive = 0
  true_negative = 0
  false_negative = 0
  logP_positive = log(P_positive)
  logP_negative = log(P_negative)
  for(index in seq(1, nrow(test_data), by=1)) {
    positive_log_p = evaluate_log_p(test_data[index,1], positive_mean, positive_var)
    negative_log_p = evaluate_log_p(test_data[index,1], negative_mean, negative_var)
    if (logP_positive + positive_log_p > logP_negative + negative_log_p) {
      if (test_data[index, 9] == 1) {
        true_positive = true_positive + 1
      } else {
        false_positive = false_positive + 1
      }
    } else {
      if (test_data[index, 9] == 0) {
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
  return(confusion_matrix)
}

# Split the data
test_train_split = function(data, train_ratio, seed) {
  set.seed(seed)
  sample = sample(seq_len(nrow(data)), size=floor(train_ratio*nrow(data)))
  train_data = data[sample,]
  test_data = data[-sample,]
  return(list(train_data, test_data))
}

# Evalute log probability
evaluate_log_p = function(feature, feature_mean, feature_var) {
  log_p = - ((feature - feature_mean)^2)/(2*feature_var) 
  log_p = log_p + log(1/sqrt(2*pi*feature_var))
  return(log_p)
}
