Part 1
Run hw1a.R, hw1b.R, hw1d.R by
$ Rscript hw1a.R
to obtain the accuracy of each prediction

Part2
Run hw2a.py by
$ python hw2a.py
To change models and preprocess images, specify it by adding command line arguments.
Options:
--model_name   ['Gaussian', 'Bernoulli', 'Random Forest'] 	Select a model for training.
--preprocess   bool						Preprocess the image by finding its bounding box and crop it.
--num_epochs   int						Number of epochs where each epoch has different train-validatio splits.
--test	       bool						Test or not. If true, prediction labels will be written into a csv file and mean images will be shown.
--path         str						Specify the path where the csv file containing prediction labels is stored.
--n_estimators int						Number of estimators in the random forest model.
--max_depth    int						The max depth for each tree in the random forest model.

