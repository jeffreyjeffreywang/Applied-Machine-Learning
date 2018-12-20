import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random
import math
import argparse
from helper import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='Gaussian', type=str)
parser.add_argument('--preprocess', default=False, type=bool)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--test', default=True, type=bool)
parser.add_argument('--path', default='yuchecw2_1.csv', type=str)
parser.add_argument('--n_estimators', default=30, type=int)
parser.add_argument('--max_depth', default=16, type=int)
args = parser.parse_args()

train_data = pd.read_csv("MNIST/train.csv")
test_data = pd.read_csv("MNIST/test.csv")
train_data = pd.DataFrame.as_matrix(train_data)
test_data = pd.DataFrame.as_matrix(test_data)
if args.preprocess:
    train_data = preprocess(train_data, False)
if not args.test:
    ratio = 0.2
    num_data = train_data.shape[0]
    num_val = (int)(num_data*ratio)
    accuracy_list = []

    for epoch in range(args.num_epochs):
        random.seed(epoch)
        val_samples = random.sample(range(num_data), num_val)
        mask_array = np.ones((num_data, 1), dtype=bool)
        for i in val_samples:
            mask_array[i] = False

        label = np.empty((num_data-num_val, 1))
        feature = np.empty((num_data-num_val, train_data.shape[1]-1))
        idx = 0
        for i in range(num_data):
            if (mask_array[i]):
                label[idx] = train_data[i,0]
                feature[idx, :] = train_data[i,1:]
                idx = idx + 1

        val_label = train_data[val_samples, 0]
        val_feature = train_data[val_samples, 1:]

        # Create a model and train it
        if args.model_name == 'Gaussian':
            model = GaussianNB()
        elif args.model_name == 'Bernoulli':
            # Thresholding
            feature[feature>127] = 255
            feature[feature<=127] = 0
            val_feature[val_feature>127] = 255
            val_feature[val_feature<=127] = 0
            model = BernoulliNB()
        elif args.model_name == 'Random_Forest':
            model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)

        label = np.ravel(label)
        fit_model = model.fit(feature, label)
        label_prediction = fit_model.predict(val_feature)
        num_correct = (val_label == label_prediction).sum()
        accuracy = num_correct/num_val
        accuracy_list.append(accuracy)
    print('Training and cross validation: uses %s model, preprocessed: %d, accuracy: %f'%(args.model_name, args.preprocess, sum(accuracy_list)/len(accuracy_list)))
else:
    label = train_data[:,0]
    feature = train_data[:,1:]
    # Create a model and train it
    if args.model_name == 'Gaussian':
        model = GaussianNB()
    elif args.model_name == 'Bernoulli':
        feature[feature>127] = 255
        feature[feature<=127] = 0
        model = BernoulliNB()
    elif args.model_name == 'Random_Forest':
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)

    label = np.ravel(label)
    fit_model = model.fit(feature, label)

    if args.preprocess:
        test_data = preprocess(test_data, True)
    test_prediction = fit_model.predict(test_data)
    df = pd.DataFrame(test_prediction)
    df_test_data = pd.DataFrame(test_data)
    df.to_csv(args.path)

    # Calculate class means
    if args.preprocess:
        original_size = 20 # box_size
    else:
        original_size = 28

    num_classes = 10
    mean = [None]*num_classes
    for label in range(num_classes):
        mean[label] = df_test_data[df[0]==label].mean()

    mean_matrix = []
    for label in range(num_classes):
        a = np.empty((original_size, original_size))
        for j in range(original_size):
            for i in range(original_size):
                if (mean[label][original_size*j+i] < 127):
                    a[j, i] = 255
                else:
                    a[j, i] = 0
        mean_matrix.append(a)

    plt.figure()
    for label in range(10):
        plt.subplot(1, 10, label+1)
        plt.imshow(mean_matrix[label], cmap='Greys')
        plt.axis('off')
    plt.show()
