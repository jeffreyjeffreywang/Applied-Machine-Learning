import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import standardize
from sklearn.cross_validation import train_test_split
import random
import argparse
from process import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--num_steps', default=300, type=int)
parser.add_argument('--eval_steps', default=30, type=int, help='compute the accuracy of the current classifier on the set held out for the epoch every 30 steps')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--m', default=1, type=float, help='parameter for learning rate')
parser.add_argument('--n', default=0.1, type=float, help='parameter for learning rate')
parser.add_argument('--lamda', default=0.001, type=float, help='regularization parameter')
parser.add_argument('--num_held_out', default=50, type=int)
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--path', default='yuchecw2_prediction.csv', type=str)
args = parser.parse_args()

train_data = pd.read_csv("train.data.csv")
label_list = [i for i in train_data['class']]
label = np.empty(len(label_list))
for i in range(label.shape[0]):
    if (label_list[i] == ' >50K'):
        label[i] = 1
    else:
        label[i] = -1

feature = pd.DataFrame.as_matrix(train_data)[:, :-1]
# Preprocess the data
feature = preprocess(feature)
# Standardize columns in train_feature
feature = standardize(feature)
# Train-validation split
train_feature, val_feature, train_label, val_label = train_test_split(
                feature, label, test_size=0.1)

plt.figure()
for lamda in [0.001, 0.01, 0.1, 1]:
    # Initialize a, b
    a = np.random.randn(train_feature.shape[1], 1)  # shape=(14, 1)
    b = 0
    accuracy_list = []
    mag_list = []
    loss_list = []
    for epoch in range(args.num_epochs):
        actual_train_feature, held_out_feature, actual_train_label, held_out_label = train_test_split(train_feature, train_label, test_size=args.num_held_out/train_feature.shape[0], random_state=epoch)
        actual_train_size = actual_train_feature.shape[0]
        for step in range(args.num_steps):
            lr = compute_lr(args.m, args.n, epoch)
            batch_num = random.sample(range(actual_train_size), args.batch_size)
            a = update_a(a, b, actual_train_feature, actual_train_label, batch_num, lr, lamda)
            b = update_b(b, a, actual_train_feature, actual_train_label, batch_num, lr, lamda)
            if (step%args.eval_steps == 0):
                eval_dict = evaluate(held_out_feature, held_out_label, a, b, lamda)
                accuracy_list.append(eval_dict['accuracy'])
                mag_list.append(eval_dict['mag'])
                loss_list.append(eval_dict['loss'])

    if (args.test):
        test_data = pd.read_csv("test.data.csv")
        test_feature = pd.DataFrame.as_matrix(test_data)
        test_feature = preprocess(test_feature)
        test_feature = standardize(test_feature)
        prediction = predict(test_feature, a, b)
        prediction[prediction['Label'] == '>'] = ">50K"
        prediction[prediction['Label'] == '<'] = "<=50K"
        prediction.to_csv(args.path)

    step_list = range(len(accuracy_list))

    for step in step_list:
        step *= args.num_steps


    label = 'λ = {}'.format(lamda)
    plt.plot(step_list, mag_list, label=label, linewidth=0.8)
    plt.ylim(0, 100)

plt.xlabel('step number')
plt.ylabel('magnitude')
plt.title('Magnitude vs step number', fontsize=12, fontweight='bold')
plt.legend(loc='lower right')
plt.show()
