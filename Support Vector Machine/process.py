import numpy as np
import pandas as pd
import math

# Input: feature matrix of shape (43958, 14)
def preprocess(train_feature):
    workclass = {' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc': 2, ' Federal-gov': 3,
                ' Local-gov': 4, ' State-gov': 5, ' Without-pay': 6, ' Never-worked': 7};
    education = {' Bachelors': 0, ' Some-college': 1, ' 11th': 2, ' HS-grad': 3, ' Prof-school': 4,
            ' Assoc-acdm': 5, ' Assoc-voc': 6, ' 9th': 7, ' 7th-8th': 8, ' 12th': 9, ' Masters': 10,
            ' 1st-4th': 11, ' 10th': 12, ' Doctorate': 13, ' 5th-6th': 14, ' Preschool': 15};
    martial_status = {' Married-civ-spouse': 0, ' Divorced': 1, ' Never-married': 2, ' Separated': 3,
                    ' Widowed': 4, ' Married-spouse-absent': 5, ' Married-AF-spouse': 6};
    occupation = {' Tech-support': 0, ' Craft-repair': 1, ' Other-service': 2, ' Sales': 3, ' Exec-managerial': 4,
            ' Prof-specialty': 5, ' Handlers-cleaners': 6, ' Machine-op-inspct': 7, ' Adm-clerical': 8,
            ' Farming-fishing': 9, ' Transport-moving': 10, ' Priv-house-serv': 11, ' Protective-serv': 12,
            ' Armed-Forces': 13};
    relationship = { ' Wife': 0, ' Own-child': 1, ' Husband': 2, ' Not-in-family': 3, ' Other-relative': 4,
                ' Unmarried': 5};
    race = {' White': 0, ' Asian-Pac-Islander': 1, ' Amer-Indian-Eskimo': 2, ' Other': 3, ' Black': 4};
    sex = {' Female': 0, ' Male': 1};
    native_country ={' United-States': 0, ' Cambodia': 1, ' England': 2, ' Puerto-Rico': 3, ' Canada': 4,
                ' Germany': 5, ' Outlying-US(Guam-USVI-etc)': 6, ' India': 7, ' Japan': 8,
                ' Greece': 9, ' South': 10, ' China': 11, ' Cuba': 12, ' Iran': 13, ' Honduras': 14,
                ' Philippines': 15, ' Italy': 16, ' Poland': 17, ' Jamaica': 18, ' Vietnam': 19,
                ' Mexico': 20, ' Portugal': 21, ' Ireland': 22, ' France': 23, ' Dominican-Republic': 24,
                ' Laos': 25, ' Ecuador': 26, ' Taiwan': 27, ' Haiti': 28, ' Columbia': 29, ' Hungary': 30,
                ' Guatemala': 31, ' Nicaragua': 32, ' Scotland': 33, ' Thailand': 34, ' Yugoslavia': 35,
                ' El-Salvador': 36, ' Trinadad&Tobago': 37, ' Peru': 38, ' Hong': 39, ' Holand-Netherlands': 40};
    missing_workclass = []
    has_workclass = []
    missing_education = []
    has_education = []
    missing_martial_status = []
    has_martial_status = []
    missing_martial_status = []
    has_martial_status = []
    missing_occupation = []
    has_occupation = []
    missing_relationship = []
    has_relationship = []
    missing_race = []
    has_race = []
    missing_sex = []
    has_sex = []
    missing_native_country = []
    has_native_country = []

    for i in range(train_feature.shape[0]):
        # workclass
        if not train_feature[i, 1] == ' ?':
            train_feature[i, 1] = workclass[train_feature[i, 1]]
            has_workclass.append(i)
        else:
            missing_workclass.append(i)
        # education
        if not train_feature[i, 3] == ' ?':
            train_feature[i, 3] = education[train_feature[i, 3]]
            has_education.append(i)
        else:
            missing_education.append(i)
        # martial_status
        if not train_feature[i, 5] == ' ?':
            train_feature[i, 5] = martial_status[train_feature[i, 5]]
            has_martial_status.append(i)
        else:
            missing_martial_status.append(i)
        # occupation
        if not train_feature[i, 6] == ' ?':
            train_feature[i, 6] = occupation[train_feature[i, 6]]
            has_martial_status.append(i)
        else:
            missing_occupation.append(i)
        # relationship
        if not train_feature[i, 7] == ' ?':
            train_feature[i, 7] = relationship[train_feature[i, 7]]
            has_relationship.append(i)
        else:
            missing_relationship.append(i)
        # race
        if not train_feature[i, 8] == ' ?':
            train_feature[i, 8] = race[train_feature[i, 8]]
            has_race.append(i)
        else:
            missing_race.append(i)
        # sex
        if not train_feature[i, 9] == ' ?':
            train_feature[i, 9] = sex[train_feature[i, 9]]
            has_sex.append(i)
        else:
            missing_sex.append(i)
        # native_country
        if not train_feature[i, 13] == ' ?':
            train_feature[i, 13] = native_country[train_feature[i, 13]]
            has_native_country.append(i)
        else:
            missing_native_country.append(i)

    substitue_val = {}
    substitue_val['workclass'] = round(train_feature[:, 1][has_workclass].mean())
    for j in missing_workclass:
        train_feature[j,1] = substitue_val['workclass']
    substitue_val['education'] = round(train_feature[:, 3][has_education].mean())
    for j in missing_education:
        train_feature[j,3] = substitue_val['education']
    substitue_val['martial_status'] = round(train_feature[:, 5][has_martial_status].mean())
    for j in missing_martial_status:
        train_feature[j,5] = substitue_val['martial_statu']
    substitue_val['occupation'] = round(train_feature[:, 6][has_occupation].mean())
    for j in missing_occupation:
        train_feature[j,6] = substitue_val['occupation']
    substitue_val['relationship'] = round(train_feature[:, 7][has_relationship].mean())
    for j in missing_relationship:
        train_feature[j,7] = substitue_val['relationship']
    substitue_val['race'] = round(train_feature[:, 8][has_race].mean())
    for j in missing_race:
        train_feature[j,8] = substitue_val['race']
    substitue_val['sex'] = round(train_feature[:, 9][has_sex].mean())
    for j in missing_sex:
        train_feature[j,9] = substitue_val['sex']
    substitue_val['native_country'] = round(train_feature[:, 13][has_native_country].mean())
    for j in missing_native_country:
        train_feature[j,13] = substitue_val['native_country']

    return train_feature

# Input
# a: numpy array of size (14, 1)
# feature: numpy array of size (14, 1)
# b: double
# Output
# gamma: double
def compute_gamma(a, feature, b):
    gamma = np.matmul(np.transpose(a), feature).item() + b
    return gamma

# Input
# gamma: numpy array of size (14, 1)
def cost_function(gamma, label):
    cost = max(0, 1-label*gamma)
    return cost

# Input
# batch_num: a list consists of indices of training feature, supposed to be length of 1
# train_feature: numpy array of shape (43958, 14)
# train_label: numpy array of shape (43958, )
def hinge_loss(batch_num, train_feature, train_label, a, b):
    loss = 0
    for i in batch_num:
        feature_i = np.expand_dims(train_feature[i,:], axis=1)  # shape=(14,1)
        gamma = compute_gamma(a, feature_i, b)
        loss += cost_function(gamma, train_label[i])
    loss /= len(batch_num)
    return loss

# Input
# lamda: regularization parameter
# a: numpy array of shape (14, 1)
def regularization_loss(lamda, a):
    return 0.5*lamda*(np.matmul(np.transpose(a), a).item())

# This is the loss we're going to minimize using stochastic gradient descent
def total_loss(batch_num, train_feature, train_label, a, b, lamda):
    return hinge_loss(batch_num, train_feature, train_label, a, b) + regularization_loss(lamda, a)

# Output
# u: numpy array of shape (15, 1)
def obtain_u(a, b):
    b_arr = np.expand_dims(np.array([b]), axis=1)
    u = np.concatenate((a, b_arr), axis=0)
    return u

def compute_lr(m, n, epoch):
    return m/(n+epoch)

# Input
# current_a: numpy array of shape (14, 1)
# current_b: double
def update_a(current_a, current_b, train_feature, train_label, batch_num, lr, lamda):
    grad = np.zeros(current_a.shape)
    for i in batch_num:
        if (cost_function(compute_gamma(current_a, train_feature[i,:], current_b), train_label[i]) == 0):
            grad += lamda*current_a
        else:
            grad = grad + (lamda*current_a - np.expand_dims(train_label[i]*train_feature[i, :], axis=1))
    grad /= len(batch_num)
    return current_a - lr*grad

def update_b(current_b, current_a, train_feature, train_label, batch_num, lr, lamda):
    grad = 0
    for i in batch_num:
        if (cost_function(compute_gamma(current_a, train_feature[i, :], current_b), train_label[i]) == 0):
            continue
        else:
            grad += (-train_label[i])
    grad /= len(batch_num)
    return current_b - lr*grad

def evaluate(held_out_feature, held_out_label, a, b, lamda):
    n_correct = 0
    for i in range(held_out_feature.shape[0]):
        gamma_i = compute_gamma(a, held_out_feature[i,:], b)
        if gamma_i*held_out_label[i] > 0:
            n_correct += 1
    accuracy = n_correct / held_out_feature.shape[0]
    mag = math.sqrt(np.matmul(a.T, a).item() + b**2)
    loss = total_loss(range(held_out_feature.shape[0]), held_out_feature, held_out_label, a, b, lamda)
    return {'accuracy': accuracy, 'mag': mag, 'loss': loss}

def predict(test_feature, a, b):
    test_label = np.empty((test_feature.shape[0]), dtype=str)
    for i in range(test_feature.shape[0]):
        gamma = compute_gamma(a, np.expand_dims(test_feature[i,:], axis=1), b)
        if (gamma > 0):
            test_label[i] = ">50K"
        else:
            test_label[i] = "<=50K"
    return pd.DataFrame(data={'Label': test_label}, dtype=str)
