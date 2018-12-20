import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import random
import math

def preprocess(data, test):
    box_size = 20
    original_size = 28
    if not test:
        streched_data = np.empty((data.shape[0], box_size*box_size+1))
    else:
        # No label column
        streched_data = np.empty((data.shape[0], box_size*box_size))
    for k in range(data.shape[0]):
        # Label
        if not test:
            streched_data[k, 0] = data[k, 0]
        # Find the margin
        bottom = find_bottom(data, k, original_size)
        top = find_top(data, k, original_size)
        num_rows = bottom-top+1
        right = find_right(data, k, original_size, test)
        left = find_left(data, k, original_size, test)
        num_cols = right-left+1
        # Find the bounding box
        bounding_box = np.empty((num_rows, num_cols))
        for j in range(top, bottom+1):
            for i in range(left, right+1):
                if not test:
                    bounding_box[j-top, i-left] = data[k, original_size*j+i+1]
                else:
                    bounding_box[j-top, i-left] = data[k, original_size*j+i]
        # plt.subplot(1,2,1)
        # plt.matshow(bounding_box)
        # plt.title("#%d bounding box"%(k))
        # Resize the bounding box
        streched_bounding_box = resize_matrix(bounding_box, box_size, box_size)
        for i in range(box_size):
            for j in range(box_size):
                if not test:
                    streched_data[k, box_size*i+j+1] = streched_bounding_box[i, j]
                else:
                    streched_data[k, box_size*i+j] = streched_bounding_box[i,j]
    return streched_data

def resize_matrix(mat, n_row_out, n_col_out):
    n_row_in = mat.shape[0]
    n_col_in = mat.shape[1]
    mat_out = np.empty((n_row_out, n_col_out))
    row_ratio = n_row_in/n_row_out
    col_ratio = n_col_in/n_col_out
    for i in range(n_row_out):
        for j in range(n_col_out):
            mat_out[i, j] = mat[math.floor(i*row_ratio), math.floor(j*col_ratio)]
    return mat_out

def find_top(data, num, original_size):
    for i in range(1, len(data[num, :])):
        if data[num, i].item() is not 0:
            return math.floor(i/original_size)

def find_bottom(data, num, original_size):
    for i in range(len(data[num, :]) - 1, 0, -1):
        if data[num, i].item() is not 0:
            return math.floor(i/original_size)

def find_left(data, num, original_size, test):
    for left in range(original_size):
        for j in range(original_size):
            if not test:
                if data[num, original_size*j+left+1].item() is not 0:
                    return left
            else:
                if data[num, original_size*j+left].item() is not 0:
                    return left

def find_right(data, num, original_size, test):
    for right in range(original_size-1,-1,-1):
        for j in range(original_size):
            if not test:
                if data[num, original_size*j+right+1].item() is not 0:
                    return right
            else:
                if data[num, original_size*j+right].item() is not 0:
                    return right
