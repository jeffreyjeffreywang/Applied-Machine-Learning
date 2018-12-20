import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='hw3-data/iris.csv', type=str, help='data for calculating mean and covmat')
parser.add_argument('--data_file_noisy', default='hw3-data/dataV.csv', type=str, help='data for resconstruction')
parser.add_argument('--save_reconstruct', default=False, type=bool)
args = parser.parse_args()

data = pd.read_csv(args.data_file)
data = pd.DataFrame.as_matrix(data)
data_noisy = pd.read_csv(args.data_file_noisy)
data_noisy = pd.DataFrame.as_matrix(data_noisy)


num_principal_components = list(range(5))
# num_principal_components = [2] # For reconstructing dataII with two principle components
for n_components in num_principal_components:
    pca_noiseless = PCA(n_components=n_components)
    pca_noiseless.fit(data)
    recon_noiseless = pca_noiseless.inverse_transform(pca_noiseless.transform(data_noisy))
    mse_noiseless = mean_squared_error(data, recon_noiseless)

    pca_noise = PCA(n_components=n_components)
    pca_noise.fit(data_noisy)
    recon_noise = pca_noise.inverse_transform(pca_noise.transform(data_noisy))
    if (args.save_reconstruct):
        pd.DataFrame(recon_noise).to_csv('yuchecw2-recon.csv')
    mse_noise = mean_squared_error(data, recon_noise)
    print('n_components: {}, mse_noiseless: {}, mse_noise: {}'.format(n_components, mse_noiseless, mse_noise))
