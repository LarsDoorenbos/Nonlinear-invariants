
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_df(class_label):
    if class_label == 0:
        df = pd.read_csv('data/annthyroid-unsupervised-ad.csv')
        LOGGER.info('thyroid')
    elif class_label == 1:
        df = pd.read_csv('data/breast-cancer-unsupervised-ad.csv')
        LOGGER.info('breast cancer')
    elif class_label == 2:
        df = pd.read_csv('data/speech-unsupervised-ad.csv')
        LOGGER.info('speech')
    elif class_label == 3:
        df = pd.read_csv('data/pen-global-unsupervised-ad.csv')
        LOGGER.info('pen global')
    elif class_label == 4:
        df = pd.read_csv('data/shuttle-unsupervised-ad.csv')
        LOGGER.info('Shuttle')
    elif class_label == 5:
        df = pd.read_csv('data/kdd99-unsupervised-ad.csv')
        LOGGER.info('kdd99')
    return df


def training_dataset(class_label, preprocessing, arch, path):
    np.random.seed(1)

    df = get_df(class_label)

    outliers = df[df['y'] == 'o']
    inliers = df[df['y'] == 'n']

    outliers = outliers.drop('y', axis=1).to_numpy().astype(np.float32)
    inliers = inliers.drop('y', axis=1).to_numpy().astype(np.float32)

    subsample_size = len(inliers) - len(outliers)
    ids = np.arange(len(inliers))
    np.random.shuffle(ids)

    data_x = inliers[ids[:subsample_size - len(outliers)]]

    if preprocessing == 'normalize_last':
        normed_data_x = data_x / np.linalg.norm(data_x, axis=1, keepdims=True)
        normed_data_x = normed_data_x - np.mean(normed_data_x, axis=0)

        data_x = data_x - np.mean(data_x, axis=0)
        return {'0': normed_data_x, '1': data_x}

    return {'0': data_x}


def test_dataset(class_label, preprocessing, arch, path):
    np.random.seed(1)

    df = get_df(class_label)
    
    outliers = df[df['y'] == 'o']
    inliers = df[df['y'] == 'n']

    outliers = outliers.drop('y', axis=1).to_numpy().astype(np.float32)
    inliers = inliers.drop('y', axis=1).to_numpy().astype(np.float32)

    subsample_size = len(inliers) - len(outliers)
    ids = np.arange(len(inliers))
    np.random.shuffle(ids)

    inliers = inliers[ids[subsample_size:]]

    if preprocessing == 'normalize_last':
        normed_inliers = inliers / np.linalg.norm(inliers, axis=1, keepdims=True)
        normed_inliers = normed_inliers - np.mean(normed_inliers, axis=0)

        inliers = inliers - np.mean(inliers, axis=0)

        normed_outliers = outliers / np.linalg.norm(outliers, axis=1, keepdims=True)
        normed_outliers = normed_outliers - np.mean(normed_outliers, axis=0)

        outliers = outliers - np.mean(outliers, axis=0)
        
        return {'0': normed_inliers, '1': inliers}, {'0': normed_outliers, '1': outliers}
    
    return {'0': inliers}, {'0': outliers}
