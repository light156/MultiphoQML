import pandas as pd
import numpy as np


def make_vowel_dataset():
    df = pd.read_excel('metric_learning/dataset/vowel_data.xlsx').to_numpy()
    data = np.array(df[:, 1:], dtype=np.int_)
    labels = np.repeat(np.arange(0, 7), 37)
    np.save('metric_learning/data/vowel_data.npy', {'data': data, 'labels': labels})


def vowel_dataset(train_ratio=0.7):
    vowel_dataset = np.load('metric_learning/dataset/vowel_data.npy', allow_pickle=True).item()
    x, y = vowel_dataset['data'], vowel_dataset['labels']

    # x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True))
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    train_num = round(37 * train_ratio)
    train_idx = [i for i in range(259) if i % 37 < train_num]
    test_idx = [i for i in range(259) if i % 37 >= train_num]
    
    return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])
