from sklearn.datasets import fetch_mldata
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def mnist_data_loader(batch_size, homedir):
    # 1. データの前処理
    # print("Step 1: preprocessing data")
    mnist = fetch_mldata('MNIST original', data_home=homedir + 'data/')
    X = mnist.data / 255
    y = mnist.target

    # 2. DataLoaderの作成
    # 2.1 データを訓練とテストに分割（6:1）
    # print("Step 2: split train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=9)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=19)

    # 2.2 データをPyTorchのTensorに変換
    X_train = torch.Tensor(X_train)
    X_valid = torch.Tensor(X_valid)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_valid = torch.LongTensor(y_valid)
    y_test = torch.LongTensor(y_test)

    # 2.3 データとラベルをセットにしたDatasetを作成
    ds_train = TensorDataset(X_train, y_train)
    ds_valid = TensorDataset(X_valid, y_valid)
    ds_test = TensorDataset(X_test, y_test)

    # 2.4 データセットのミニバッチサイズを指定した，Dataloaderを作成
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return loader_train, loader_valid, loader_test
