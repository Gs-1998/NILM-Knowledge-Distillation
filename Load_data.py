import numpy as np

from Preprocessing import create_house_dataframe, date
import matplotlib.pyplot as plt


def load_data(savepath="dataset",appliance="washer"):
    X_train_seg = np.load(savepath+appliance+"/X_train_seg.npy")
    y_train_seg = np.load(savepath+ appliance + "/y_train_seg.npy")
    X_test_seg = np.load(savepath + appliance + "/X_test_seg.npy")
    y_test_seg = np.load(savepath + appliance + "/y_test_seg.npy")
    Main = np.load(savepath + appliance + "/Main.npy")
    return X_train_seg,y_train_seg,X_test_seg,y_test_seg,Main

