from Preprocessing import create_house_dataframe
import os
from redd_parameters import *
import argparse
import numpy as np
import matplotlib.pyplot as plt


def remove_space(string):
    return string.replace(" ","")

def get_arguments():
    parser = argparse.ArgumentParser(description='window 50 - 100 - 200')

    parser.add_argument('--window',
                        type=int,
                        default=200,
                        help='largezza finestra')
    parser.add_argument('--path',
                        type=remove_space,
                        default='/home/gstraccia/REDD/',
                        help='path dataset')
    parser.add_argument('--savepath',
                        type=remove_space,
                        default='/home/gstraccia/',
                        help='path dove salvare il dataset')
    parser.add_argument('--appliance',
                        type=str,
                        default='washingmachine',
                        help='nome del carico microwave - fridge - dishwasher - washingmachine')


    return parser.parse_args()


args = get_arguments()

windows_length = args.window
appliance_name = args.appliance
path_dataset = args.path


def remove_abnormal_points(y, left_threshold=200, right_threshold=80):
    y_ = []
    for i, value in enumerate(y):

        if i == 0 or i == len(y) - 1:
            y_.append(y[i])
        else:
            if y[i] - y[i - 1] > left_threshold and y[i] - y[i + 1] > right_threshold:
                #                        print('index:',i)
                y_.append(y[i + 1] + 1)
            else:
                y_.append(y[i])
    return np.array(y_)


def get_differential_sequence(X):
    X_d = []
    pre_item = 0
    for item in X:
        X_d.append(item - pre_item)
        pre_item = item
    X_d = np.array(X_d)  # shape = (samples,)
    return X_d


def get_odd_data(X_o, y_o, MAX_X, MAX_y):
    plt.rcParams["figure.figsize"] = [21, 9]
    plt.plot((X_o), color='C0', label='Main value')
    plt.plot((y_o), color='C3', label='True Value')

    plt.legend()
    plt.show()
    X_o = remove_abnormal_points(X_o)
    X_o = X_o / MAX_X

    y_o = remove_abnormal_points(y_o)
    y_o = y_o / MAX_y

    X = np.expand_dims(X_o, 1)  # (samples,3)
    y = np.expand_dims(y_o, 1)  # (samples,3)
    return X, y


def shift_segment(X, y, seg_length, stride, print_info=True):
    X_o_seg = []
    y_o_seg = []


    for i in range(len(X) - seg_length + 1):
        if i % stride == 0:
            assert len(X[i:i + seg_length]) == seg_length
            X_o_seg.append(X[i:i + seg_length].reshape(-1))
            y_o_seg.append(y[i + seg_length // 2 - 1, 0])
    if print_info == True:
        print(' ' * 7, 'sequence length = {}'.format(len(X)))
        print(' ' * 7, 'windows length = {}'.format(seg_length))
        print(' ' * 7, 'stride = {}'.format(stride))
        print(' ' * 7, 'segments =', len(y_o_seg))
    # (segments,seg_length)
    return np.array(X_o_seg, dtype=np.float32), np.array(y_o_seg,dtype=np.float32)




def truncate(X_train1, y_train1, window_size):
    size = X_train1.shape[0]
    index = 1
    while (size - index) % (window_size * 2) != 0:
        size -= index
        # print(size)
    return X_train1[:size - 1, :], y_train1[:size - 1, :]



train,test = create_house_dataframe(path_dataset,params_appliance,appliance_name)

train_d=[]
test_d=[]


for t in params_appliance[appliance_name]['train_build']:
    pos = params_appliance[appliance_name]['houses'].index(t)
    channels=params_appliance[appliance_name]['channels'][pos]
    X= train[t]["main"].values.astype(np.float32)
    y= train[t][channels].values.astype(np.float32)
    train_d.append((X,y))

t_build = params_appliance[appliance_name]['test_build']

X_t = test[t_build]['main'].values.astype(np.float32)
pos = params_appliance[appliance_name]['houses'].index(t_build)
y_t = test[t_build][params_appliance[appliance_name]['channels'][pos]].values.astype(np.float32)



print('-------- Load Training Data ---------')
first = True
MAX_X = 2000
MAX_y = 200
stride = 1



for X, y in train_d:
    print(max(X))
    print(max(y))
    X_i, y_i = get_odd_data(X, y, MAX_X, MAX_y)
    X_seg_i, y_seg_i = shift_segment(X_i, y_i, windows_length, stride)

    print(X_seg_i.shape, y_seg_i.shape)

    if first == True:

        X_train = X_i  # shape=(samples,3)
        y_train = y_i  # shape=(samples,2)

        # shape=(samples,seg_length)
        X_o_train_seg = X_seg_i
        y_o_train_seg = y_seg_i

        first = False
    else:
        # shape=(samples+,3)
        X_train = np.vstack((X_train, X_i))
        y_train = np.vstack((y_train, y_i))

        # shape=(samples+,seg_length)
        X_o_train_seg = np.vstack((X_o_train_seg, X_seg_i))
        y_o_train_seg = np.hstack((y_o_train_seg, y_seg_i))



X_train_seg = X_o_train_seg
y_train_seg = y_o_train_seg

print('-------- Load Testing Data ---------')
X_i,y_i = get_odd_data(X_t, y_t, MAX_X, MAX_y)



X_seg_i,y_seg_i = shift_segment(X_i,y_i,windows_length,stride)
X_test = X_i
y_test = y_i
X_test_seg = X_seg_i
y_test_seg = y_seg_i


print('\nX_train.shape = {}'.format(X_train.shape))
print('y_train.shape = {}'.format(y_train.shape))

print('X_o_train_seg.shape = {}'.format(X_train_seg.shape))
print('y_o_train_seg.shape = {}'.format(y_train_seg.shape))

print('\nX_test.shape = {}'.format(X_test.shape))
print('y_test.shape = {}'.format(y_test.shape))

print('X_o_test_seg.shape = {}'.format(X_test_seg.shape))
print('y_o_test_seg.shape = {}'.format(y_test_seg.shape))

X_train_seg = np.expand_dims(X_train_seg,axis=2)
X_test_seg = np.expand_dims(X_test_seg,axis=2)

print('X_o_train_seg.shape = {}'.format(X_train_seg.shape))
print('y_o_train_seg.shape = {}'.format(y_train_seg.shape))

print('X_o_test_seg.shape = {}'.format(X_test_seg.shape))
print('y_o_test_seg.shape = {}'.format(y_test_seg.shape))



if not os.path.exists(args.savepath+"dataset/"):
    print("make directory " + args.savepath)
    os.makedirs(args.savepath+"dataset/")
if not os.path.exists(args.savepath+"dataset/"+args.appliance):
    print("make directory " + args.appliance)
    os.makedirs(args.savepath+"dataset/"+args.appliance)

print("save array")
np.save(args.savepath+"dataset/"+args.appliance +"/X_train_seg", X_train_seg)
np.save(args.savepath+"dataset/"+args.appliance +"/y_train_seg", y_train_seg)
np.save(args.savepath+"dataset/"+args.appliance +"/X_test_seg", X_test_seg)
np.save(args.savepath+"dataset/"+args.appliance +"/y_test_seg", y_test_seg)
np.save(args.savepath+"dataset/"+args.appliance +"/Main", X_t)
