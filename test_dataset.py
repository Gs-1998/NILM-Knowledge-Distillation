import matplotlib.pyplot as plt
import os
import logging    # first of all import the module
import argparse


from Load_data import load_data



def remove_space(string):
    return string.replace(" ","")

def get_arguments():
    parser = argparse.ArgumentParser(description='window 50 - 100 - 200')

    parser.add_argument('--appliance',
                        type=remove_space,
                        default='dishwasher',
                        help="carico")
    parser.add_argument('--dataset_savepath',
                        type=remove_space,
                        default='/home/gstraccia/dataset/',
                        help="dataset")

    return parser.parse_args()


args = get_arguments()


X_train_seg,y_train_seg,X_test_seg,y_test_seg, X2 = load_data(savepath=args.dataset_savepath,appliance=args.appliance)


MAX_X = 2000
MAX_y = 200







plt.rcParams["figure.figsize"] = [21,9]
plt.plot((X2[8100:14100]), color = 'C0', label = 'Main value')
plt.plot((y_test_seg*MAX_y)[8000:14000], color = 'C3', label = 'True Value')

#plt.plot((X2)[70100:80100], color = 'C2', alpha = 0.6, label = 'Main Value')
plt.legend()
plt.show()

#washingmachine
#[227000:231000]
#plt.savefig("../images/"+args.appliance+"S.png")