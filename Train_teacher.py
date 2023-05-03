import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
from Load_data import load_data
from Metrics import *
from Model import create_model
import warnings
from keras.callbacks import ModelCheckpoint,EarlyStopping
import argparse
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"






def remove_space(string):
    return string.replace(" ","")

def get_arguments():
    parser = argparse.ArgumentParser(description='window 50 - 100 - 200')

    parser.add_argument('--window',
                        type=int,
                        default=300,
                        help='largezza finestra')
    parser.add_argument('--appliance',
                        type=remove_space,
                        default='fridge',
                        help="carico")
    parser.add_argument('--dataset_savepath',
                        type=remove_space,
                        default='/home/gstraccia/dataset/',
                        help="dataset")
    parser.add_argument('--name_log',
                        type=remove_space,
                        default='fridge',
                        help='nome del file log')
    parser.add_argument('--log_path',
                        type=remove_space,
                        default='',
                        help='path file log')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='larghezza batch size')
    parser.add_argument('--teacher_model',
                        type=remove_space,
                        default='model/',
                        help='path salva modello')
    parser.add_argument('--model',
                        type=remove_space,
                        default='GRU',
                        help='path salva modello')

    return parser.parse_args()


args = get_arguments()

log_file=args.log_path+"/"+args.name_log+".log"
img = args.name_log+".png"
print(log_file)
print(img)
windows_length = args.window
logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

MAX_X = 2000
MAX_y = 200

X_train_seg,y_train_seg,X_test_seg,y_test_seg, X2 = load_data(savepath=args.dataset_savepath,appliance=args.appliance)
#X_train_seg,y_train_seg,X_test_seg,y_test_seg, X2 = load_data2()

if not os.path.exists(args.teacher_model+"/model/"):
    print("make directory " + args.teacher_model+"/model/")
    os.makedirs(args.teacher_model+"/model/")


callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(filepath=args.teacher_model+"/model/"+args.appliance +args.model+str(args.batch_size)+".h5", verbose=0, save_best_only=True)
]
if args.model=="GRU":
    model_CONV1D = create_model(windows_length, 'GRU')
if args.model=="SGRU":
    model_CONV1D = create_model(windows_length, 'SGRU')

model_CONV1D.compile(loss='mse', optimizer="adam", metrics='acc')
print("ingresso")
print(X_train_seg.shape)
history_i = model_CONV1D.fit(X_train_seg,y_train_seg,epochs=20,batch_size=args.batch_size,validation_split=0.005,validation_data=(X_test_seg,y_test_seg),callbacks=callbacks)

pred= model_CONV1D.predict(X_test_seg)

mse_loss_norm = mse_loss(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
mae_loss_norm = mae_loss(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
#logging.warning('Mean square error on test set: '+str( mse_loss_norm)
#logging.warning('Mean absolute error on the test set: '+ mae_loss_norm)

rpaf = recall_precision_accuracy_f1(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y, 50)
rete = relative_error_total_energy(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
mae = mean_absolute_error(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)


logging.warning("============ Recall: {}".format(rpaf[0]))
logging.warning("============ Precision: {}".format(rpaf[1]))
logging.warning("============ Accuracy: {}".format(rpaf[2]))
logging.warning("============ F1 Score: {}".format(rpaf[3]))
logging.warning("============ Relative error in total energy: {}".format(rete))
logging.warning("============ Mean absolute error(in Watts): {}".format(mae))


