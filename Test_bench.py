import matplotlib.pyplot as plt
import os
import logging
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from Metrics import *
import warnings
warnings.filterwarnings("ignore")

from Load_data import load_data
from keras.models import load_model


def remove_space(string):
    return string.replace(" ","")

def get_arguments():
    parser = argparse.ArgumentParser(description='window 50 - 100 - 200')

    parser.add_argument('--appliance',
                        type=remove_space,
                        default='fridge',
                        help="carico")
    parser.add_argument('--model',
                        type=remove_space,
                        default='/home/gstraccia/model_teacher/model/fridgeSGRU100.h5',
                        help="modello")
    parser.add_argument('--name_log',
                        type=remove_space,
                        default='fridge',
                        help='nome del file log')
    parser.add_argument('--log_path',
                        type=remove_space,
                        default='',
                        help='path file log')
    parser.add_argument('--dataset_savepath',
                        type=remove_space,
                        default='/home/gstraccia/dataset/',
                        help="dataset")
    parser.add_argument('--image',
                        type=remove_space,
                        default='/home/gstraccia/dataset/',
                        help="dataset")

    return parser.parse_args()

window = 200

args = get_arguments()
log_file=args.log_path+"/"+args.name_log+".log"
logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

X_train_seg,y_train_seg,X_test_seg,y_test_seg, X_test = load_data(savepath=args.dataset_savepath, appliance=args.appliance)

model =  load_model(args.model)

MAX_X = 2000
MAX_y = 200

pred= model.predict(X_test_seg)

mse_loss_norm = mse_loss(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
mae_loss_norm = mae_loss(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)

rpaf = recall_precision_accuracy_f1(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y, 50)
rete = relative_error_total_energy(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
mae = mean_absolute_error(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
nde = get_nde(y_test_seg*MAX_y,pred.reshape(-1)*MAX_y)

logging.warning("============ Recall: {}".format(rpaf[0]))
logging.warning("============ Precision: {}".format(rpaf[1]))
logging.warning("============ Accuracy: {}".format(rpaf[2]))
logging.warning("============ F1 Score: {}".format(rpaf[3]))
logging.warning("============ Relative error in total energy (SAE): {}".format(rete))
logging.warning("============ Mean absolute error (MAE): {}".format(mae))
logging.warning("============ Normalized Decomposition Error (NDE): {}".format(nde))

plt.rcParams["figure.figsize"] = [21,9]

if args.appliance =="fridge":
    start = 227000
    stop = 231000
if args.appliance =="washingmachine":
    start = 118000
    stop = 120000
if args.appliance == "microwave":
    start = 8000
    stop = 10000
if args.appliance == "dishwasher":
    start = 8000
    stop = 14000


offset = int(window /2)
plt.plot((X_test)[start + offset:stop + offset], color ='C2', alpha = 0.6, label ='Main Value')
plt.plot((pred.reshape(-1)*MAX_y)[start:stop], color = 'C0', alpha = 0.6, label = 'Predicted value')
plt.plot((y_test_seg*MAX_y)[start:stop], color = 'C3', alpha = 0.6, label = 'True Value')
plt.title(args.appliance+ " Teacher Network")
plt.legend()
#plt.show()

plt.savefig(args.image+args.appliance+"teacher.png")