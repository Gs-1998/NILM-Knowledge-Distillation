import matplotlib.pyplot as plt
import os
import logging    # first of all import the module
from keras.callbacks import ModelCheckpoint,EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from Metrics import *
from Model import create_model
import warnings
warnings.filterwarnings("ignore")

from Load_data import load_data
from Distiller import KnowledgeDistillation
from keras.models import load_model
import argparse



def remove_space(string):
    return string.replace(" ","")

def get_arguments():
    parser = argparse.ArgumentParser(description='window 50 - 100 - 200')

    parser.add_argument('--appliance',
                        type=remove_space,
                        default='washingmachine',
                        help="carico")
    parser.add_argument('--model',
                        type=remove_space,
                        default='/home/gstraccia/model_teacher/model/washingmachineSGRU50.h5',
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


args = get_arguments()

window =200
log_file=args.log_path+"/"+args.name_log+".log"
logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

X_train_seg,y_train_seg,X_test_seg,y_test_seg, X2 = load_data(savepath=args.dataset_savepath,appliance=args.appliance)



model_student = create_model(200,"STUDENT")

model_teacher =  load_model(args.model)

Distiller = KnowledgeDistillation(student = model_student, teacher = model_teacher)
Distiller.compile(optimizer="adam", loss_fn="mse", metrics=["acc"],alpha=0.8)

Distiller.fit(X_train_seg, y_train_seg, epochs=5,batch_size=100, validation_data=(X_test_seg,y_test_seg))

student = Distiller.student
pred= student.predict(X_test_seg)

MAX_X = 2000
MAX_y = 200


accuracy = get_accuracy(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y, 50)
sae = get_sae(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
mae = get_mae(pred.reshape(-1)*MAX_y, y_test_seg*MAX_y)
nde = get_nde(y_test_seg*MAX_y,pred.reshape(-1)*MAX_y)

logging.warning("============ Accuracy: {}".format(accuracy))
logging.warning("============ Relative error in total energy (SAE): {}".format(sae))
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
plt.plot((X2)[start+offset:stop+offset], color = 'C2', alpha = 0.6, label = 'Main Value')
plt.plot((pred.reshape(-1)*MAX_y)[start:stop], color = 'C0', alpha = 0.6, label = 'Predicted value')
plt.plot((y_test_seg*MAX_y)[start:stop], color = 'C3', alpha = 0.6, label = 'True Value')
plt.title(args.appliance+ " Student Network")
plt.legend()

plt.savefig(args.image+args.appliance+"student.png")