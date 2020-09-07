
""" Project Configuration file. """

import os, sys, numpy as np, datetime, time

def add_pypath(path):
    """ Insert path to system path. """
    if path not in sys.path:
        sys.path.insert(0, path)

def make_folder(path):
    """ Make folder. """
    if not os.path.exists(path):
        os.makedirs(path)

ROOT_DIR = './'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
EXP_NAME = 'release'

### INFERENCE
overlay_clr = [0.5, 0.5, 1.0]	##Purple
TF_version = 1.13
restore_str = "./weights/weak_sup_train"

### DATALOADER
TRANS_MAX_JITTER = 20 #Pix space 
SC_MAX_JITTER = 1.0
SC_MIN_JITTER = 0.75

### ARCH
JOINT_RES = 24
IMG_W = 224                     
IMG_H = 224
IMG_SIZE = 224
emb_size = 32
n_preds = emb_size + 16 ##32+10+3+3
num_stages = 3
PRED_DYN_SCALE_AND_ALIGN = False		## Auto scale and align
GT_DYN_SCALE_AND_ALIGN = True

### Organize and Save 
LOG_PATH = ROOT_DIR + 'logs/'

tf_log_path = os.path.join(LOG_PATH, EXP_NAME, 'tf_logs/')

current_time = datetime.datetime.now().strftime('M:%m_D:%d_T:%H_%M')

train_log_dir = tf_log_path + current_time + '/train/'
val_log_dir = tf_log_path + current_time + '/val/'
test_log_dir = tf_log_path + current_time + '/test/'

MODEL_WT_PATH = ROOT_DIR + 'weights/'
model_save_path = os.path.join(MODEL_WT_PATH, EXP_NAME + '/')

#make_folder(model_save_path)
model_load_path = model_save_path

SMPL_NEUTRAL = os.path.join(ROOT_DIR, 'assets', 'neutral_smpl.pkl')
SMPL_MALE = os.path.join(ROOT_DIR, 'assets', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
SMPL_FEMALE = os.path.join(ROOT_DIR, 'assets', 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')

#make_folder(os.path.join(LOG_PATH, EXP_NAME))

#os.system(('cp {} {}/').format(os.path.join(ROOT_DIR, '*.py'), os.path.join(LOG_PATH, EXP_NAME)))
