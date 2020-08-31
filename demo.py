
""" Demo/Inference file. """

import sys
import os

import numpy as np
import scipy.io as scio
import cv2
import pickle as pkl 
import argparse

import tensorflow as tf
from tensorflow.contrib import slim

from model_composer import ModelComposer
import utils
import config as cfg

###
BATCH_SIZE = 1
IMG_H = cfg.IMG_H
IMG_W = cfg.IMG_W

out_dir= "./data/"

### Argparse 
parser = argparse.ArgumentParser(description='Tester')    
parser.add_argument('--img_path', required=False, type=str,
                    help='Path to image, performs a single image inference. Assumes proper cropping is done already.')
parser.add_argument('--bbox', required=False, type=str,
                    help='Bbox as a json file, assumes the format of [Topleft_x, Topleft_y, bbox_width, bbox_height].')
parser.add_argument('--j2d_det', required=False, type=str,
                    help='OpenPose/CenterTrack detections as a json file, assumes the format as dumped by OpenPose when run with "--write_json" option.')
parser.add_argument('--webcam', required=False, type=int,
                    help='Camera ID to read, performs webcam inference.')

def preprocess_image(in_img, bbox=None, j2d_det=None, cvt_clr=False):
    """ Preprocess image. """
    if(type(bbox) != type(None)):
        in_img = utils.bbox_crop(in_img, bbox)
    elif(type(j2d_det) != type(None)):
        in_img = utils.j2d_crop(in_img, j2d_det)
    else:
        in_img = cv2.resize(in_img, (IMG_W, IMG_H))

    if(cvt_clr):
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)

    in_proc_img = in_img / 255.0  
    # Normalize image to [-1, 1]
    in_proc_img = 2 * (in_proc_img - 0.5)

    return in_img, in_proc_img

class Tester(object):

    def __init__(self):
        self.model_c = ModelComposer(BATCH_SIZE, is_training=False, is_fine_tune=False) 

    def restore_from_path(self, session, load_dir):
        """ Restore Model from path. """
        print ("\nRestoring checkpoint %s \n" %(load_dir))
        if(self.model_c.restore_path_weights(session, load_dir)):
            sys.exit(0)

    def restore_prev_iter_model(self, session, load_dir):
        """ Restore Model from latest checkpoint. """
        print ("\nRestoring latest checkpoint...\n")
        self.iteration = utils.get_latest_iter( wt_dir=load_dir, wt_type='iter')    
        print ('LOADING iter weights from iter-%d' %(self.iteration))
        if(self.model_c.restore_previter_weights(self.iteration, session, load_dir)):
            sys.exit(0)       

    def restore_prev_best_model(self, session, load_dir):
        """ Restore Model from best validation checkpoint. """
        print ("\nRestoring best checkpoint...\n")
        self.iteration = utils.get_latest_iter( wt_dir=load_dir, wt_type='best')    
        print ('LOADING best weights which is from iter-%d' %(self.iteration))
        if(self.model_c.restore_prevbest_weights(self.iteration, session, load_dir)):
            sys.exit(0)      

    def test(self, session, cmd_args):
        """ Restore Model and run inference. """
        self.model_c.compose_model(session)  
        img_path = args.img_path
        webcam_id = args.webcam  
        restore_str = cfg.restore_str

        init_op = tf.global_variables_initializer()
        session.run(init_op)                    

        if(restore_str == 'iter'):           
            self.restore_prev_iter_model(session=session,load_dir=cfg.model_load_path)
        elif(restore_str == 'best'): 
            self.restore_prev_best_model(session=session,load_dir=cfg.model_load_path)
        elif(type(restore_str) != type(None)):                
            self.restore_from_path(session=session,load_dir=restore_str)
        else:
            print ("\nERROR: Unable to restore model based on given restore string...\n")
            exit(0)

        if(type(webcam_id) == type(None)):
            ## Single image read        
            if(img_path == None):
                img_path = "./data/hiphop.png"        
                in_img = cv2.imread(img_path)             
            else:
                in_img = cv2.imread(img_path)           

            in_img, in_proc_img = preprocess_image(in_img, bbox=args.bbox, j2d_det=args.j2d_det, cvt_clr=True)
            test_panel = self.run_test_iter(session, in_img, in_proc_img)  

            #### Save panel image
            test_out = cv2.cvtColor(test_panel, cv2.COLOR_BGR2RGB).astype(np.uint8) 
            cv2.imwrite(out_dir + "out_panel.png", test_out)
            cv2.imshow("Output Panel", test_out)
            cv2.waitKey(0)                  

        else:
            ## Webcam inference
            vidcap = cv2.VideoCapture(webcam_id)
            print('\nPress "Esc", "q" or "Q" to exit.\n')

            while (1):               
                success, in_img = vidcap.read()
                if not success:
                    print("Frame not read. ")
                    break

                ## Center square crop 
                o_h, o_w = in_img.shape[:2]
                c_y, c_x = o_h/2, o_w/2 
                res = np.min([o_w, o_h])
                
                in_img = in_img[c_y-(res/2):c_y+(res/2), c_x-(res/2):c_x+(res/2)]

                in_img, in_proc_img = preprocess_image(in_img, cvt_clr=True)
                test_panel = self.run_test_iter(session, in_img, in_proc_img)

                test_out = cv2.resize(cv2.cvtColor(test_panel, cv2.COLOR_BGR2RGB), (IMG_W*3, IMG_H*3))
 
                cv2.imshow("Output panel", test_out)
                key = cv2.waitKey(1) 
                if (key & 0xFF == ord('q') or key == 27):
                    break

    def run_test_iter(self, session, in_image, proc_img):         
        """ Perform inference and panel creation. """
        model_c = self.model_c      
        in_model = model_c.in_gt_nodes

        output_feed = [ model_c.ren_img, model_c.ren_side1, model_c.ren_olay, model_c.ren_olside1, model_c.ren_img_raw ]
       
        proc_img = np.expand_dims(proc_img, axis=0)

        input_feed = {  in_model['in_img']: proc_img }  
        
        ## Get predictions
        [ ren_fr, ren_s1, ren_olay, ren_ols1, ren_fr_dir ] = session.run(output_feed, input_feed)

        ## Collage rendered images 
        img_id = -1                  
        ren_fr = (ren_fr[img_id]).astype(np.uint8)
        ren_s1 = (ren_s1[img_id]).astype(np.uint8)

        ren_ol = (ren_olay[img_id]).astype(np.uint8)
        ren_ols1 = (ren_ols1[img_id]).astype(np.uint8)

        ren_fr_dir = (ren_fr_dir[img_id]).astype(np.uint8)
        
        row1 = utils.create_collage([in_image, ren_ol, ren_ols1])        
        row2 = utils.create_collage([ren_fr_dir, ren_fr, ren_s1])
        test_panel = utils.create_collage([row1, row2],axis=0)
        
        return test_panel

if __name__== "__main__":  
    args = parser.parse_args()     

    ### Setup TF session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=config)

    ### Run inference
    tester = Tester()
    tester.test(session=session, cmd_args=args)
    
    print ("\n........................TESTING ENDED.............................\n")
