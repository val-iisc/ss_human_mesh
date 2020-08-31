
""" Pipeline Setup file. """

import sys
import os
import numpy as np 
import time as time

import tensorflow as tf 

from model_arch import NetModel
import utils
import config as cfg

class ModelComposer(object):

    def __init__(self, batch_size, is_training=True, is_fine_tune=False):
        """ Init for Pipeline """
        self.batch_size = batch_size
        self.part_batch = (self.batch_size / 2)
        self.is_training = is_training
        
        ### Create a new network model       
        self.net_scope =  "netmodel_prime"      
        self.net_model = NetModel(self.net_scope, batch_size=self.batch_size, is_training=is_training, is_fine_tune=is_fine_tune)       
        
        ### Get suitable params
        self.net_params = utils.get_network_params(self.net_scope)                  
        self.resnet_params = utils.get_net_train_params("resnet_v2_50")
        
        ### Define all trainable params
        self.trainable_params = self.net_params + self.resnet_params           
        self.var_list = tf.global_variables() 

        ### Weight save & restore        
        self.bestwt_saver = tf.train.Saver(self.var_list, max_to_keep=5)         
        self.iterwt_saver = tf.train.Saver(self.var_list, max_to_keep=5)

        ### Scope/Name
        self.scope = "composer" 
        self.name = "model_composer"

        ### Get critical network nodes        
        self.nodes_net = self.net_model.get_network_nodes()

        ## Nodes
        self.in_gt_nodes = self.nodes_net['inputs_and_gt']      
        self.cnn_nodes = self.nodes_net['cnn_layer']
        self.smpl_nodes = self.nodes_net['smpl_layer']
        self.cam_mesh_nodes = self.nodes_net['cam_mesh_module']
        self.ren_nodes = self.nodes_net['renderer_layer']        
            
    def gen_mask_render(self):
        """ Render FG-BG binary mask. """        
        vclr_white = tf.ones_like(self.smpl_nodes['scaled_pred_verts'])        
        self.ren_mask = self.net_model.call_main_render_layer(verts=self.smpl_nodes['scaled_pred_verts'], vclr=vclr_white)*255       
     
    def gen_non_symm_render(self):
        """ Render direct picked vtx_clr image. """ 
        vclr_cp = self.cam_mesh_nodes['pred_vclr_cm']        
        self.ren_img_raw = self.net_model.call_main_render_layer(verts=self.smpl_nodes['scaled_pred_verts'], vclr=vclr_cp)*255

    def gen_overlay_render(self):
        """ Render mesh overlay image. """          
        listy_ol = self.net_model.call_overlay_render_layer(verts=self.smpl_nodes['scaled_pred_verts'])
        self.ren_olay = listy_ol[0]*255
        self.ren_olside1 = listy_ol[1]*255
        
    def vis_render_img(self):
        """ Render occlusion-aware vtx_clr image. """
        pred_vclrs = self.cam_mesh_nodes['pred_vclr_cm_symm']
        listy = self.net_model.call_vis_render_layer(verts=self.smpl_nodes['scaled_pred_verts'], vclr=pred_vclrs)

        self.ren_img = listy[0]*255
        self.ren_side1   = listy[1]*255
        #self.ren_side2   = listy[2]*255

    def compose_model(self, session, en_render=True):  
        """ Setup processing pipeline. """      
        if(en_render):
            self.vis_render_img()
            self.gen_overlay_render() 
            self.gen_non_symm_render()

        ## Raw outs 
        self.pred_pose = self.cnn_nodes['pred_pose']
        self.pred_betas = self.cnn_nodes['pred_betas']
        self.pred_proj_cam = self.cnn_nodes['pred_sc_trans']

        ## SMPL outs
        self.pred_verts = self.smpl_nodes['pred_verts']     
        self.pred_j3d = self.smpl_nodes['pred_j3d']

        ## SMPL->cam outs
        self.scaled_pred_verts = self.smpl_nodes['scaled_pred_verts']          
        self.scaled_pred_j3d = self.smpl_nodes['scaled_pred_j3d']
        self.pred_j2d = self.smpl_nodes['pred_j2d']       

        ## Mesh vtx_clrs
        self.pred_vclr_pick = self.cam_mesh_nodes['pred_vclr_cm']       
        self.pred_vclr_pick_symm = self.cam_mesh_nodes['pred_vclr_cm_symm']       

    def restore_path_weights(self, session, load_dir):
        """ Restore network from a path. """
        try:
            print ('Trying to load path weights...')  
            self.iterwt_saver.restore(session, save_path = '%s' %(load_dir))  
            print ("LOADED path weights successfully.")
            return False
        except Exception as ex:
            print('Could not load weights in path: ', load_dir)
            return True

    def restore_previter_weights(self, iter_no, session, load_dir):
        """ Restore network from recent/latest iteration. """
        try:
            print ('Trying to load iter weights...')
            self.iterwt_saver.restore(session, save_path = '%siter-%d' % (load_dir, iter_no))  
            print ("LOADED iter weights successfully.")
            return False
        except Exception as ex:
            print('Could not load iter weights') 
            return True      

    def restore_prevbest_weights(self, iter_no, session, load_dir):
        """ Restore network from best val iteration. """
        try:
            print ('Trying to load best weights...')
            self.bestwt_saver.restore(session, save_path = '%sbest-%d' % (load_dir, iter_no)) 
            print ("LOADED best weights successfully.")  
            return False         
        except Exception as ex:
            print('Could not load best weights')
            return True            
