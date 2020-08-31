
""" Network Architecture file. """

import sys
import os
import numpy as np 
import time as time
import pickle as pkl

import tensorflow as tf 
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

from smpl.smpl_layer import SmplTPoseLayer
from smpl.batch_lbs import batch_rodrigues

from render.render_layer_ortho import RenderLayer
import render.vertex_normal_expose as dirt_expose

import utils
import config as cfg

### Arch specific flags
JOINT_RES = cfg.JOINT_RES
IMG_W = cfg.IMG_W                     
IMG_H = cfg.IMG_H

emb_size = cfg.emb_size
n_preds = cfg.n_preds 

PI = np.pi

def Res50_backbone_setup(img_in, is_training=True, weight_decay=0.001, reuse=False):
    """ Resnet v2-50, CNN backbone to process image. """        
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
        net, end_points = resnet_v2.resnet_v2_50(img_in, num_classes=None, is_training=is_training, reuse=reuse, scope='resnet_v2_50')
        net = tf.squeeze(net, axis=[1, 2])  

    return net

def FC_Encoder( cnn_ft, l_neurons=1024, num_preds=n_preds, is_training=True, reuse=False, name="FC_Encoder"):
    """ FC layer to process image features. """
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(cnn_ft, l_neurons, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, l_neurons, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(factor=.01, mode='FAN_AVG', uniform=True)
        net_out = slim.fully_connected( net, num_preds, activation_fn=None, weights_initializer=small_xavier, scope='fc3')

    return net_out

class NetModel(object):

    def __init__(self, scope_name, batch_size, is_training=True, is_fine_tune=False):
        """ Init for Model-arch """ 
        self.scope = scope_name
        self.batch_size = batch_size
        self.is_training = is_training
        self.is_fine_tune = is_fine_tune
        self.num_stages = cfg.num_stages

        with tf.variable_scope(self.scope):
            self.add_placeholders()            

        self.build_network()            

    def add_placeholders(self):
        """ Input Placeholders """
        self.in_img = tf.placeholder(tf.float32, shape=[self.batch_size, IMG_H, IMG_W, 3])       

    def setup_cnn_layers(self):
        """ CNN + FC regressor, Img -> [Pose_emb, Beta, Cam, Scale, Trans] """
        ### CNN backbone
        res_img_ft = Res50_backbone_setup(self.in_img, is_training = self.is_training and not self.is_fine_tune, reuse=False)

        with tf.variable_scope(self.scope) as scope:    
            FC_layer = FC_Encoder

            ### Base init params
            base_pose_emb = np.zeros([1, cfg.emb_size], dtype=np.float32)
            base_betas = np.array([[0.20560974, 0.33556297, -0.35068282, 0.35612896, 0.41754073, 0.03088791, 0.30475676, 0.23613405, 0.20912662, 0.31212646]], dtype=np.float32) 
            base_cam = np.array([[ 1.0, 0, 0]], dtype=np.float32)
            base_sc_trans = np.array([[ 0, 0, 0]], dtype=np.float32) 
            base_params = np.concatenate([base_pose_emb, base_cam, base_betas, base_sc_trans], axis=1)          

            base_params = tf.Variable(base_params, name="base_params", dtype=tf.float32)

            prev_pred = tf.tile(base_params, [self.batch_size, 1])

            ### FC layers
            for i in np.arange(self.num_stages):                
                curr_state = tf.concat([res_img_ft, prev_pred], 1)

                if i == 0:
                    delta_pred = FC_layer(curr_state, l_neurons=2048, is_training=self.is_training, reuse=False)
                else:
                    delta_pred = FC_layer(curr_state, l_neurons=2048, is_training=self.is_training, reuse=True)

                # Curr pred
                curr_pred = prev_pred + delta_pred  
                prev_pred = curr_pred   

            ### Final output
            net_pred = curr_pred    
            
            ### Theta/Pose 
            self.pred_pose_emb = tf.math.tanh( net_pred[:, :emb_size], name='tanh_emb')     ### Nxemb_size
            
            ### Betas/Shape 
            self.pred_betas = net_pred[:,emb_size+3:emb_size+13]    ### Nx10
            
            ### Global Camera/person orientation  
            self.pred_cam_axan = tf.math.tanh(net_pred[:, emb_size:emb_size+3], name='tanh_cam') * PI     ### Nx3       
            
            ### Scale and Trans
            mean_scale = np.full((self.batch_size, 1), 0.8*cfg.IMG_H, dtype=np.float32)
            var_scale = mean_scale/3
            self.pred_scale = mean_scale + ( tf.math.tanh(net_pred[:, emb_size+13:emb_size+14], name='tanh_scale') * var_scale )        ### Nx1                    
            self.pred_trans = tf.math.tanh(net_pred[:, emb_size+14:], name='tanh_trans') * (cfg.IMG_W/4)            ### Nx2         

            self.pred_sc_trans = tf.concat([self.pred_scale, self.pred_trans],axis=1)       ### Nx3
                                       
    def setup_pose_emb_layer(self):        
        """ Pose-prior module, [Pose_emb,Cam] -> Pose. """
        with tf.variable_scope('AAE_Decoder', reuse=tf.AUTO_REUSE):

            fc_out = tf.layers.dense(self.pred_pose_emb, 512, activation=tf.nn.relu)
            fc_out = tf.layers.dense(fc_out, 1024, activation=tf.nn.relu)  
            fc_out = tf.layers.dense(fc_out, 1024, activation=tf.nn.relu)       
            
            theta_dec = tf.layers.dense(fc_out, 23*3)        

        self.repose = theta_dec   ### Nx23x3                     
        self.pred_pose = tf.concat( (self.pred_cam_axan, self.repose), axis = 1 ) ### Nx24x3

    def setup_smpl_layers(self): 
        """ SMPL + Projection module, [Theta, Beta, Cam]-> [Mesh, J3D, J2D]. """
        smpl = SmplTPoseLayer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=True)

        offsets = tf.zeros(None, 1)    ### Nx1
        zero_trans = tf.zeros_like(self.pred_cam_axan)  ### Nx3

        ### Pred SMPL out  
        pred_pose_reshape = tf.reshape(batch_rodrigues(tf.reshape(self.pred_pose, [-1, 3])), [-1, 24, 3, 3])       ### Nx24x3x3
        self.cam_smpl_out = smpl([pred_pose_reshape, self.pred_betas, zero_trans, offsets])
        
        ## Raw Vertices  [in world space]
        self.pred_verts = self.cam_smpl_out[0]   # Nx6890x3           
        ## Raw Joints 3d [in world space]
        self.pred_j3d = self.cam_smpl_out[1]     # Nx24x3
        
        ### Scale and Trans 
        if(cfg.PRED_DYN_SCALE_AND_ALIGN):
            ## For known cropping [200/224]
            self.scaled_pred_verts, self.scaled_pred_j3d, self.app_scale_pred, self.app_trans_pred = utils.tf_dyn_scale_and_align(vertices=self.pred_verts, joints_3d=self.pred_j3d, scale=200, add_trans=0)
            self.app_sc_trans_pred = tf.concat([self.app_scale_pred, self.app_trans_pred], axis=1)
            
        else:            
            ## Apply predicted scale and trans
            self.scaled_pred_verts, self.scaled_pred_j3d = utils.for_tpix_tf_do_scale_and_align(vertices=self.pred_verts, joints_3d=self.pred_j3d, scale=self.pred_scale, trans=self.pred_trans)   
        
        ### Project onto 2D for Joints2D          
        self.pred_j2d = utils.tf_orthographic_project(self.scaled_pred_j3d)         # Nx24x2
        #self.pred_j2d = utils.tf_align_with_image_j2d(self.pred_j2d, self.in_img.shape[1], self.in_img.shape[2]) 

    def setup_cam_mesh_relation_module(self):
        """ Mesh to Image relation + Reflectional Symmetry module. [Vtx, Img] -> [Vtx_clr, Vtx_clr_symm] """
        ### Unprocess image
        self.denorm_image = utils.denormalize_image(self.in_img)        

        ### Occlusion-aware weights
        pred_camfront_occ_resolved = utils.get_occ_aware_cam_facing_mask(self.scaled_pred_verts, self.batch_size)

        pred_img_clr_picked = utils.colour_pick_img(self.denorm_image, self.scaled_pred_verts, self.batch_size)
        pred_img_clr_picked_resolved = tf.multiply(pred_img_clr_picked, pred_camfront_occ_resolved)         

        ### Apply Reflectional Symmetry
        self.pred_vclr_cm = pred_img_clr_picked 
        self.pred_vclr_cm_symm = utils.apply_ref_symmetry(pred_img_clr_picked_resolved, pred_camfront_occ_resolved, self.batch_size)        
       
    def setup_renderer_layer(self):       
        """ Rendering Module, Init for differentiable-renderers. """
        MESH_PROP_FACES_FL = './assets/smpl_sampling.pkl'
        
        with open(os.path.join(os.path.dirname(__file__), MESH_PROP_FACES_FL), 'rb') as f:
            sampling = pkl.load(f)
        
        M = sampling['meshes']
        
        self.faces = M[0]['f'].astype(np.int32)
        self.faces = tf.convert_to_tensor(self.faces,dtype=tf.int32)

        bgcolor = tf.zeros(3)       ## Black bg        
        fixed_t = [0.0, 0.0, 0.0]

        ### View 1, front view
        fixed_rt = np.array([1.0, 0.0, 0.0]) * PI                
        self.renderer = RenderLayer(IMG_W, IMG_H, 3, bgcolor=bgcolor, f=self.faces, camera_f=[IMG_W, IMG_H], camera_c=[IMG_W/2.0, IMG_H/2.0], camera_rt=fixed_rt, camera_t=fixed_t)
        
        ### Overlay Renderer
        bg_overlay = self.denorm_image
        self.renderer_olay = RenderLayer(IMG_W, IMG_H, 3, bgcolor=bg_overlay, f=self.faces, camera_f=[IMG_W, IMG_H], camera_c=[IMG_W/2.0, IMG_H/2.0], camera_rt=fixed_rt, camera_t=fixed_t)
        
        ### View 2, -60 deg side view
        fixed_rt = np.array([2.72, 0.0, -1.57])
        self.renderer2 = RenderLayer(IMG_W, IMG_H, 3, bgcolor=bgcolor, f=self.faces, camera_f=[IMG_W, IMG_H], camera_c=[IMG_W/2.0, IMG_H/2.0], camera_rt=fixed_rt, camera_t=fixed_t)
       
        '''
        ### View 3, +60 deg side view
        #fixed_rt = np.array([ 2.72, 0.0, 1.57]) 
        self.renderer3 = RenderLayer(IMG_W, IMG_H, 3, bgcolor=bgcolor, f=self.faces, camera_f=[IMG_W, IMG_H], camera_c=[IMG_W/2.0, IMG_H/2.0], camera_rt=fixed_rt, camera_t=fixed_t)
        '''
        ########
    def call_main_render_layer(self, verts, vclr): 
        """ Render front view, [Vtx, Vtx_clr] -> [Ren_img] """     
        return self.renderer.call(v=verts, vc=vclr)

    def call_overlay_render_layer(self, verts):
        """ Render Mesh Overlays, [Vtx, Img] -> [Overlay_img] """
        fixed_clr_2 = np.array(cfg.overlay_clr).astype(np.float32)     #### clr of overlay mesh

        vert_norms = dirt_expose.get_vertex_normals(verts, self.faces)

        s_norm = tf.reduce_mean(vert_norms, axis=2, keepdims=True)
        s_norm = utils.tf_norm(s_norm, axis=1) 
        overlay_vclr = tf.image.adjust_gamma( tf.tile(s_norm, [1,1,3]) , 0.35) * fixed_clr_2

        return [self.renderer_olay.call(v=verts, vc=overlay_vclr, is_img_bg=True), self.renderer2.call(v=verts, vc=overlay_vclr)]

    def call_vis_render_layer(self, verts, vclr): 
        """ Render multiple views, [Vtx, Vtx_clr] -> [ren_V1_img, ....] """     
        return [self.renderer.call(v=verts, vc=vclr), self.renderer2.call(v=verts, vc=vclr)]#, self.renderer3.call(v=verts, vc=vclr)]          

    def build_network(self):
        """ Setup Arch and initialize sub modules. """  
        self.setup_cnn_layers()   
        self.setup_pose_emb_layer()     
        self.setup_smpl_layers()
        self.setup_cam_mesh_relation_module()
        self.setup_renderer_layer()

    def get_network_nodes(self):  
        """ Important Nodes to tap into.  """ 
        inputs =    { "in_img": self.in_img 
                    } 

        cnn_outs =  { "pred_pose": self.pred_pose, "pred_betas": self.pred_betas, 
                      "pred_cam_axan": self.pred_cam_axan , "pred_scale": self.pred_scale, 
                      "pred_trans": self.pred_trans, "pred_sc_trans": self.pred_sc_trans, 
                    } 

        smpl_outs = { "pred_verts": self.pred_verts, "pred_j3d": self.pred_j3d, "pred_j2d": self.pred_j2d,
                      "scaled_pred_verts": self.scaled_pred_verts, "scaled_pred_j3d": self.scaled_pred_j3d                      
                    }

        render_outs = { "renderer": self.renderer 
                      }

        cam_mesh_outs = { "pred_vclr_cm": self.pred_vclr_cm, "pred_vclr_cm_symm": self.pred_vclr_cm_symm 
                        }

        return { "inputs_and_gt": inputs, "cnn_layer": cnn_outs, "smpl_layer": smpl_outs, "renderer_layer": render_outs, "cam_mesh_module": cam_mesh_outs }

    '''
    def print_DEBUG_STR(self):
        print ("\n\n:::::::: NetModel DEBUG_STR ::::::::\n")
        ### Can add debug string options here
        print ("\n\n")
    '''