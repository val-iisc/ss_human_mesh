
""" General Utilities file. """

import sys
import os

############################   NON-TF UTILS   ##########################

from skimage.util import img_as_float
import numpy as np
import cv2
import pickle
from PIL import Image
from io import BytesIO
import math
import tqdm
import scipy
import json

import matplotlib
gui_env = ['Agg','TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print ("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print ("utils.py Using:",matplotlib.get_backend())

from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from mpl_toolkits.mplot3d import Axes3D

import config as cfg

######### Basic Utils #########

def adjust_gamma(image, gamma=1.0):
    """ Gamma correct images. """
    ## Build a LUT mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    ## Apply gamma correction using the LUT
    return cv2.LUT(image, table)


def scipy_sharpen(img_flt, alpha=30):
    """ Sharpen images. """
    from scipy import ndimage

    blurred_f = ndimage.gaussian_filter(img_flt, 3)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

    img_flt = blurred_f + alpha * (blurred_f - filter_blurred_f)
    return img_flt

def read_pickle(path):
    """ Load Pickle file. """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def save_pickle(data, path):
    """ Save Pickle file. """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

######### Pose quality and Metrics  #########

def compute_similarity_transform(S1, S2):
    """ Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem. """
    transposed = False
    if S1.shape[0] != 3 and S2.shape[0] != 3:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    ## Mean
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    ## Compute variance of X1 used for scale
    var1 = np.sum(X1**2)

    ## The outer product of X1 and X2
    K = X1.dot(X2.T)

    ## Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    ## Singular vectors of K
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    ## Construct Z that fixes the orientation of R to get det(R)=1
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    ## Construct R
    R = V.dot(Z.dot(U.T))

    ## Recover scale
    scale = np.trace(R.dot(K)) / var1

    ## Recover translation
    t = mu2 - scale*(R.dot(mu1))

    ## Error
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat



def compute_error(pred_3d_all, gt_3d_all, full_out=True):
    """ MPJPE and PA_MPJPE metric computation. """
    pred_3d_all_flat = pred_3d_all.copy()
    pred_3d_all_flat = pred_3d_all_flat - pred_3d_all_flat[:, 0:1,:]
    gt_3d_all_flat = gt_3d_all.copy()
    gt_3d_all_flat = gt_3d_all_flat - gt_3d_all_flat[:, 0:1,:]
    
    joint_wise_error = []
    error = []
    pa_joint_wise_error = []
    pa_error = []
    
    for i in range(len(pred_3d_all_flat)):
        each_pred_3d = pred_3d_all_flat[i]
        each_gt_3d  = gt_3d_all_flat[i]

        tmp_err = np.linalg.norm(each_pred_3d-each_gt_3d, axis=1)
        joint_wise_error.append(tmp_err)
        error.append(np.mean(tmp_err))

        pred3d_sym = compute_similarity_transform(each_pred_3d.copy(), each_gt_3d.copy())
        tmp_pa_err = np.linalg.norm(pred3d_sym-each_gt_3d, axis=1)
        pa_joint_wise_error.append(tmp_pa_err)
        pa_error.append(np.mean(tmp_pa_err))

    joint_wise_error = np.array(joint_wise_error)

    if(full_out):        
        mpjpe = np.mean(error)*1000         ### Note: unit is mm
        pampjpe = np.mean(pa_error)*1000    ### Note: unit is mm
        return mpjpe, pampjpe
        
    else:
        return error, pa_error

###### Alternative manual regressors ######
def smplx45_to_17j(pose_smpl):
    """ SMPLX 45 joint J3D to 17 joint J3D. """ 
    ## Remove fingers
    pose_smpl = pose_smpl[:-10]
    ## Remove extra def feet
    pose_smpl = pose_smpl[:-6]
    ## Remove face
    pose_smpl = pose_smpl[:-5]
    ## Remove wrist
    pose_smpl = pose_smpl[:-2]
    ## Remove extra def spine    
    pose_smpl = np.delete(pose_smpl, 3, 0)    ## 3
    pose_smpl = np.delete(pose_smpl, 5, 0)    ## 6
    pose_smpl = np.delete(pose_smpl, 7, 0)    ## 9
    ## Remove torso
    pose_smpl = np.delete(pose_smpl, 10, 0)     ## 10
    pose_smpl = np.delete(pose_smpl, 10, 0)    ## 11
    
    ## Hip altitude increase and widen
    alt_f = 0.8
    wide_f = 8.0

    pelvis = pose_smpl[0].copy()
    r_hip = pose_smpl[2].copy()
    l_hip = pose_smpl[1].copy()

    ## Alt inc
    r_p_dir = pelvis - r_hip
    l_p_dir = pelvis - l_hip

    mag_rp = np.linalg.norm(r_p_dir)
    r_p_dir /= mag_rp
    mag_lp = np.linalg.norm(l_p_dir)
    l_p_dir /= mag_lp

    r_hip = r_hip + (r_p_dir*mag_rp*alt_f)
    l_hip = l_hip + (l_p_dir*mag_lp*alt_f)

    ## H-Widen
    hip_ctr = (r_hip + l_hip) / 2.0
    r_dir = r_hip - hip_ctr
    l_dir = l_hip - hip_ctr

    ## Unit vec
    mag = np.linalg.norm(r_dir)
    r_dir /= mag
    l_dir /= np.linalg.norm(l_dir)

    r_hip = r_hip + (r_dir*mag*wide_f)
    l_hip = l_hip + (l_dir*mag*wide_f)
        
    ## place back
    pose_smpl[2] = r_hip 
    pose_smpl[1] = l_hip

    return pose_smpl

def smpl23_to_17j_3d(pose_smpl):
    """ Simple SMPL 23 joint J3D to 17 joint J3D. """ 
    smpl_to_17j = [ [0,1],[8,11],
                [12],[17],[19],  ### or 15 , 17
                [13],[18], [20],  ### or 16 , 18                
                [14],[0],[3],
                [9,6],[9],[1],
                [4],[10,7],[10] ]

    pose_17j = np.zeros((len(smpl_to_17j),3))
    for idx in range(len(smpl_to_17j)):
        sel_idx = smpl_to_17j[idx]
        if(len(sel_idx) == 2):
            pose_17j[idx] = (pose_smpl[sel_idx[0]] + pose_smpl[sel_idx[1]]) / 2.0
           
        else:
            pose_17j[idx] = pose_smpl[sel_idx[0]]

    return pose_17j

""" SMPL J17 reordering vec. """
smpl_reorder_vec = [0, 9,
                    12, 14, 16,
                    11, 13, 15,
                    10,
                    2, 4, 6, 8,
                    1, 3, 5, 7 ]

def reorder_smpl17_to_j17(pose_3d):
    """ SMPL reorder SMPL J17 to standard J17. """
    pose_3d = pose_3d[smpl_reorder_vec]
    return pose_3d

def smpl24_to_17j_adv(pose_smpl): 
    """ Improved SMPL 23 joint J3D to 17 joint J3D. """    
    ## Hip altitude increase and widen
    alt_f = 0.8
    wide_f = 8.0

    pelvis = pose_smpl[0].copy()
    r_hip = pose_smpl[2].copy()
    l_hip = pose_smpl[1].copy()

    ## Alt inc
    r_p_dir = pelvis - r_hip
    l_p_dir = pelvis - l_hip

    mag_rp = np.linalg.norm(r_p_dir)
    r_p_dir /= mag_rp
    mag_lp = np.linalg.norm(l_p_dir)
    l_p_dir /= mag_lp

    r_hip = r_hip + (r_p_dir*mag_rp*alt_f)
    l_hip = l_hip + (l_p_dir*mag_lp*alt_f)

    ## H-Widen
    hip_ctr = (r_hip + l_hip) / 2.0
    r_dir = r_hip - hip_ctr
    l_dir = l_hip - hip_ctr

    ## Unit vec
    mag = np.linalg.norm(r_dir)
    r_dir /= mag
    l_dir /= np.linalg.norm(l_dir)

    r_hip = r_hip + (r_dir*mag*wide_f)
    l_hip = l_hip + (l_dir*mag*wide_f)
    
    ## Place back
    pose_smpl[2] = r_hip 
    pose_smpl[1] = l_hip

    ## Neck to head raise  with tilt towards nose
    alt_f = 0.7   
    head = pose_smpl[15].copy()
    neck = pose_smpl[12].copy()    

    ## Alt inc
    n_h_dir = head - neck    
    mag_nh = np.linalg.norm(n_h_dir)
    n_h_dir /= mag_nh
    head = head + (n_h_dir*mag_nh*alt_f)
    
    ## Place back
    pose_smpl[15] = head
    
    ## Remove wrist    
    pose_smpl = pose_smpl[:-2]

    ## Remove extra def spine    
    pose_smpl = np.delete(pose_smpl, 3, 0)    ## 3
    pose_smpl = np.delete(pose_smpl, 5, 0)    ## 6
    pose_smpl = np.delete(pose_smpl, 7, 0)    ## 9
    
    ## Remove torso
    pose_smpl = np.delete(pose_smpl, 10, 0)     ## 10
    pose_smpl = np.delete(pose_smpl, 10, 0)    ## 11
    
    return pose_smpl

def hip_straighten(pose_smpl):
    """ Straighten Hip in J17. """
    #pelvis = pose_smpl[0].copy()
    r_hip = pose_smpl[2].copy()
    l_hip = pose_smpl[1].copy()

    pelvis = (r_hip + l_hip) / 2

    pose_smpl[0] = pelvis

    return pose_smpl

""" Limb parents for SMPL joints. """
limb_parents = [ 0,                             
                0, 0, 0, 
                1, 2, 3, 4, 
                5, 6, 7, 8, 
                9, 9, 9, 
                12,12,12,
                16,17,18,19,20,21
                ]

""" 3D skeleton plot colours for SMPL joints. """
colors = np.array([[0,0,255], [0,255,0], [255,0,0], [255,0,255], [0,255,255], [255,255,0], [127,127,0], [0,127,0], [100,0,100],
              [255,0,255], [0,255,0], [0,0,255], [255,255,0], [127,127,0], [100,0,100], [175,100,195],
              [0,0,255], [0,255,0], [255,0,0], [255,0,255], [0,255,255], [255,255,0], [127,127,0], [0,127,0], [100,0,100],
              [255,0,255], [0,255,0], [0,0,255], [255,255,0], [127,127,0], [100,0,100], [175,100,195]])

def fig2data(fig):
    """ Convert a Matplotlib figure to a 4D numpy array with RGBA channels. """
    ## Draw the renderer
    fig.canvas.draw()

    ## Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    ## Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def draw_limbs_3d_plt(joints_3d, ax, limb_parents=limb_parents):   
    ## Direct 3d plotting
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        #ax.text(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], s=str(i))       
        ax.plot(x_pair, y_pair, z_pair, color=colors[i]/255.0, linewidth=3, antialiased=True)

def plot_skeleton_3d(joints_3d, flag=-1, limb_parents=limb_parents, title=""):
    ## 3D Skeleton plotting
    fig = plt.figure(frameon=False, figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.clear()
    
    ## Axis setup
    if (flag == 0):
        ax.view_init(azim=0, elev=0)
    elif (flag == 1):
        ax.view_init(azim=90, elev=0)     

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)

    scale = 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    draw_limbs_3d_plt(joints_3d * scale, ax, limb_parents)
    
    ax.set_title(title)
    plt_img =  fig2data(fig)
    plt.close(fig)
       
    return plt_img

def skeleton_image(joints_2d, img):
    """ 2D Joint skeleton Overlay. """ 
    img_copy = img.copy()       
    
    for i in range(joints_2d.shape[0]):            
        x_pair = [joints_2d[i, 0], joints_2d[limb_parents[i], 0]]
        y_pair = [joints_2d[i, 1], joints_2d[limb_parents[i], 1]]
        img_copy = cv2.line(img_copy, (int(x_pair[0]),int(y_pair[0])), (int(x_pair[1]),int(y_pair[1])), colors[i],4)

    return img_copy

def create_collage(img_list, axis=1):
    """ Collage a set of images to form a panel. (numpy) """
    np_new_array = np.concatenate([i for i in img_list], axis=axis)
    return np_new_array

def align_by_pelvis(joints):  
    """ Center by pelvis joint. """   
    hip_id = 0
    joints -= joints[hip_id, :]
    return joints

def mesh2d_center_by_nose(mesh2d,w=224 ,h=224): 
    """ Simple mesh centering by nose/pelvis vtx. (numpy) """   
    #hip_id = 0
    nose_id = 0
    ctr = mesh2d[nose_id,:]
    mesh_ret = mesh2d - ctr + np.array([ w/2, h/5 ])
    return mesh_ret

def align_with_image_j2d(points2d, img_width, img_height):
    """ Perform center alignment to image coordinate system. (numpy) """  
    points2d[:,0] += img_width/2
    points2d[:,1] += img_height/2
    return points2d

""" Input preprocess """
def get_transform(center, scale, res, rot=0):
    """ Generate transformation matrix. """
    h = 224 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot ## To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        ## Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """ Transform pixel location to different reference. """
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def crop(img, center, scale, res, rot=0):
    """ Crop image according to the supplied bounding box. """
    ## Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    ## Bottom right point
    br = np.array(transform([res[0]+1, res[1]+1], center, scale, res, invert=1)) - 1
    
    ## Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    ## Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    ## Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        ## Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = scipy.misc.imresize(new_img, res)
    return new_img

def j2d_crop(img, j2d_file, rescale=1.2, detection_thresh=0.2):
    """ Get center and scale for Bbox from OpenPose/Centertrack detections."""
    with open(j2d_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    ## Adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale

    img = crop(img, center, scale, (cfg.IMG_W, cfg.IMG_H))
    return img

def bbox_crop(img, bbox):
    """ Crop, center and scale image based on BBox """
    with open(bbox, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)

    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
   
    img = crop(img, center, scale, (cfg.IMG_W, cfg.IMG_H))
    return img

###########################   TF UTILS   #############################

import pickle as pkl
import tensorflow as tf
import tensorflow_graphics as tfg

from render.render_layer_ortho import RenderLayer
import render.vertex_normal_expose as dirt_expose       

PI = np.pi

def tfread_image(image,fmt='png', channels=3):
    """ Simple read and decode image. """
    if (fmt == 'png'):
        return tf.image.decode_png(image, channels=channels)
    elif (fmt == 'jpg'):
        return tf.image.decode_jpeg(image, channels=channels)
    else:
        print ("ERROR specified format not found....")   

def tf_norm(tensor, axis=1):
    """ Min-Max normalize image. """    
    min_val = tf.reduce_min(tensor, axis=axis, keepdims=True)
    normalized_tensor = tf.div( tf.subtract(tensor, min_val), tf.subtract(tf.reduce_max(tensor, axis=axis, keepdims=True), min_val))
    return normalized_tensor
    
def tfresize_image(image, size=(cfg.IMG_W, cfg.IMG_H)):
    """ Resize image. """
    return tf.image.resize(image, size)

def denormalize_image(image):
    """ Undo normalization of image. """
    image = (image / 2) + 0.5
    return image

def unprocess_image(image):
    """ Undo preprocess image. """
    # Normalize image to [0, 1]
    image = (image / 2) + 0.5
    image = image * 255.0 #[0,1] to [0,255] range

    return image

def preprocess_image(image, do_znorm=True):
    """ Preprocess image. """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (cfg.IMG_W, cfg.IMG_H))
    image /= 255.0  # normalize to [0,1] range

    if(do_znorm):
        # Normalize image to [-1, 1]
        image = 2 * (image - 0.5)

    return image

def load_and_preprocess_image(path):
    """ Simple read and preprocess for just image. """
    image = tf.io.read_file(path)
    processed_image = preprocess_image(image)
    return processed_image

def load_and_preprocess_image_and_mask(path, j2d, j3d, beta, mask_path, pose, camera, data_id):
    """ Simple read and preprocess for image and mask. """
    image = tf.io.read_file(path)
    proc_image = preprocess_image(image)

    ## For Mask
    mask = tf.io.read_file(mask_path)
    proc_mask = preprocess_image(mask, do_znorm=False)

    return proc_image, j2d, j3d, beta, proc_mask, pose, camera, data_id

def tf_create_collage(img_list, axis=2):
    """ Collage a set of images to form a panel. """
    tf_new_array = tf.concat([i for i in img_list], axis=axis)
    return tf_new_array

def log_images(tag, image, step, writer):
    """ Logs a list of images to tensorboard. """
    height, width, channel = image.shape
    image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    ## Create an Image object
    img_sum = tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)
    ## Create a Summary value
    im_summary = tf.Summary.Value(tag='%s' % (tag), image=img_sum)

    ## Create and write Summary
    summary = tf.Summary(value=[im_summary])
    writer.add_summary(summary, step)

def get_network_params(scope):
    """ Get all accessable variables. """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

def get_net_train_params(scope):
    """ Get Trainable params. """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def copy_weights(iter_no, wt_dir, label='best'):    
    """ Backup the Weights to pretrained_weights/ given iteration number and label i.e 'iter' or 'best' """
    files = os.listdir(wt_dir+label+"wt_")
    match_substr = '%s-%d' % (label, iter_no)
    files = [f for f in files if match_substr in f]
    for f in files:
        cmd = 'cp %s%s pretrained_weights/' % (wt_dir, f)
        print (cmd)
        os.system(cmd)

def get_most_recent_iteration(wt_dir, label='iter'):
    """ Gets the most recent iteration number from weights/ dir of given label: ('best' or 'iter') """
    files = os.listdir(wt_dir)
    files = [f for f in files if label in f]
    numbers = {long(f[f.index('-') + 1:f.index('.')]) for f in files}
    return max(numbers)

def copy_latest(wt_dir, wt_type='best'):
    """ Backup latest weights. """
    latest_iter = get_most_recent_iteration(label=wt_type, wt_dir=wt_dir)
    copy_weights(latest_iter, label=wt_type, wt_dir=wt_dir)
    return latest_iter

def get_latest_iter(wt_dir, wt_type='best'):
    """ Get latest weights. """
    latest_iter = get_most_recent_iteration(label=wt_type, wt_dir=wt_dir)    
    return latest_iter

def tf_align_by_pelvis(joints):
    """ Simple centering by pelvis location. """  
    hip_id = 0
    pelvis = joints[:, hip_id:hip_id+1, :]
    return tf.subtract(joints, pelvis)

def tf_mesh2d_center_by_nose(mesh2d,w=224 ,h=224):
    """ Simple mesh centering by nose/pelvis vtx. """     
    #hip_id = 0
    nose_id = 0
    ctr = mesh2d[nose_id:nose_id+1,:]
    mesh_ret = tf.add(tf.subtract(mesh2d, ctr), [[ w/2, h/5 ]])
    return mesh_ret
        
def tf_perspective_project(points3d, focal, prin_pt, name="perspective_project"):
    """ Simple Perspective Projection. """
    fx = focal[0]
    fy = focal[1]
    tx = prin_pt[0]
    ty = prin_pt[1]
    
    intrin = tf.convert_to_tensor(np.array([ [fx, 0., tx],
                                             [0., fy, ty],
                                             [0., 0., 1.]]))   
    intrin = tf.tile(intrin,[points3d.shape[0]])
    p_cam3d = tf.matmul(points3d, intrin, name=name)    
    points2d = (points3d[:,:,0:2] / points3d[:,:,2])  ### project 
    return points2d

def tf_orthographic_project(points3d, name="orthographic_project"):
    """ Simple Orthographic Projection. """
    return points3d[:,:,0:2]    ## X,Y,Z

def tf_dyn_scale_and_align(vertices, joints_3d,  scale, add_trans):
    """ Dynamic scale and trans adjust. """
    xy_max = tf.expand_dims(tf.reduce_max(vertices, axis=1),  axis=1)
    xy_min = tf.expand_dims(tf.reduce_min(vertices, axis=1),  axis=1)
    #person_ctr = (xy_max + xy_min)/2.0
    person_range = tf.abs(xy_max-xy_min)
    person_sc = tf.expand_dims(tf.reduce_max(person_range[:,:,0:2], axis=2),  axis=2)    
    
    ### Scale person to detector scale 
    vertices = tf.div(vertices, person_sc)
    vertices = vertices * scale

    joints_3d = tf.div(joints_3d, person_sc)
    joints_3d = joints_3d * scale

    ### Bbox center
    xy_max = tf.expand_dims(tf.reduce_max(vertices, axis=1),  axis=1)
    xy_min = tf.expand_dims(tf.reduce_min(vertices, axis=1),  axis=1)
    person_ctr = (xy_max + xy_min)/2.0

    add_trans = tf.concat([add_trans, tf.zeros_like(add_trans[:,:,0:1])], axis=2)

    vertices = vertices - person_ctr + add_trans
    joints_3d = joints_3d - person_ctr + add_trans

    return vertices, joints_3d, scale[:,0], ((add_trans-person_ctr)[:,0,:2])

def tf_do_scale_and_align(vertices, joints_3d,  scale, trans): 
    """ Perform Scale and trans. (in world space) """   
    scale = tf.reshape(scale, [-1, 1, 1])
    trans = tf.reshape(trans, [-1, 1, 2])

    z = tf.zeros_like(trans[:,:,0:1])
    shift = tf.concat([trans, z], axis=2)

    ### Trans in world space
    vertices = vertices + shift         
    joints_3d = joints_3d + shift 
    
    ### Scale person 
    vertices = vertices * scale
    joints_3d = joints_3d * scale

    return vertices, joints_3d

def for_tpix_tf_do_scale_and_align(vertices, joints_3d,  scale, trans):    
    """ Perform Scale and trans. (in Pixel space) """ 
    xy_max = tf.expand_dims(tf.reduce_max(vertices, axis=1),  axis=1)
    xy_min = tf.expand_dims(tf.reduce_min(vertices, axis=1),  axis=1)
    #person_ctr = (xy_max + xy_min)/2.0
    person_range = tf.abs(xy_max-xy_min)
    person_sc = tf.expand_dims(tf.reduce_max(person_range[:,:,0:2], axis=2),  axis=2)    ##ignore z

    ### Unit scale
    vertices = tf.div(vertices, person_sc)
    joints_3d = tf.div(joints_3d, person_sc)

    ###
    scale = tf.reshape(scale, [-1, 1, 1])
    trans = tf.reshape(trans, [-1, 1, 2])

    z = tf.zeros_like(trans[:,:,0:1])
    shift = tf.concat([trans, z], axis=2)     
    
    ### Scale person 
    vertices = vertices * scale
    joints_3d = joints_3d * scale

    ### Trans in cam space
    vertices = vertices + shift             
    joints_3d = joints_3d + shift

    return vertices, joints_3d

def tf_align_with_image_j2d(points2d, img_width, img_height):
    """ Perform center alignment to image coordinate system. (in Pixel space) """ 
    if(img_width == img_height):
        points2d = points2d + (img_width/2)
    else:
        width_tf = tf.zeros((points2d.shape[0], points2d.shape[1], 1),dtype=tf.int32) + (img_width/2)
        height_tf = tf.zeros((points2d.shape[0], points2d.shape[1], 1),dtype=tf.int32) + (img_height/2)
        concatd = tf.concat([width_tf, height_tf], axis=2)
        points2d = points2d + concatd

    return points2d

############ Render pipeline utils ############
MESH_PROP_FACES_FL = './assets/smpl_sampling.pkl'

""" Read face definition. Fixed for a SMPL model. """
with open(os.path.join(os.path.dirname(__file__), MESH_PROP_FACES_FL), 'rb') as f:
    sampling = pkl.load(f)

M = sampling['meshes']

faces = M[0]['f'].astype(np.int32)
faces = tf.convert_to_tensor(faces,dtype=tf.int32)

def_bgcolor = tf.zeros(3) + [0, 0.5, 0]     ## Green BG

def colour_pick_img(img_batch, vertices, batch_size):
    """ Pick clr based on mesh registration. [Vtx, Img] -> [Vtx_clr] """
    proj_verts = tf_orthographic_project(vertices) 
    verts_pix_space = tf_align_with_image_j2d(proj_verts, cfg.IMG_W, cfg.IMG_H)

    #### Pick colours and resolve occlusion softly
    verts_pix_space = tf.cast(verts_pix_space, dtype=tf.int32)
    verts_pix_space = tf.concat([verts_pix_space[:,:,1:], verts_pix_space[:,:,0:1]], axis=2)
    
    if(cfg.TF_version >= 1.14): 
        #### Alternative colour pick for TF 1.14 & above, faster inference.
        clr_picked = tf.gather_nd(params=occ_aware_mask, indices=verts_pix_space, batch_dims=1)  ### NOTE: only for tf 1.14 and above

    else:
        ### For TF 1.13 and older
        for b in range(batch_size):
            if b == 0:
                clr_picked = [tf.gather_nd(params=img_batch[b], indices=verts_pix_space[b])]
            else:
                curr_clr_pick = [tf.gather_nd(params=img_batch[b], indices=verts_pix_space[b])]
                clr_picked = tf.concat([clr_picked, curr_clr_pick], axis=0)

    img_clr_picked = tf.cast(clr_picked, dtype=tf.float32)

    return img_clr_picked

def get_occ_aware_cam_facing_mask(vertices, batch_size, part_based_occlusion_resolve=False, bgcolor=def_bgcolor):
    """ Occlusion-aware vtx weighting, depth based or part-based. [Vtx] -> [Vtx_occ_wtmap] """
    if (part_based_occlusion_resolve):
        vertex_colors = np.zeros((batch_size, 6890, 3))

        ### Part segmentation_generation
        vtx_prts = np.load("vtx_clr_smpl_proj_final_part_segmentations.npy")

        ### Vertex parts modify for maximal seperation
        vtx_prts = vtx_prts + 1
        vtx_prts[vtx_prts == 2] = 5
        vtx_prts[vtx_prts == 22] = 7
        vtx_prts[vtx_prts == 8] = 22
        vtx_prts[vtx_prts == 12] = 2
        vtx_prts[vtx_prts == 23] = 13
        vtx_prts[vtx_prts == 19] = 4
        vtx_prts[vtx_prts == 21] = 18

        #### part labelled
        vtx_part_labels = np.zeros(vertices.shape)
        vtx_prts = np.expand_dims(vtx_prts, axis=1)
        vtx_prts = vtx_prts / 24.0
        part_label = np.concatenate([vtx_prts, vtx_prts, vtx_prts], axis=1)
        vtx_part_labels[:] = part_label         ##broadcast to form batch

    #### Render cam setup             
    fixed_rt = np.array([1.0, 0.0, 0.0])   ### tilt,pan,roll
    angle = np.linalg.norm(fixed_rt)
    axis = fixed_rt / angle   
    ang = np.pi 
    new_an_ax = axis * (ang)  
    fixed_rt  = new_an_ax 
    fixed_t = [0., 0., 0.]
    ##

    fixed_renderer = RenderLayer(cfg.IMG_W, cfg.IMG_H, 3, bgcolor=bgcolor, f=faces, camera_f=[cfg.IMG_W, cfg.IMG_H], camera_c=[cfg.IMG_W/2.0, cfg.IMG_H/2.0], camera_rt=fixed_rt, camera_t=fixed_t)

    vert_norms = dirt_expose.get_vertex_normals(vertices, faces)

    #### Verts selection based on norm
    vert_norms_flat = tf.reshape(vert_norms, [-1, 3])
    fake_angle = tf.ones_like(vert_norms_flat[:,0:1], dtype=tf.float32)     ## unit mag
    euler_angles = tfg.geometry.transformation.euler.from_axis_angle(axis=vert_norms_flat, angle=fake_angle)
    vert_norms_euler = tf.reshape(euler_angles, [-1, 6890, 3])

    ### Diff. margin formulation
    quant_sharpness_factor = 50
    verts_ndiff = vert_norms_euler[:,:,2:] * -1     ## invert as cam faces 
    verts_ndiff = verts_ndiff * quant_sharpness_factor   ## centrifugal from 0.0 to get quantization effect

    #verts_ndiff = tf.math.sign(verts_ndiff)
    #verts_ndiff = tf.nn.relu(verts_ndiff)
    verts_ndiff = tf.nn.sigmoid(verts_ndiff)

    if(part_based_occlusion_resolve):
        vtx_part_labels=  tf.convert_to_tensor(vtx_part_labels, dtype=tf.float32)

        ## Normal part based resolving occlusion based render
        cam_facing_vtx_clrs = tf.multiply(vtx_part_labels, verts_ndiff)
    else:
        ## Depth based occlusion aware picking to be debugged
        depth_vertices = vertices[:,:,2:]

        ## Normalize the depth between 0 and 1
        min_val = tf.reduce_min(depth_vertices, axis=1, keepdims=True)
        normalized_depth_vertices = tf.div( tf.subtract(depth_vertices, min_val), tf.subtract(tf.reduce_max(depth_vertices, axis=1, keepdims=True), min_val))

        cam_facing_vtx_clrs = tf.tile(normalized_depth_vertices, [1,1,3])
        cam_facing_vtx_clrs = tf.multiply(cam_facing_vtx_clrs, verts_ndiff)         

    ## Mask render for occlusion resolution
    occ_aware_mask = fixed_renderer.call(vertices, vc=cam_facing_vtx_clrs) ## occulsion aware z-buffered parts masks
    
    clr_picked = colour_pick_img(occ_aware_mask, vertices, batch_size)    

    ## Occlusion resolution based on z-buffered parts
    if(part_based_occlusion_resolve):
        occ_sel_diff = (vtx_part_labels[:,:,0:1] - clr_picked[:,:,0:1] ) * 10.0

    else:
        ### Depth based colour pick
        occ_sel_diff = (normalized_depth_vertices[:,:,0:1] - clr_picked[:,:,0:1] ) * 10.0

    ### Diff. margin soft selection 
    occ_sel = tf.nn.sigmoid(occ_sel_diff) * tf.nn.sigmoid(-1 * occ_sel_diff) * 4.0

    #### Select front facing
    final_front_facing_occ_resolved = tf.multiply(occ_sel, verts_ndiff)

    return final_front_facing_occ_resolved

def apply_ref_symmetry(vclr_picked_resolved, front_facing_occ_resolved_mask, batch_size):
    """ Reflectional symmetry module. [Vtx_clr, Vtx_wtmap] -> [Vtx_clr_symm] """
    symm_arr = np.load("./assets/basic_vtx_clr_symm_map.npy")
    symm_arr_transpose = np.transpose(symm_arr)

    sym_map = tf.expand_dims(symm_arr, axis=0)
    sym_map = tf.tile(sym_map, [batch_size,1,1])

    sym_map_transpose = tf.expand_dims(symm_arr_transpose, axis=0)
    sym_map_transpose = tf.tile(sym_map_transpose, [batch_size, 1, 1])

    ## Group clr value calc
    num = tf.matmul(sym_map, vclr_picked_resolved)
    den = tf.matmul(sym_map, front_facing_occ_resolved_mask)

    den = den + 0.00001                 
    calc_val = tf.truediv(num, den)
    ### Value assign using symmtery
    vclr_symm = tf.matmul(sym_map_transpose, calc_val)

    return vclr_symm
