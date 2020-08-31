
""" Batched Render file. """

import dirt
import numpy as np
import tensorflow as tf

from dirt import matrices
import dirt.lighting as lighting
from tensorflow.python.framework import ops

def orthgraphic_projection(w, h, near=0.1, far=10., name=None):
    """Constructs a orthographic projection matrix.
    This function returns a orthographic projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.
    """
    with ops.name_scope(name, 'OrthographicProjection', [w, h, near, far]) as scope:
        ### symmetric view case      
        right = w / 2 
        left = -right 
        top = h / 2 
        bottom = -top 

        elements = [
            [2. / (right-left), 0., 0, -(right+left)/(right-left) ],
            [0., 2. / (top-bottom), 0, -(top+bottom)/(top-bottom) ],
            [0., 0., -2. / (far - near), -(far+near)/(far-near)   ],
            [0.,0.,0.,1.]
        ]

        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))

def perspective_projection(f, c, w, h, near=0.1, far=10., name=None):
    """Constructs a perspective projection matrix.
    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.
    """
    with ops.name_scope(name, 'PerspectiveProjection', [f, c, w, h, near, far]) as scope:
        f = 0.5 * (f[0] + f[1])
        pixel_center_offset = 0.5
        right = (w - (c[0] + pixel_center_offset)) * (near / f)
        left = -(c[0] + pixel_center_offset) * (near / f)
        top = (c[1] + pixel_center_offset) * (near / f)
        bottom = -(h - c[1] + pixel_center_offset) * (near / f)

        elements = [
            [2. * near / (right - left), 0., (right + left) / (right - left), 0.],
            [0., 2. * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]

        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))

def render_colored_batch(m_v, m_f, m_vc, width, height, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None,batch_size=None,cam_pred=None):
    """ Render a batch of meshes with fixed BG. Supported projection types 1) Perspective, 2) Orthographic. """

    with ops.name_scope(name, "render_batch", [m_v]) as name:
        assert (num_channels == m_vc.shape[-1] == bgcolor.shape[0])

        #projection_matrix = perspective_projection(camera_f, camera_c, width, height, .1, 10)
        projection_matrix = orthgraphic_projection(width, height, -(width/2), (width/2))    ### im_w x im_h x im_w cube

        ## Camera Extrinsics, rotate & trans
        view_matrix = matrices.compose( matrices.rodrigues(camera_rt.astype(np.float32)),
                                        matrices.translation(camera_t.astype(np.float32)),
                                        )
        ## Fixed clr BG
        bg = tf.tile(bgcolor[tf.newaxis,tf.newaxis,tf.newaxis,...],[tf.shape(m_v)[0],width,height,1])

        m_v = tf.cast(m_v, tf.float32)
        m_v = tf.concat([m_v, tf.ones_like(m_v[:, :, -1:])], axis=2)

        ## Extrinsic multiplication 
        m_v = tf.matmul(m_v, tf.tile(view_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))

        ## Intrinsic Camera projection 
        m_v = tf.matmul(m_v, tf.tile(projection_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))
        
        m_f = tf.tile(tf.cast(m_f, tf.int32)[tf.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        ## Rasterize
        return dirt.rasterise_batch(bg, m_v, m_vc, m_f, name=name)

def render_overlay_colored_batch(m_v, m_f, m_vc, width, height, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None,batch_size=None,cam_pred=None):
    """ Render a batch of meshes with corresponding BG images. Supported projection types 1) Perspective, 2) Orthographic. """ 

    with ops.name_scope(name, "render_batch", [m_v]) as name:

        #projection_matrix = perspective_projection(camera_f, camera_c, width, height, .1, 10)
        projection_matrix = orthgraphic_projection(width, height, -(width/2), (width/2))    ### im_w x im_h x im_w cube

        ## Camera Extrinsics, rotate & trans
        view_matrix = matrices.compose( matrices.rodrigues(camera_rt.astype(np.float32)),
                                        matrices.translation(camera_t.astype(np.float32)),
                                        )
        ## Image BG
        bg = bgcolor 

        m_v = tf.cast(m_v, tf.float32)
        m_v = tf.concat([m_v, tf.ones_like(m_v[:, :, -1:])], axis=2)

        ## Extrinsic multiplication  
        m_v = tf.matmul(m_v, tf.tile(view_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))

        ## Intrinsic Camera projection 
        m_v = tf.matmul(m_v, tf.tile(projection_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))
       
        m_f = tf.tile(tf.cast(m_f, tf.int32)[tf.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        ## Rasterize
        return dirt.rasterise_batch(bg, m_v, m_vc, m_f, name=name)
