
""" Renderering layer wrapper file. """

import numpy as np
from keras.engine.topology import Layer
from render_ortho import render_colored_batch, render_overlay_colored_batch

class RenderLayer(Layer):

    def __init__(self, width, height, num_channels, bgcolor, f, camera_f, camera_c,
                 camera_t=np.zeros(3), camera_rt=np.zeros(3), **kwargs):
    	""" Init wrapper of renderer """
        self.width = width
        self.height = height
        self.num_channels = num_channels
        
        self.f = f
        self.bgcolor = bgcolor
       
        self.camera_f = np.array(camera_f).astype(np.float32)
        self.camera_c = np.array(camera_c).astype(np.float32)
        self.camera_t = np.array(camera_t).astype(np.float32)
        self.camera_rt = np.array(camera_rt).astype(np.float32)

        super(RenderLayer, self).__init__(**kwargs)

    def call(self, v, vc ,cam_pred=None, is_img_bg=False):
    	""" Render mesh and mesh overlays """
        if(not is_img_bg):
            assert(self.num_channels == vc.shape[-1] == self.bgcolor.shape[0])

            return render_colored_batch(m_v=v, m_f=self.f, m_vc=vc, width=self.width, height=self.height,
                                    camera_f=self.camera_f, camera_c=self.camera_c, num_channels=self.num_channels,
                                    camera_t=self.camera_t, camera_rt=self.camera_rt, bgcolor=self.bgcolor,cam_pred=cam_pred)
        
        else:            
            return render_overlay_colored_batch(m_v=v, m_f=self.f, m_vc=vc, width=self.width, height=self.height,
                                    camera_f=self.camera_f, camera_c=self.camera_c, num_channels=self.num_channels,
                                    camera_t=self.camera_t, camera_rt=self.camera_rt, bgcolor=self.bgcolor,cam_pred=cam_pred)

    def compute_output_shape(self, input_shape):
    	""" Final output shape """
        return input_shape[0], self.height, self.width, self.num_channels


