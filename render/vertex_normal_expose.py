
""" Expose Normals and other rasterizer properties. """

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import dirt
from dirt import matrices
import dirt.lighting as lighting

def get_vertex_normals(vertices, faces):
	""" Get vertex normals. [Vtx, Face_def] -> [Vtx_normals] """
	vtx_nrm = lighting.vertex_normals(vertices=vertices, faces=faces)

	return vtx_nrm

def get_pre_split_vertex_normals(vertices, faces):
	""" Get Pre-split vertex normals, computationlly slightly more efficient. [Vtx, Face_def] -> [Vtx_normals] """
	norms_by_vertex = lighting.vertex_normals_pre_split(vertices=vertices, faces=faces)

	return norms_by_vertex

def get_face_normals(vertices, faces):
	""" Get face normals along with constituent vertex IDs. [Vtx, Face_def] -> [Face_normals, vtx_ids] """
	face_nrm, verts_idx = lighting._get_face_normals(vertices=vertices, faces=faces)

	return [face_nrm, verts_idx]
