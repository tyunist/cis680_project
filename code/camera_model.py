import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting

import h5py
import pdb 
from scipy import interpolate

# Load data 
data_path = '/home/tynguyen/cis680/data/3d-mnist/'
with h5py.File(data_path + 'full_dataset_vectors.h5', 'r') as hf:
    # 1e4 training samples, each of which are flattened voxel (16*16*16)
    # each element in 4096-D vector is intensity, ranging from 0 or 1  
    x_train_raw = hf["X_train"][:] # (10000, 4096) 
    y_train_raw = hf["y_train"][:] #(10000,) 

# Check length of the dataset 
assert(len(x_train_raw) == len(y_train_raw))


# 1D vector to rgb values, provided by ../input/plot3d.py
def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]

# Transform data from 1d to 3d rgb
def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    return np.asarray(data_t, dtype=np.float32)

def rot_x(theta):
  cos = np.cos(theta)
  sin = np.sin(theta)
  return np.array([
                  [1,  0,   0],
                  [0, cos, -sin], 
                  [0, sin,  cos] 
                  ])


def rot_y(theta):
  cos = np.cos(theta)
  sin = np.sin(theta)
  return np.array([
                  [cos, 0, sin],
                  [0,   1,  0], 
                  [-sin,0, cos] 
                  ])

def rot_z(theta):
  cos = np.cos(theta)
  sin = np.sin(theta)
  return np.array([
                  [cos, -sin, 0],
                  [sin,  cos, 0], 
                  [0  ,  0,   1] 
                  ])
img_w = 240
img_h = 240


#img = np.zeros([img_h, img_w, 3]) 

# Intrinsic parameters
K = np.array([ [200, 0, 0], 
               [0, 200, 0], 
               [0, 0, 1]])

# Extrinsic parameters 
#Rt = np.array([[0.5, -np.sqrt(3)/2, 0, 100], 
#               [np.sqrt(3)/2, 0.5, 0, 100], 
#               [0, 0, 1, 120]])


T = np.array([50, 50, 120]).reshape([3,1]) 
theta_x = 0
theta_y = 0
theta_z = 90

R_x = rot_x(theta_x/180.*np.pi)
R_y = rot_y(theta_y/180.*np.pi)
R_z = rot_z(theta_z/180.*np.pi)

R = np.dot(np.dot(R_z, R_y), R_x)
Rt = np.concatenate([R, T], 1)  
P = (np.dot(K, Rt))
print P
 
def project_3d_2d(data_sample, N=16, visual=False): 
    # obj_data = x_train[d_num,:,:,:,0].flatten() # (16x16x16, ) 
    xx, yy, zz = np.meshgrid(range(N), range(N), range(N)) 

    xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten() 
    ones = np.ones_like(xx)

    xyz_ones = np.vstack([xx, yy, zz, ones])


    #pdb.set_trace() 

    uvs = np.dot(np.dot(K, Rt), xyz_ones)

    # Normalize the last row to get uv_ones 
    uv_ones = uvs*1.0/(uvs[2]) 

    # # Sort uv_ones according to the first row 
    # sorted_indices = np.argsort(uv_ones[0])

    # # Sort uv_ones 
    # for i in range(3):
    #   uv_ones[i] = uv_ones[i, sorted_indices]

    # Floor indices 
    uu = np.floor(uv_ones[0])
    vv = np.floor(uv_ones[1])
 
    img = np.zeros([img_h, img_w, 3])
     
    for i in range(img_h):
        for j in range(img_w):
            index = np.where((np.abs(uu-i)<=0.2)*(np.abs(vv-j)<=0.2))[0]
            try:
                img[i,j] = np.mean(data_sample[index])*255
            except:
                pass 
    img = np.clip(img, 0, 255) 
    if visual:
        plt.imshow(img.astype(np.uint8))
        plt.show() 
 
 

def test_project_3d_2d():
    # Each data point now has shape 16x16x16x3 (rgb) 
    x_train = rgb_data_transform(x_train_raw)
    data_sample = x_train[0,:,:,:,0]
    project_3d_2d(data_sample) 

