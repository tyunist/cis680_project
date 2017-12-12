import numpy as np
from IPython.display import IFrame
from IPython.display import display 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os, pdb
import h5py
from utils import TEMPLATE_POINTS, TEMPLATE_VG, plot_voxelgrid, plot_points, array_to_color, disp_image
from voxelgrid import VoxelGrid
from camera_model import project_3d_2d 


data_path = "/home/tynguyen/cis680/data/3d-mnist"
train_dir = os.path.join(data_path, 'train_point_clouds.h5')
with h5py.File(train_dir, 'r') as hf:
  #print('==> Dicts: ', hf.keys())
  for i in range(5):
    print('===> Iter ',i) 
    number = str(i)  
    zero = hf[number]
    digit_zero = (zero["img"][:], zero["points"][:], zero.attrs["label"])
    pc_zero = digit_zero[1]  # Numpy, 25700 x 3 
    lab_zero = digit_zero[2] # Numpy int64 
    img_zero = digit_zero[0] # Numpy 30 x 30, float64 
    print('===> Label:', zero.attrs["label"])    
    print pc_zero.max(0)
   
    # Display image  
    #disp_image(img_zero) 
  
    # Display point cloud \n",
#     iframe = plot_points(pc_zero)\n",
#     display(iframe)\n",
 
    # Create voxel 
    N = 60
    voxel_grid = VoxelGrid(pc_zero, x_y_z=[N, N, N], bb_cuboid=False)
    
    # Display 3D Voxel
    #iframe = plot_voxelgrid(voxel_grid, cmap='hot')
    #display(iframe)

    # Project points into 2D image 
    scaled_shape = voxel_grid.shape / np.min(voxel_grid.shape) # shape: step size in discretization 
    #points = np.argwhere(voxel_grid.vector)[:, ::-1] * scaled_shape
    
    s_m = plt.cm.ScalarMappable(cmap= "Oranges")
    # Voxel_grid.vector = x_y_z shape  
    # [:, :-1]: get rid of the last dimension. i. e (2 x 3 x 4 x 5) => (2 x 3 x 4)
    rgb = np.ones([N*N*N, 4])
    indices = np.where(voxel_grid.vector.reshape(-1)>0) 
    rgb[indices] = (0.0, 0.0, 0.0, 0.0) 
    #rgb[indices] = s_m.to_rgba(voxel_grid.vector.reshape(-1)[indices]) # (x_size*y_size*z_size, 4)
    #rgb = rgb[voxel_grid.vector.reshape(-1) > 0][:,:-1]  # Consider only > 0 and ignore last collum => (N,3)
    rgb = rgb[:,:-1]
    #rgb[np.where(voxel_grid.vector.reshape(-1)==0)] = (1,1,1)
    # data_sample = rgb[0,:,:,:,0]
    #project_3d_2d(rgb,N=N, visual=True) 

    
    # Play around with vector of voxel 
    vector = voxel_grid.vector 
    xx, yy, zz = np.where(vector>0) 

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    scatter = ax.scatter(xx, yy, zz, cmap='coolwarm',linewidth=0, antialiased=False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    # Display 2D images on different z levels
    #voxel_grid.plot(d=2)
    #plt.show() 
