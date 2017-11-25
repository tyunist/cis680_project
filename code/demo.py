import numpy as np
from IPython.display import IFrame
from IPython.display import display 
from matplotlib import pyplot as plt
import os, pdb
import h5py
from utils import TEMPLATE_POINTS, TEMPLATE_VG, plot_voxelgrid, plot_points, array_to_color, disp_image
from voxelgrid import VoxelGrid
data_path = "/home/tynguyen/cis680/data/3d-mnist"
train_dir = os.path.join(data_path, 'train_point_clouds.h5')
 
with h5py.File(train_dir, 'r') as hf:
    zero = hf["0"]
    digit_zero = (zero["img"][:], zero["points"][:], zero.attrs["label"])
    pc_zero = digit_zero[1]  # Numpy, 25700 x 3 
    lab_zero = digit_zero[2] # Numpy int64 
    img_zero = digit_zero[0] # Numpy 30 x 30, float64 
    
    print pc_zero.max(0)
   
    # Display image \n",
#     disp_image(img_zero)\n",
  
    # Display point cloud \n",
#     iframe = plot_points(pc_zero)\n",
#     display(iframe)\n",
 
    # Create voxel 
    voxel_grid = VoxelGrid(pc_zero, x_y_z=[30, 30, 30], bb_cuboid=True)
    
    # Display 3D Voxel
    #iframe = plot_voxelgrid(voxel_grid, cmap='hot')
    #display(iframe)
    
    # Display 2D images on different z levels
    voxel_grid.plot(d=2)
    plt.show() 
