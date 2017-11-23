#  HAKUNA MATATA

"""
VoxelGrid Class
"""

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import IFrame
from IPython.display import display 
 

# from utils import plot_voxelgrid



def plot_voxelgrid(v_grid, cmap="Oranges", axis=False):

    scaled_shape = v_grid.shape / np.min(v_grid.shape)

    # coordinates returned from argwhere are inversed so use [:, ::-1]
    points = np.argwhere(v_grid.vector)[:, ::-1] * scaled_shape

    s_m = plt.cm.ScalarMappable(cmap=cmap)
    rgb = s_m.to_rgba(v_grid.vector.reshape(-1)[v_grid.vector.reshape(-1) > 0])[:,:-1]

    camera_position = points.max(0) + abs(points.max(0))
    look = points.mean(0)
    
    if axis:
        axis_size = points.ptp() * 1.5
    else:
        axis_size = 0

    with open("plotVG.html", "w") as html:
        html.write(TEMPLATE_VG.format( 
            camera_x=camera_position[0],
            camera_y=camera_position[1],
            camera_z=camera_position[2],
            look_x=look[0],
            look_y=look[1],
            look_z=look[2],
            X=points[:,0].tolist(),
            Y=points[:,1].tolist(),
            Z=points[:,2].tolist(),
            R=rgb[:,0].tolist(),
            G=rgb[:,1].tolist(),
            B=rgb[:,2].tolist(),
            S_x=scaled_shape[0],
            S_y=scaled_shape[2],
            S_z=scaled_shape[1],
            n_voxels=sum(v_grid.vector.reshape(-1) > 0),
            axis_size=axis_size))

    return IFrame("plotVG.html",width=800, height=800)



class VoxelGrid(object):
    
    def __init__(self, points, x_y_z=[1, 1, 1], bb_cuboid=True, build=True):
        """
        Parameters
        ----------         
        points: (N,3) ndarray
                The point cloud from wich we want to construct the VoxelGrid.
                Where N is the number of points in the point cloud and the second
                dimension represents the x, y and z coordinates of each point.
        
        x_y_z:  list
                The segments in wich each axis will be divided.
                x_y_z[0]: x axis 
                x_y_z[1]: y axis 
                x_y_z[2]: z axis

        bb_cuboid(Optional): bool
                If True(Default):   
                    The bounding box of the point cloud will be adjusted
                    in order to have all the dimensions of equal lenght.                
                If False:
                    The bounding box is allowed to have dimensions of different sizes.
        """
        self.points = points

        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        if bb_cuboid:
            #: adjust to obtain a  minimum bounding box with all sides of equal lenght 
            diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
            xyzmin = xyzmin - diff / 2
            xyzmax = xyzmax + diff / 2
        
        self.xyzmin = xyzmin
        self.xyzmax = xyzmax

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num 
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)
        
        self.segments = segments

        self.shape = shape

        self.n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        self.n_x = x_y_z[0]
        self.n_y = x_y_z[1]
        self.n_z = x_y_z[2]
        
        self.id = "{},{},{}-{}".format(x_y_z[0], x_y_z[1], x_y_z[2], bb_cuboid)

        if build:
            self.build()


    def build(self):

        structure = np.zeros((len(self.points), 4), dtype=int)

        structure[:,0] = np.searchsorted(self.segments[0], self.points[:,0]) - 1

        structure[:,1] = np.searchsorted(self.segments[1], self.points[:,1]) - 1

        structure[:,2] = np.searchsorted(self.segments[2], self.points[:,2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:,3] = ((structure[:,1] * self.n_x) + structure[:,0]) + (structure[:,2] * (self.n_x * self.n_y)) 
        
        self.structure = structure

        vector = np.zeros(self.n_voxels)
        count = np.bincount(self.structure[:,3])
        vector[:len(count)] = count

        self.vector = vector.reshape(self.n_z, self.n_y, self.n_x)

 
    def plot(self, d=2, cmap="Oranges", axis=False):

        if d == 2:

            fig, axes= plt.subplots(int(np.ceil(self.n_z / 4)), 4, figsize=(8,8))

            plt.tight_layout()

            for i,ax in enumerate(axes.flat):
                if i >= len(self.vector):
                    break
                im = ax.imshow(self.vector[i], cmap=cmap, interpolation="none")
                ax.set_title("Level " + str(i))

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('NUMBER OF POINTS IN VOXEL')

        elif d == 3:
            return plot_voxelgrid(self, cmap=cmap, axis=axis)

