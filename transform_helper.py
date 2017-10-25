import scipy.ndimage as ndi
import numpy as np
from PIL import Image

def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix
        
def apply_transform(x, transform_matrix, channel_axis=2, fill_mode='nearest', fill_value=0.):
        x = np.rollaxis(x, channel_axis, 0)
        x = x.astype('float32')
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

def random_transform_fn(x, T):
        """
        Randomly transform an image from the given parameters

        Transforms:
        - rotate
        - shift
        - shear
        - zoom
        - flip
        
        Arguments
        ---------
        x : np.ndarray
        y : np.ndarray
        T : dictionary
                holds values for the various transforms
                example:
                T = { 
                        "rotation_range"  : 15,
                        "shift_range"     : [0.3,0.3],
                        "shear_range"     : 0.1,
                        "zoom_range"      : [1,1.4],
                        "horizontal_flip" : True,
                        "vertical_flip"   : False,
                        "x_fill_mode"     : "constant",
                        "y_fill_mode"     : "nearest",
                        "fill_value"      : 0
                }
        """
        x = np.asarray(x)

        # only support tf ordering
        orig_dim = x.ndim
        if x.ndim == 2:
                x = np.expand_dims(x,-1)

        img_row_axis = 0
        img_col_axis = 1
        channel_axis = 2

        ### ROTATION
        if T['rotation_range'] > 0:
                theta = np.pi / 180 * np.random.uniform(-T['rotation_range'],
                                        T['rotation_range'])
        else:
                theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        ### SHIFT HEIGHT
        if T['shift_range'][0] > 0:
                tx = np.random.uniform(-T['shift_range'][0], 
                        T['shift_range'][0]) * x.shape[img_row_axis]
        else:
                tx = 0
        ### SHIFT WIDTH
        if T['shift_range'][1] > 0:
                ty = np.random.uniform(-T['shift_range'][1], 
                        T['shift_range'][1]) * x.shape[img_col_axis]
        else:
                ty = 0
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        ### SHEAR
        if T['shear_range'] > 0:
                shear = np.random.uniform(-T['shear_range'],T['shear_range'])
        else:
                shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        ### ZOOM
        if T['zoom_range'][0] == 1. and T['zoom_range'][1] == 1.:
                zx, zy = 1, 1
        else:
                zx, zy = np.random.uniform(T['zoom_range'][0], T['zoom_range'][1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        ### COMBINE MATRICES INTO ONE TRANSFORM MATRIX
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                        translation_matrix),
                                        shear_matrix),
                                        zoom_matrix)
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        ### APPLY COMBINED TRANSFORM ON X IMAGE
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        #x = apply_transform(x, transform_matrix, channel_axis,
        #                       fill_mode=T['x_fill_mode'], fill_value=T['fill_value'])
        xs = np.dsplit(x,3)
        xs = [apply_transform(c, transform_matrix, channel_axis,fill_mode='constant',
                                fill_value=fv) 
          for c,fv in zip(xs,[255.0,0,255.0])]
        x = np.concatenate(xs, axis=2)
    
        ### HORIZONTAL FLIP
        if T['horizontal_flip'] == True:
                if np.random.random() < 0.5:
                        x = np.asarray(x).swapaxes(img_col_axis, 0)
                        x = x[::-1, ...]
                        x = x.swapaxes(0, img_col_axis)

        ### VERTICAL FLIP
        if T['vertical_flip']:
                if np.random.random() < 0.5:
                        x = np.asarray(x).swapaxes(img_row_axis, 0)
                        x = x[::-1, ...]
                        x = x.swapaxes(0, img_row_axis)


        if orig_dim == 2:
                x = np.squeeze(x)
                return Image.fromarray(x.astype(np.uint8))
        else:
                return Image.fromarray(x.astype(np.uint8))
