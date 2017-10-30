# Configuration File

# Base directory for data formats
#name = 'GURO_CELL'
#name = 'INBREAST'
name = 'crop_rot_n'
#name = 'pretraining_rot'
data_name = 'pretraining_n'
resume = True

data_base = '/home/cnnserver2/healthy_data/'+data_name
aug_base = '/home/cnnserver2/healthy_data/'+data_name
test_dir = '/home/cnnserver2/healthy_data/'+data_name+'/test/'

scale_or_crop = 'crop'

# model option
batch_size = 64
num_epochs = 400
lr_decay_epoch=25
feature_size = 100

# meanstd options
# INBREAST
#mean = [0.60335361908536667, 0.60335361908536667, 0.60335361908536667]
#std = [0.075116530817055119, 0.075116530817055119, 0.075116530817055119]

# GURO_EXTEND
#mean = [0.48359630772217554, 0.48359630772217554, 0.48359630772217554]
#std = [0.13613821516980551, 0.13613821516980551, 0.13613821516980551]

# GURO+INBREAST
# mean = [0.51508365254458033, 0.51508365254458033, 0.51508365254458033]
# std = [0.12719534902225299, 0.12719534902225299, 0.12719534902225299]

# t512n
mean = [0.59959447, 0.39508969, 0.60466146] 
std = [0.21119411, 0.23215199, 0.23345646]
#Transformation parameters

class_weight = [0.25,1.0]
# class_weight = None

T = { 
	"rotation_range"  : 180,
	"shift_range"     : [0,0],
	"shear_range"     : 0,
	"zoom_range"      : [1,1],
	"horizontal_flip" : False,
	"vertical_flip"   : False,
	"x_fill_mode"     : "constant",
	"y_fill_mode"     : "nearest",
	"fill_value"      : 0
}
