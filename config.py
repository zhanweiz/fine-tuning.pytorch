# Configuration File

# Base directory for data formats
#name = 'GURO_CELL'
#name = 'INBREAST'
name = 'pretraining_rot_n'
#name = 'pretraining_rot'
data_name = 'pretraining'
resume = True

data_base = '/home/cnnserver2/healthy_data/'+data_name
aug_base = '/home/cnnserver2/healthy_data/'+data_name
test_dir = '/home/cnnserver2/healthy_data/'+data_name+'/test/'

# model option
batch_size = 64
num_epochs = 400
lr_decay_epoch=50
feature_size = 100

# meanstd options
# INBREAST
#mean = [0.60335361908536667, 0.60335361908536667, 0.60335361908536667]
#std = [0.075116530817055119, 0.075116530817055119, 0.075116530817055119]

# GURO_EXTEND
#mean = [0.48359630772217554, 0.48359630772217554, 0.48359630772217554]
#std = [0.13613821516980551, 0.13613821516980551, 0.13613821516980551]

# GURO+INBREAST
mean = [0.51508365254458033, 0.51508365254458033, 0.51508365254458033]
std = [0.12719534902225299, 0.12719534902225299, 0.12719534902225299]
