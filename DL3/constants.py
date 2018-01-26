global_path = "output/"

# Database
train_data_dir = '../../Databases/MIT_split_validation/train'
val_data_dir = '../../Databases/MIT_split_validation/validation'
test_data_dir = '../../Databases/MIT_split_validation/test'

img_width = 224
img_height = 224

nb_train = 1681
nb_validation = 520
nb_test = 487

augmentation_increment = 20

# Training parameters
cnn_model = "cnns_model"

number_of_epoch = 15
batch_size = 32

loss = 'categorical_crossentropy'
optimizer = 'Adam'

regularization = 0.0005 #0.1 (train0, train1, train2)
batch_normalization = True
LRN2D_norm = True
alpha = 0.0001
beta = 0.75

dropout = 0
stddev = 0

early_patience = 3
early_delta = 0
