DO_TRAIN = True
DO_TEST = False

# Data generation
DATASET_DIR = '../../Databases/MIT_split'
CLASSES = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

DATA_SIZE = 256
N_TRAIN = 1881
N_TEST = 807

IMG_SIZE = 32
BATCH_SIZE = 16

# Features extraction
INPUT_PATH = '../../DL1-OUTPUT/train/train1/'
FEATURES_MODEL = INPUT_PATH + 'weights.h5'
CODEBOOK_PATH = INPUT_PATH + 'codebook.pkl'

FEATURES_EXTRACTOR = 'simple'  # simple, bow

PATCH_SIZE = 32
N_PATCHES = 12

LAYER_FEATURES_GENERATOR = 'hidden2'
BOW_K = 32

# Classifiers
CLASSIFIER = 'svm'  # svm, mlp
EPOCHS = 100
