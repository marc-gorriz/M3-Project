DO_TRAIN = False
DO_TEST = True

# Data generation
DATASET_DIR = '../../Databases/MIT_split'
CLASSES = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

DATA_SIZE = 256
N_TRAIN = 1881
N_TEST = 807

IMG_SIZE = 32
BATCH_SIZE = 16

# Features extraction
INPUT_PATH = '../../outputs/DL1-OUTPUT-0004/train1/'
FEATURES_MODEL = INPUT_PATH + 'weights.h5'
#CODEBOOK_PATH = INPUT_PATH + 'codebook.pkl'

FEATURES_EXTRACTOR = 'bow'  # simple, bow

PATCH_SIZE = 32
N_PATCHES = 48

LAYER_FEATURES_GENERATOR = 'hidden1'
BOW_K = 512

# Classifiers
CLASSIFIER = 'mlp'  # svm, mlp
EPOCHS = 80
