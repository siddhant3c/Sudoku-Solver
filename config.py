### IMAGE PREPROCESSING HYPERPARAMETERS

IMG_PATH = r"Images\sudoku_img.jpeg"
IMG_HEIGHT = 450
IMG_WIDTH = 450

### MODEL TRAINING HYPERPARAMETERS

# NUM_EPOCHS = 12
# BATCH_SIZE = 1000
MODEL_PATH = r"\models\my_model_cnn.h5"

path = r"Dataset"
testRatio = 0.2
valRatio = 0.2
imageDimensions = (28, 28, 3)

batchSizeVal = 50
epochsVal = 50
stepsPerEpochVal = 6534//batchSizeVal