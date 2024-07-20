"""create a model from dataset"""
import os 
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pickle

from keras.utils import to_categorical

epoch = 30

datapath = "./data"
outputmodel = "./model/VideoClassificationModel.keras"
outputlabelbinarize = "./model/VideoClassificationBinarizer"

# High level variable
sports_labels = set(["circle", "square", "star", "triangle"])

# Print statement
print("Image is being loaded")

# Variable assignments
path_to_images = list(paths.list_images(datapath))
data = []
labels = []

# Loop over images in path_to_images
for image_path in path_to_images:
    # Split the image path using OS path separator
    label = image_path.split(os.path.sep)[-2]
    
    # Check if label is in sports_labels
    if label not in sports_labels:
        continue

    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Image at path {image_path} could not be loaded.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # Append image and label to data and label lists
        data.append(image)
        labels.append(label)
    except cv2.error as e:
        print(f"OpenCV error with image {image_path}: {e}")
    except ValueError as ve:
        print(ve)
    except Exception as ex:
        print(f"An unexpected error occurred with image {image_path}: {ex}")

data = np.array(data)
labels = np.array(labels)

# Hot encoded values 0,1,2
ib = LabelBinarizer()
labels = ib.fit_transform(labels)

# Split the data into train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

trainingAugementation = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

validationAugmentation = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainingAugementation.mean = mean
validationAugmentation.mean = mean

baseModel = ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(ib.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for basemodelLayers in baseModel.layers:
    basemodelLayers.trainable = False

opt = SGD(learning_rate=0.0001, momentum=0.9, decay=1e-4/epoch)

Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

History = model.fit(
    trainingAugementation.flow(X_train, Y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=validationAugmentation.flow(X_test, Y_test),
    validation_steps=len(X_test) // 32,
    epochs=epoch
)

model.save(outputmodel)
LabelBinarizer = open(outputlabelbinarize + ".pickle", "wb")
LabelBinarizer.write(pickle.dumps(ib))
LabelBinarizer.close()
