import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np
from PIL import ImageFile

from sufflenet_model import ShuffleNetV2
from data import train_generator, val_generator


model = ShuffleNetV2(input_shape=(224, 224, 3),
					 num_classes=1000,
					 repetitions=(4, 8, 4),
					 initial_channels=24,
					 groups=3,
					 dropout_rate=0.5).build_model()

optimizer = Adam(learning_rate=0.0001)  # Lower learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    reduce_lr
]

model.summary()

print(train_generator.class_indices)
print(val_generator.class_indices)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights_dict,
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Define the path to save the model
save_path = 'D:/Projects/IIPP Projects/model_h5_format/model_scratch_1.h5'

# Save the trained model
model.save(save_path)