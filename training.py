from tensorflow.keras.optimizers import Adam
from core.data_loader import DataLoader
from core.resnet import build_resnet_model
from utils.callbacks import callbacks
from utils.misc_utils import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

depth = 47
batch_size = 64
epochs = 50

# TF GPU memory graph
limit_gpu()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print('[INFO]... ',len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

dl = DataLoader()

train_generator,xtest,ytest = dl.from_common_dir(
    directory='./Train',
    target_size=(32, 32),
    color_mode='rgb',
    batch_size=batch_size
)

model = build_resnet_model(
    input_shape=(32, 32, 3),
    depth=depth,
    num_classes=dl.num_classes
)

callbacks = callbacks(
    save_path='./assets/weights/microplastic',
    depth=depth
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(amsgrad=True, decay=0.001/epochs),
    metrics=['accuracy']
)


history = model.fit(
    x=train_generator,
    epochs=epochs,
    steps_per_epoch=int(dl.train_len/batch_size),
    callbacks=callbacks,
    validation_data=(xtest, ytest),
    validation_steps=int(dl.val_len/batch_size)
)

visualize(
    history=history.history,
    save_dir='./assets/logs_microplastic'
)


# Plot history: MAE
plt.title('Loss/Accuracy')
plt.ylabel('Value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

results = model.predict(xtest, batch_size=batch_size)
print(results)

# convert from class probabilities to actual class predictions
predicted_classes = np.argmax(results, axis=1)
print(predicted_classes)

# names of predicted classes
class_names = ["PS-B", "PS-R", "PS-O"]
labels = np.argmax(ytest, axis=1)
labels[1]
# Generate the confusion matrix
cnf_matrix = confusion_matrix(labels, predicted_classes)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
