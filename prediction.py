import numpy as np
from imutils import paths
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import pandas as pd
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# load ResNet model

model = load_model('./assets/weights/microplastic/ResNet47-epoch.hdf5') # path to trained model

# plot structure of ResNet

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# make predictions

results = []
image_paths = list(paths.list_images('./Test/Mixed-HSF')) # path to experimental dataset
data = np.array([np.array(load_img(image_path, target_size=(32, 32), color_mode='rgb')) for image_path in image_paths]) / 255.0
print(data)

prediction = model.predict_on_batch(data)
labels = np.argmax(prediction, axis=1)

# save results

submission = pd.DataFrame({'file': image_paths, 'label': labels})
submission.to_csv("./results/result.csv", index=False)

# confusion matrix building
# load csv with true labels

true_labels = pd.read_csv('./labels.csv', delimiter=';', index_col = "file")
submission.set_index('file', drop=True, inplace=True)
confusion = pd.concat([submission, true_labels], axis=1)

cm = confusion_matrix(y_true=confusion['true'], y_pred=confusion['label'])

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['PS-B','PS-R', 'PS-O']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.show()
