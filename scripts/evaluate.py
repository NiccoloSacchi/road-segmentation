""" All functions used to evaluate the predictions and show plots """

from sklearn.metrics import classification_report,confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from tf_aerial_images import make_img_overlay
from matplotlib.ticker import MaxNLocator

# TODO fix this function
def display_prediction(img, prediction, ax=None):
    """ Display predictions on the image. 
        img: original image. shape=(W, H, 3)
        prediction: prediction image (with restepct to the input image it is reduced by 
                a factor of n on each side). shape=(W/n, H/n)
    """
    def predictions_to_img(Y, size):
        """ From the matrix with the predicted patches build an image of the given size 
            size=(height, width). """
        # how many times we have to repeat h
        h_times = size[0]/Y.shape[0]
        # how many times we have to repeat w
        w_times = size[1]/Y.shape[1]

        if h_times%1 != 0 or w_times%1 != 0:
            print("Impossible to build an image of the given size: current shape is", Y.shape, "target shape is", size)
            return None

        gt_img = np.repeat(np.repeat(Y, h_times, axis=0), w_times, axis=1)
        return gt_img
    
    h, w = img.shape[0], img.shape[1] # 608, 608 for test set

    predicted_gt = predictions_to_img(prediction, (h, w))
    new_img = make_img_overlay(img, predicted_gt)

    if ax != None:
        ax.imshow(new_img)
        return
    
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(new_img)
    return
      
def evaluate_predictions(pred, true):
    """ Evaluate the predictions """

    target_names = ['0 (background)', '1 (road)']

    # print precision, recall, F1 score
    print(classification_report(true, pred, target_names=target_names))

    # Plotting the confusion matrix
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
            cm = np.apply_along_axis(lambda a: (a // 0.01)*0.01, 1, cm)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = cm.max() / 1.2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    # Compute confusion matrix
    cnf_matrix = (confusion_matrix(true, pred))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=target_names,
                          title='Confusion matrix')

    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                         title='Normalized confusion matrix')
    
def plot_history(history, last_epochs=-1):
    num_epochs = len(history['loss']) if last_epochs==-1 else last_epochs
    # visualizing losses and accuracy
    train_loss=history['loss'][-num_epochs:]
    val_loss=history['val_loss'][-num_epochs:]
    train_acc=history['acc'][-num_epochs:]
    val_acc=history['val_acc'][-num_epochs:]
    xc=range(len(history['loss']))[-num_epochs:]

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches((7, 14))
    axs[0].plot(xc,train_loss)
    axs[0].plot(xc,val_loss)
    axs[0].set_xlabel('num of Epochs')
    axs[0].set_ylabel('loss')
    axs[0].set_title('train_loss vs val_loss')
    axs[0].grid(True)
    axs[0].legend(['train','val'])
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[1].plot(xc,train_acc)
    axs[1].plot(xc,val_acc)
    axs[1].set_xlabel('num of Epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].set_ylim([0, 1])
    axs[1].set_title('train_acc vs val_acc')
    axs[1].grid(True)
    axs[1].legend(['train','val'],loc=4)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))