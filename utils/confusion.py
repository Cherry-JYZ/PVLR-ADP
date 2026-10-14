import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_pixel_confusion_matrix(gt_masks, pred_masks, num_classes, save_path='confusion_matrix.png'):

    gt_all = np.concatenate([m.flatten() for m in gt_masks])
    pred_all = np.concatenate([m.flatten() for m in pred_masks])
    cm = confusion_matrix(gt_all, pred_all, labels=np.arange(num_classes))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cm
