import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import pickle

def draw_text_in_dataset(ax, text):
    at = AnchoredText(text,
                      loc=2, prop=dict(size=32), frameon=True,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


def plot_dataset_with_label(X, y, max_elements=None):

    count = X.shape[0]
    
    if not max_elements is None:
        count = min(max_elements, count)

    rows = int(count / 4) + 1

    fig = plt.figure(figsize=(40., rows * 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, 4),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, (label, img) in zip(grid, [(y[i], X[i][:,:,0]) for i in range(count)]):
        ax.imshow(img)
        draw_text_in_dataset(ax, label)
        
    plt.show()

def plot_dataset(dataset_path, max_elements=None):

    with open(dataset_path, "rb") as file:
        X, y = pickle.load(file)
    
    subjects = np.array(list(X.keys()))
    
    count = subjects.shape[0]
    
    if not max_elements is None:
        count = min(max_elements, count)

    rows = int(count / 4) + 1

    fig = plt.figure(figsize=(40., rows * 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, 4),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, (subject, im) in zip(grid, [(subject, X[subject][0,:,:,0]) for subject in subjects[:count]]):
        ax.imshow(im)
        draw_text_in_dataset(ax, subject)
        # ax.imshow(im_roi, alpha=0.5)

    plt.show()
    
"""
Plot result of a model's history
"""
class ResultPlotter:
    
    def __init__(self, model, loss_name="loss", accuracy_name="acc"):
        self.model = model
        self.loss_name = loss_name
        self.accuracy_name = accuracy_name
        
    def plot(self):
        plt = self._prepare()
        plt.show()
        
        
    def save(self, output_name):
        plt = self._prepare()
        plt.savefig(output_name)
    
    def _prepare(self):
        acc = self.model.history.history[self.accuracy_name]
        val_acc = self.model.history.history[f'val_{self.accuracy_name}']

        loss = history.history[self.loss_name]
        val_loss = history.history[f"val_{self.loss_name}"]

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        # plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        return plt
       