import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix():
    y_ground_truth = np.int64(np.genfromtxt('prediction_ground_truth.csv', delimiter=','))
    y_gcn_predictions = np.argmax(np.genfromtxt('prediction_gcn.csv', delimiter=','), axis=1)
    y_toronto_predictions = np.argmax(np.genfromtxt('prediction_toronto.csv', delimiter=','), axis=1)
    y_zca_predictions = np.argmax(np.genfromtxt('prediction_zca.csv', delimiter=','), axis=1)


    print y_ground_truth[3:10]
    print y_gcn_predictions[3:10]

    cm = confusion_matrix(y_ground_truth, y_gcn_predictions, ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck'])


    plt.matshow(cm)
    plt.title("Cifar-10 Confusion Matrix")
    plt.colorbar()
    plt.ylabel('true labels')
    plt.xlabel('predicted labels')
    # plt.savefig('./confusion.png', dpi=150)
    plt.show()








if __name__ == '__main__':
    plot_confusion_matrix()