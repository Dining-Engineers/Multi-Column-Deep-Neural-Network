import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix():
    y_ground_truth = np.float64(np.genfromtxt('prediction_ground_truth.csv', delimiter=','))
    y_gcn_predictions = np.float64(np.argmax(np.genfromtxt('prediction_gcn.csv', delimiter=','), axis=1))
    y_toronto_predictions = np.float64(np.argmax(np.genfromtxt('prediction_toronto.csv', delimiter=','), axis=1))
    y_zca_predictions = np.float64(np.argmax(np.genfromtxt('prediction_zca.csv', delimiter=','), axis=1))

    plot_cm(y_ground_truth, y_gcn_predictions, "Single GCN")
    plot_cm(y_ground_truth, y_toronto_predictions, "Single Toronto")
    plot_cm(y_ground_truth, y_zca_predictions, "Single ZCA")



def plot_cm(ytrue, ypred, name):



    cm = confusion_matrix(ytrue, ypred)


    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure()
    plt.title("Cifar-10 Confusion Matrix - "+name)


    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), labels, rotation=45)
    plt.yticks(range(height), labels)
    # plt.savefig('confusion_matrix.png', format='png')
    plt.ylabel('true labels')
    plt.xlabel('predicted labels')
    plt.show()

    # plt.clf()
    # # plt.savefig('./confusion.png', dpi=150)


if __name__ == '__main__':
    plot_confusion_matrix()