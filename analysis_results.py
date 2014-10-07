import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pylearn2
from pylearn2.datasets.cifar10 import CIFAR10
import theano
from create_dataset import load_dataset
from utils import load_model_from_pkl, get_nparray_from_design_matrix, get_statistics, \
    get_nparray_from_design_matrix_b01c
from pylearn2.utils import serial


def analysis():

    y_ground_truth = np.float64(np.genfromtxt('csv_prediction/prediction_ground_truth.csv', delimiter=','))
    y_gcn_predictions = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_gcn.csv', delimiter=','), axis=1))
    y_toronto_predictions = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_toronto.csv', delimiter=','), axis=1))
    y_zca_predictions = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_zca.csv', delimiter=','), axis=1))
    y_multi_gcn_tor = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_gcn_toronto.csv', delimiter=','), axis=1))
    y_multi_gcn_zca = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_gcn_toronto.csv', delimiter=','), axis=1))
    y_multi_zca_tor = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_gcn_toronto.csv', delimiter=','), axis=1))
    y_multi_gcn_tor_zca = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_gcn_toronto.csv', delimiter=','), axis=1))
    y_multi_naive_gcn_tor = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_naive_gcn_toronto.csv', delimiter=','), axis=1))
    y_multi_naive_gcn_zca = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_naive_gcn_zca.csv', delimiter=','), axis=1))
    y_multi_naive_zca_tor = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_naive_zca_toronto.csv', delimiter=','), axis=1))
    y_multi_naive_gcn_tor_zca = np.float64(np.argmax(np.genfromtxt('csv_prediction/prediction_multicolumn_naive_gcn_toronto_zca.csv', delimiter=','), axis=1))

    print "Results _______________\t______________________"
    print " ______METHOD__________\t_____MEAN____VAR______"
    print "Single GCN             \t ", get_statistics(y_ground_truth, y_gcn_predictions)
    print "Single TOR             \t ", get_statistics(y_ground_truth, y_toronto_predictions)
    print "Single ZCA             \t ", get_statistics(y_ground_truth, y_zca_predictions)
    print "Multi-Naive GCN_TOR    \t ", get_statistics(y_ground_truth, y_multi_naive_gcn_tor)
    print "Multi GCN_TOR          \t ", get_statistics(y_ground_truth, y_multi_gcn_tor)
    print "Multi-Naive GCN_ZCA    \t ", get_statistics(y_ground_truth, y_multi_naive_gcn_zca)
    print "Multi GCN_ZCA          \t ", get_statistics(y_ground_truth, y_multi_gcn_zca)
    print "Multi-Naive ZCA_TOR    \t ", get_statistics(y_ground_truth, y_multi_naive_zca_tor)
    print "Multi ZCA_TOR          \t ", get_statistics(y_ground_truth, y_multi_zca_tor)
    print "Multi-Naive GCN_TOR_ZCA\t ", get_statistics(y_ground_truth, y_multi_naive_gcn_tor_zca)
    print "Multi GCN_TOR_ZCA      \t ", get_statistics(y_ground_truth, y_multi_gcn_tor_zca)
    print "_______________________________________"

    plot_single_cm(y_ground_truth, y_gcn_predictions, "Single GCN")
    plot_single_cm(y_ground_truth, y_toronto_predictions, "Single Toronto")
    plot_single_cm(y_ground_truth, y_zca_predictions, "Single ZCA")
    plot_single_cm(y_ground_truth, y_multi_gcn_tor, "Multi GCN_TOR")
    plot_single_cm(y_ground_truth, y_multi_naive_gcn_tor, "Multi-Naive GCN_TOR")
    plot_single_cm(y_ground_truth, y_multi_gcn_zca, "Multi GCN_ZCA")
    plot_single_cm(y_ground_truth, y_multi_naive_gcn_zca, "Multi-Naive GCN_ZCA")
    plot_single_cm(y_ground_truth, y_multi_zca_tor, "Multi ZCA_TOR")
    plot_single_cm(y_ground_truth, y_multi_naive_zca_tor, "Multi-Naive ZCA_TOR")
    plot_single_cm(y_ground_truth, y_multi_gcn_tor_zca, "Multi GCN_TOR_ZCA")
    plot_single_cm(y_ground_truth, y_multi_naive_gcn_tor_zca, "Multi-Naive GCN_TOR_ZCA")


def plot_single_cm(ytrue, ypred, name):

    cm = confusion_matrix(ytrue, ypred)
    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)


    # cm /= cm.sum(axis=1)*100
    cm = cm / cm.astype(np.float).sum(axis=1)
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
    # plt.show()

    # plt.clf()
    plt.savefig('./img/confusion'+name.replace(' ', '_')+'.png', dpi=150)


def get_mcdnn_predictions(model_pkl_url, dataset_list):

    dataset_size = 10000
    batch_size = 128
    model = serial.load(model_pkl_url)

    dataset = load_dataset('test', dataset_list)
    it = dataset.iterator(mode='sequential', batch_size=128)
    # loro
    inputs = model.get_input_space().make_theano_batch()
    assert len(inputs) == 4 or len(inputs) == 3
    f_model = theano.function(inputs, model.fprop(inputs), name='morte')
    # where to save the predictions
    y_predictions = np.zeros((dataset_size, 10))

    print len(inputs), inputs
    i = 0
    try:
        while 1:

            batch_start = i
            batch_end = i+batch_size if i+batch_size < dataset_size else dataset_size


            if len(inputs) == 3:
                x1_batch, x2_batch, y_batch = it.next()
                y = f_model(x1_batch, x2_batch)
            else:
                x1_batch, x2_batch, x3_batch, y_batch = it.next()
                y = f_model(x1_batch, x2_batch, x3_batch)


            y_predictions[batch_start:batch_end] = y

            # print batch_start, ':', batch_end, '   ', get_statistics(np.argmax(y_batch, axis=1), y)
            i += batch_size
    except StopIteration:
        pass

    # save predicition for this column ( still onehot)
    with open('csv_prediction/prediction_multicolumn_'+"_".join(dataset_list)+'.csv', 'w') as file_handle:
        np.savetxt(file_handle, y_predictions, delimiter=',')
    # print "Column ", key
    y_ground_truth = np.float64(np.genfromtxt('csv_prediction/prediction_ground_truth.csv', delimiter=','))

    print "total\t ", get_statistics(y_ground_truth, y_predictions)


def get_all_mcdnn_predictions():

    # dataset_list = ['gcn', 'toronto']
    # get_mcdnn_predictions('pkl/best/multicolumn_2COL_GCN_TOR_best.pkl', dataset_list)
    # dataset_list = ['gcn', 'zca']
    # get_mcdnn_predictions('pkl/best/multicolumn_2COL_GCN_ZCA_best.pkl', dataset_list)
    # dataset_list = ['zca', 'toronto']
    # get_mcdnn_predictions('pkl/best/multicolumn_2COL_ZCA_TOR_best.pkl', dataset_list)
    dataset_list = ['gcn', 'toronto', 'zca']
    get_mcdnn_predictions('pkl/best/multicolumn_3COL_best.pkl', dataset_list)



if __name__ == '__main__':
    # plot_confusion_matrix()
    get_all_mcdnn_predictions()