import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pylearn2
from pylearn2.datasets.cifar10 import CIFAR10
import theano
from create_dataset import load_dataset
from utils import load_model_from_pkl, get_nparray_from_design_matrix, get_statistics, \
    get_nparray_from_design_matrix_b01c


def plot_confusion_matrix():
    y_ground_truth = np.float64(np.genfromtxt('csv/prediction_ground_truth.csv', delimiter=','))
    y_gcn_predictions = np.float64(np.argmax(np.genfromtxt('csv/prediction_gcn.csv', delimiter=','), axis=1))
    y_toronto_predictions = np.float64(np.argmax(np.genfromtxt('csv/prediction_toronto.csv', delimiter=','), axis=1))
    y_zca_predictions = np.float64(np.argmax(np.genfromtxt('csv/prediction_zca.csv', delimiter=','), axis=1))
    # y_multi_naive_predictions = np.float64(np.argmax(np.genfromtxt('csv/prediction_multicolumn_naive.csv', delimiter=','), axis=1))


    plot_single_cm(y_ground_truth, y_gcn_predictions, "Single GCN")
    plot_single_cm(y_ground_truth, y_toronto_predictions, "Single Toronto")
    plot_single_cm(y_ground_truth, y_zca_predictions, "Single ZCA")
    # plot_single_cm(y_ground_truth, y_multi_naive_predictions, "Multi Naive")


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
    plt.show()

    # plt.clf()
    # plt.savefig('./img/confusion'+name.replace(' ', '_')+'.png', dpi=150)


def get_mcdnn_predictions(model_pkl_url, dataset_list):

    # print "Loading gcn dataset.."
    # cifar10_gcn = CIFAR10(which_set='test',
    #                          gcn=55.,
    #                          axes=['b', 0, 1, 'c'])
    #
    # print "Loading torontoprepro dataset.."
    # cifar10_toronto = CIFAR10(which_set='test',
    #                          toronto_prepro=True,
    #                          axes=['b', 0, 1, 'c'])
    dataset_size = 10000
    batch_size = 128
    i = 0
    from pylearn2.utils import serial

    model = serial.load(model_pkl_url)

    dataset = load_dataset('test', dataset_list)
    it = dataset.iterator(mode='sequential', batch_size=2)
    a, b, c = it.next()

    print a.shape, b.shape, c.shape


    # loro
    inputs = model.get_input_space().make_theano_batch()
    theano_inputs = [inputs]
    f_model = theano.function(inputs, model.fprop(inputs), name='morte')







    # X1, X2 = model.get_input_space().make_theano_batch()
    # print model.get_input_space().make_theano_batch()
    # Y = model.fprop(model.get_input_space().make_theano_batch())
    # # get prediction
    # f_model = theano.function((X1, X2), Y)
    print model.layers
    print model.layers[0].layers
    y_predictions = np.zeros((dataset_size, 10))

    while i < dataset_size:
        batch_start = i
        batch_end = i+batch_size-1 if i+batch_size-1 < dataset_size-1 else dataset_size-1

        # x1_batch, y1_batch = get_nparray_from_design_matrix_b01c(cifar10_gcn, batch_start, batch_end)
        # x2_batch, y2_batch = get_nparray_from_design_matrix_b01c(cifar10_toronto, batch_start, batch_end)

        # x_batch, y_batch = get_nparray_from_design_matrix(column[0], 0, 127)


        y = f_model((a, b))


        y_predictions[batch_start:batch_end] = y # np.argmax(y, axis=1)
        print batch_start, ':', batch_end, '   ', get_statistics(y1_batch, y)
        i += batch_size

    # save predicition for this column ( still onehot)
    # with open('csv/prediction_'+key+'.csv', 'w') as file_handle:
    #     np.savetxt(file_handle, column[2], delimiter=',')
    # print "Column ", key
    y_ground_truth = np.float64(np.genfromtxt('csv/prediction_ground_truth.csv', delimiter=','))

    print "total\t ", get_statistics(y_ground_truth, y_predictions)


def get_all_mcdnn_predictions():

    dataset_list = ['gcn', 'toronto']
    get_mcdnn_predictions('pkl/best/multicolumn_2COL_GCN_TOR_best.pkl', dataset_list)
    # dataset_list = ['gcn', 'zca']
    # get_mcdnn_predictions('pkl/best/multicolumn_2COL_GCN_ZCA_best.pkl', dataset_list)
    # dataset_list = ['zca', 'toronto']
    # get_mcdnn_predictions('pkl/best/multicolumn_2COL_ZCA_TOR_best.pkl', dataset_list)
    # dataset_list = ['gcn', 'toronto', 'zca']
    # get_mcdnn_predictions('pkl/best/multicolumn_3COL_best.pkl', dataset_list)



if __name__ == '__main__':
    # plot_confusion_matrix()
    get_all_mcdnn_predictions()