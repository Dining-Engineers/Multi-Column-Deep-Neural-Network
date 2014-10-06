from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils import string_utils
from utils import *


class MCDNN():

    def __init__(self, models):
        self.n_column = len(models)
        self.columns = models
        # get predictions from first model since every model share the same dataset
        self.y_ground_truth = models[models.keys()[0]][0].y.T[0]
        with open('prediction_ground_truth.csv', 'w') as file_handle:
                np.savetxt(file_handle, self.y_ground_truth, delimiter=',')
        # Cifar10 has 10000 img in test set for 10 classes
        self.dataset_size = 10000
        self.n_classes = 10
        self.batch_size = 128
        self.y_predictions = np.zeros((self.dataset_size, self.n_classes))

    def get_columns_predictions(self):
        for key, column in self.columns.iteritems():
            i = 0
            model = load_model_from_pkl(column[1])
            while i < self.dataset_size:
                batch_start = i
                batch_end = i+self.batch_size-1 if i+self.batch_size-1 < self.dataset_size-1 else self.dataset_size-1

                x_batch, y_batch = get_nparray_from_design_matrix(column[0], batch_start, batch_end)
                # x_batch, y_batch = get_nparray_from_design_matrix(column[0], 0, 127)

                f_model = model
                y = f_model(x_batch)

                column[2][batch_start:batch_end] = y # np.argmax(y, axis=1)
                print batch_start, ':', batch_end, '   ', get_statistics(y_batch, y)
                i += self.batch_size

            # save predicition for this column ( still onehot)
            with open('prediction_'+key+'.csv', 'w') as file_handle:
                np.savetxt(file_handle, column[2], delimiter=',')
            print "Column ", key
            print "\t ", get_statistics(self.y_ground_truth, column[2])

    def get_mcdnn_predictions(self):

        self.y_predictions = np.zeros((self.dataset_size, self.n_classes))

        for key, column in self.columns.iteritems():
            self.y_predictions += column[2]

        self.y_predictions /= self.n_column

        with open('prediction_multicolumn_naive.csv', 'w') as file_handle:
            np.savetxt(file_handle, self.y_predictions, delimiter=',')
        print "MCDNN results: "
        print "\t ", get_statistics(self.y_ground_truth, self.y_predictions)


if __name__ == '__main__':
    # get dataset CIFAR10

    print "Loading gcn dataset.."
    cifar10_gcn = CIFAR10(which_set='test',
                             gcn=55.,
                             axes=['c', 0, 1, 'b'])
    print "Loading torontoprepro dataset.."
    cifar10_toronto = CIFAR10(which_set='test',
                             toronto_prepro=True,
                             axes=['c', 0, 1, 'b'])

    print "Loading zca dataset.."
    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar10')
    cifar10_zca = ZCA_Dataset(preprocessed_dataset=serial.load(data_dir+"/pylearn2_gcn_whitened/test.pkl"),
                              preprocessor=serial.load(data_dir+"/pylearn2_gcn_whitened/preprocessor.pkl"),
                              axes=['c', 0, 1, 'b'])

    columns = {
        'gcn': (cifar10_gcn, 'pkl/best/singlecolumn_complex_GCN_paper_best.pkl', np.zeros((10000, 10))),
        'toronto': (cifar10_toronto, 'pkl/best/singlecolumn_complex_TORONTO_paper_best.pkl', np.zeros((10000, 10))),
        'zca': (cifar10_zca, 'pkl/best/singlecolumn_complex_ZCA_paper_best.pkl', np.zeros((10000, 10)))
    }

    multi_column_dnn = MCDNN(columns)
    multi_column_dnn.get_columns_predictions()
    multi_column_dnn.get_mcdnn_predictions()

