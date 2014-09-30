import numpy
import pylearn2
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.models.mlp import Layer, Linear
from pylearn2.models import Model
from pylearn2.space import CompositeSpace, Conv2DSpace
from pylearn2.datasets import preprocessing
import functools
from theano import printing
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano import shared, config, _asarray
from numpy import sqrt, prod, repeat, pi, exp, zeros, sum

wraps = functools.wraps
from theano.compat import OrderedDict

floatX = config.floatX


class PreprocessorBlock(Layer):
    """
    A Layer with no parameters that converts the input from
    one space to another.

    Parameters
    ----------
    layer_name : str
        Name of the layer.
    output_space : Space
        The space to convert to.
    """

    def __init__(self, layer_name, gcn=None, toronto=None):
        super(PreprocessorBlock, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self._params = []

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        self.output_space = Conv2DSpace(shape=[32, 32],
                                        num_channels=3,
                                        axes=('b', 0, 1, 'c'))


    # def lecun_lcn(self, X, kernel_size=7, threshold=1e-4, use_divisor=False):
    #     """
    #     Yann LeCun's local contrast normalization
    #     Orginal code in Theano by: Guillaume Desjardins
    #     """
    #
    #     filter_shape = (1, 1, kernel_size, kernel_size)
    #     filters = gaussian_filter(kernel_size).reshape(filter_shape)
    #     filters = shared(_asarray(filters, dtype=floatX), borrow=True)
    #
    #     convout = conv2d(X, filters=filters, filter_shape=filter_shape,
    #                      border_mode='full')
    #
    #     # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
    #     mid = int(numpy.floor(kernel_size / 2.))
    #     new_X = X - convout[:, :, mid:-mid, mid:-mid]
    #
    #     if use_divisor:
    #         # Scale down norm of kernel_sizexkernel_size patch
    #         sum_sqr_XX = conv2d(T.sqr(T.abs_(X)), filters=filters,
    #                             filter_shape=filter_shape, border_mode='full')
    #
    #         denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    #         per_img_mean = denom.mean(axis=[2, 3])
    #         divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
    #         divisor = T.maximum(divisor, threshold)
    #
    #         new_X /= divisor
    #
    #     return new_X  # T.cast(new_X, floatX)
    #
    # def local_mean_subtraction(self, X, kernel_size=5):
    #
    #     filter_shape = (1, 1, kernel_size, kernel_size)
    #     filters = mean_filter(kernel_size).reshape(filter_shape)
    #     filters = T.shared(_asarray(filters, dtype=floatX), borrow=True)
    #
    #     mean = conv2d(X, filters=filters, filter_shape=filter_shape,
    #                   border_mode='full')
    #     mid = int(numpy.floor(kernel_size / 2.))
    #
    #     return X - mean[:, :, mid:-mid, mid:-mid]


    def global_contrast_normalize(self, X, scale=1., subtract_mean=True,
                                  use_std=False, sqrt_bias=0., min_divisor=1e-8):

        """

        :rtype : object
        """
        ndim = X.ndim
        # print "DIO", type(X)
        if not ndim in [3, 4]: raise NotImplementedError("X.dim>4 or X.ndim<3")

        scale = float(scale)
        # mean = X.mean(axis=0)
        mean = X.mean(axis=ndim - 1)

        # print "AAA", mean.ndim #ndim - 1)
        new_X = X.copy()

        if subtract_mean:
            if ndim == 3:
                new_X = X - mean[:, :, None]
            else:
                # new_X = X - mean[None, :, :, :]
                new_X = X - mean[:, :, :, None]



        if use_std:
            normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim - 1)) / scale
            # normalizers = T.sqrt(sqrt_bias + X.var(axis=0)) / scale

        else:
            normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim - 1)) / scale
            # normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=0)) / scale

        # Don't normalize by anything too small.
        T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

        if ndim == 3:
            new_X /= normalizers[:, :, None]
        else:
            # new_X /= normalizers[None, :, :, :]
            new_X /= (normalizers[:, :, :, None] + min_divisor)

        return new_X

    def toronto_preproc(self, X):
        X /= 255.
        # x2 should be all dataset
        X2 = X.copy()
        X2 /= 255.
        return X - X2.mean(axis=0)


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = state_below.copy()
        # p.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
        # return(p)
        if self.toronto is not None:
            print "preprocessing: toronto"
            return self.toronto_preproc(p)

        if self.gcn is not None:
            return self.global_contrast_normalize(p)

        return p


class Average(Layer):
    """
    Monitoring channels are hardcoded for C01B batches
    """

    def __init__(self, layer_name):
        Model.__init__(self)
        self.__dict__.update(locals())
        del self.self
        self._params = []

    def set_input_space(self, space):
        self.input_space = space
        assert isinstance(space, CompositeSpace)
        self.output_space = space.components[0]

    def fprop(self, state_below):
        rval = state_below[0]
        # print "state below has:", len(state_below), " layers"
        for i in xrange(1, len(state_below)):
            rval = rval + state_below[i]
        rval.came_from_sum = True
        # average
        rval /= len(state_below)
        return rval

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = OrderedDict()

        if state is None:
            state = self.fprop(state_below)
        vars_and_prefixes = [(state, '')]

        for var, prefix in vars_and_prefixes:

            # print "average output: ", var.ndim, type(var)
            # if not hasattr(var, 'ndim') or var.ndim != 4:
            # print "expected 4D tensor, got "
            #     print var
            #     print type(var)
            #     if isinstance(var, tuple):
            #         print "tuple length: ", len(var)
            #     assert False
            v_max = var.max(axis=1)
            v_min = var.min(axis=1)
            v_mean = var.mean(axis=1)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())]:
                rval[prefix + key] = val

        return rval


def gaussian_filter(kernel_shape):
    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma ** 2
        return 1. / Z * exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / sum(x)


def mean_filter(kernel_size):
    s = kernel_size ** 2
    x = repeat(1. / s, s).reshape((kernel_size, kernel_size))
    return x