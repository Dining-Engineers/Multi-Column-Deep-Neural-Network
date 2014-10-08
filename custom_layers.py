import numpy
import pylearn2
from pylearn2.datasets.preprocessing import ZCA
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


class PretrainedMLP(Layer):
    """
    A layer whose weights are initialized, and optionally fixed,
    based on prior training.

    Parameters
    ----------
    layer_content : Model
        Should implement "upward_pass" (RBM and Autoencoder do this)
    freeze_params: bool
        If True, regard layer_conent's parameters as fixed
        If False, they become parameters of this layer and can be
        fine-tuned to optimize the MLP's cost function.
    """

    def __init__(self, layer_name, layer_content, freeze_params=True):
        super(PretrainedMLP, self).__init__()
        self.__dict__.update(locals())
        # model = layer_content
        # # X = model.get_input_space().make_theano_batch()
        # # Y = model.fprop(X)
        # # self.pretrained_fprop = theano.function([X], Y)

        del self.self

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        # print self.get_input_space()
        # print space
        assert self.get_input_space() == space

        # self.input_space = space.components[0]


    @wraps(Layer.get_params)
    def get_params(self):

        if self.freeze_params:
            return []
        return self.layer_content.get_params()

    @wraps(Layer.get_input_space)
    def get_input_space(self):

        return self.layer_content.get_input_space()

    @wraps(Layer.get_output_space)
    def get_output_space(self):

        return self.layer_content.get_output_space()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        return OrderedDict([])

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        # get prediction
        return self.layer_content.layers[-2].fprop(state_below) #self.layer_content.upward_pass(state_below)


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
        print space
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
        self.output_space = space

        # Conv2DSpace(shape=[32, 32],
        #                                 num_channels=3,
        #                                 axes=('b', 0, 1, 'c'))


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
            print "preproc gcn"
            return self.global_contrast_normalize(p, scale=self.gcn)

        return p


class SpaceConverter2(Layer):
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

    def __init__(self, layer_name, output_space):
        super(SpaceConverter2, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self._params = []

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        print space
        self.input_space = space

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        print state_below
        return self.input_space.format_as(state_below, self.output_space)