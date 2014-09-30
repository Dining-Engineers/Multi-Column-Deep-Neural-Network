from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.models.mlp import Layer, Linear
from pylearn2.models import Model
from pylearn2.space import CompositeSpace
import functools
from theano import printing
from theano.scalar import float64

wraps = functools.wraps
from theano.compat import OrderedDict



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

    def __init__(self, layer_name, gcn):
        super(PreprocessorBlock, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self._params = []

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        if hasattr(self, 'gcn'):
            gcn = float(self.gcn)
            X = global_contrast_normalize(state_below, scale=gcn)
        return X


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
            #     print "expected 4D tensor, got "
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
            for key, val in [('max_x.max_u',    v_max.max()),
                             ('max_x.mean_u',   v_max.mean()),
                             ('max_x.min_u',    v_max.min()),
                             ('min_x.max_u',    v_min.max()),
                             ('min_x.mean_u',   v_min.mean()),
                             ('min_x.min_u',    v_min.min()),
                             ('range_x.max_u',  v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u',  v_range.min()),
                             ('mean_x.max_u',   v_mean.max()),
                             ('mean_x.mean_u',  v_mean.mean()),
                             ('mean_x.min_u',   v_mean.min())]:
                rval[prefix+key] = val

        return rval