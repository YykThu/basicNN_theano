from envorinment import *


# graph_building_related
# # initialize params
def init_param(param_type, param_size, param_name):
    if param_type == 'weight':
        dim_in = param_size[-1]
        param = theano.shared(value=np.random.uniform(low=-1/np.sqrt(dim_in),
                                                      high=1/np.sqrt(dim_in),
                                                      size=param_size).astype(dtype=floatX),
                              name=param_name)
    elif param_type == 'bias':
        param = theano.shared(value=np.zeros(shape=param_size).astype(dtype=floatX),
                              name=param_name)
    elif param_type == 'embedding_matrix':
        param = theano.shared(value=np.random.randn(param_size[0], param_size[1]).astype(dtype=floatX))
    else:
        raise TypeError('wrong param type')

    return param


# # concat names
def pp(prex, name):
    return '{prex}_{name}'.format(prex=prex, name=name)


# data_processing_related
# # convert to floatX
def convert_float32(data):
    return np.array(data).astype(dtype=floatX)


# # convert to int32
def convert_int32(data):
    return np.array(data).astype(dtype=np.int32)


# training_related
# # rms_prop optimizer
def rms_prop(cost, params, learning_rate, rho, epsilon):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, T.cast(acc_new, 'float32')))
        updates.append((p, T.cast(p - learning_rate * g, 'float32')))
    return updates
