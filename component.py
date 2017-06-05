from utils import *


class Embed(object):
    def __init__(self, vocab_size, dim_embed, pre_matrix=None, pre_trained=False, prex=''):
        self.vocab_size = vocab_size
        self.dim_embed = dim_embed
        self.pre_matrix = pre_matrix
        self.pre_trained = pre_trained
        self.prex = pp(prex, 'Embed')
        self._init_params()

    def _init_params(self):
        if self.pre_trained:
            self.W = theano.shared(value=self.pre_matrix,
                                   name=pp(self.prex, 'embedding matrix'))
        else:
            self.W = init_param(param_type='embedding_matrix',
                                param_size=[self.vocab_size,
                                            self.dim_embed],
                                param_name=pp(self.prex, 'embedding matrix'))
        self.params = [self.W]

    def apply(self, index):
        embedded = self.W[index]
        return T.cast(embedded, 'float32')


class FC(object):
    def __init__(self, dim_in, dim_out, linear=False, prex=''):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = linear
        self.prex = pp(prex, 'FC')
        self._init_params()

    def _init_params(self):
        self.W = init_param(param_type='weight',
                            param_size=[self.dim_in, self.dim_out],
                            param_name=pp(self.prex, 'W'))
        self.b = init_param(param_type='bias',
                            param_size=self.dim_out,
                            param_name=pp(self.prex, 'b'))
        self.params = [self.W, self.b]

    def step_forward(self, x):
        e = T.dot(x, self.W) + self.b
        if self.linear:
            y = e
        else:
            y = T.nnet.sigmoid(e)
        return y


class Softmax(object):
    def __init__(self, dim_in, n_class, prex=''):
        self.dim_in = dim_in
        self.n_class = n_class
        self.prex = pp(prex, 'Softmax')
        self._init_params()

    def _init_params(self):
        self.W = init_param(param_type='weight',
                            param_size=[self.dim_in, self.n_class],
                            param_name=pp(self.prex, 'W'))
        self.b = init_param(param_type='bias',
                            param_size=[self.n_class],
                            param_name=pp(self.prex, 'b'))
        self.params = [self.W]

    def step_forward(self, x):
        e = T.dot(x, self.W) + self.b
        l = T.nnet.softmax(e)
        return l


class LSTM(object):
    def __init__(self, dim_in, dim_hid, embed=None, prex=''):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.prex = pp(prex, 'LSTM')
        self.embed = embed
        self._init_params()

    def _init_params(self):
        self.W_in = init_param(param_type='weight',
                               param_size=[4, self.dim_in, self.dim_hid],
                               param_name=pp(self.prex, 'W_in'))
        self.W_hid = init_param(param_type='weight',
                                param_size=[4, self.dim_hid, self.dim_hid],
                                param_name=pp(self.prex, 'W_hid'))
        self.b = init_param(param_type='bias',
                            param_size=[4, self.dim_hid],
                            param_name=pp(self.prex, 'bias'))
        self.params = [self.W_in, self.W_hid, self.b]

    def step_forward_emb(self, x, m, htm1, ctm1):
        x_embed = self.embed.apply(x)
        z = T.tanh(T.dot(x_embed, self.W_in[0]) + T.dot(htm1, self.W_hid[0]) + self.b[0])
        i = T.nnet.sigmoid(T.dot(x_embed, self.W_in[1]) + T.dot(htm1, self.W_hid[1]) + self.b[1])
        f = T.nnet.sigmoid(T.dot(x_embed, self.W_in[2]) + T.dot(htm1, self.W_hid[2]) + self.b[2])
        o = T.nnet.sigmoid(T.dot(x_embed, self.W_in[3]) + T.dot(htm1, self.W_hid[3]) + self.b[3])

        c = f * ctm1 + i * z
        c = m[:, None] * c + (1 - m)[:, None] * ctm1
        h = T.tanh(o * c)
        h = m[:, None] * h + (1 - m)[:, None] * htm1

        return T.cast(h, 'float32'), T.cast(c, 'float32')

    def step_forward(self, x, m, htm1, ctm1):
        z = T.tanh(T.dot(x, self.W_in[0]) + T.dot(htm1, self.W_hid[0]) + self.b[0])
        i = T.nnet.sigmoid(T.dot(x, self.W_in[1]) + T.dot(htm1, self.W_hid[1]) + self.b[1])
        f = T.nnet.sigmoid(T.dot(x, self.W_in[2]) + T.dot(htm1, self.W_hid[2]) + self.b[2])
        o = T.nnet.sigmoid(T.dot(x, self.W_in[3]) + T.dot(htm1, self.W_hid[3]) + self.b[3])

        c = f * ctm1 + i * z
        c = m[:, None] * c + (1 - m)[:, None] * ctm1
        h = T.tanh(o * c)
        h = m[:, None] * h + (1 - m)[:, None] * htm1

        return T.cast(h, 'float32'), T.cast(c, 'float32')


class BiLSTM(object):
    def __init__(self, embed, dim_hf_encode, prex=''):
        self.embed = embed
        self.dim_embed = self.embed.dim_embed
        self.dim_hf_encode = dim_hf_encode
        self.prex = pp(prex, 'bi-LSTM')
        self._init_params()

    def _init_params(self):
        self.l_lstm = LSTM(dim_in=self.dim_embed,
                           dim_hid=self.dim_hf_encode,
                           embed=self.embed,
                           prex=self.prex)

        self.r_lstm = LSTM(dim_in=self.dim_embed,
                           dim_hid=self.dim_hf_encode,
                           embed=self.embed,
                           prex=self.prex)

        self.params = self.embed.params + self.l_lstm.params + self.r_lstm.params

    def step_forward(self, id_sequence, mask_sequence):
        n_steps = id_sequence.shape[0]
        batch_size = id_sequence.shape[1]
        r_id_sequence = mask_sequence[::-1]
        r_mask_sequence = mask_sequence[::-1]
        init_out = [T.alloc(np.float32(0.), batch_size, self.dim_hf_encode),
                    T.alloc(np.float32(0.), batch_size, self.dim_hf_encode)]

        [l_h, _], _ = theano.scan(fn=self.l_lstm.step_forward_emb,
                                  sequences=[id_sequence, mask_sequence],
                                  outputs_info=init_out,
                                  n_steps=n_steps)

        [r_h, _], _ = theano.scan(fn=self.r_lstm.step_forward_emb,
                                  sequences=[r_id_sequence, r_mask_sequence],
                                  outputs_info=init_out,
                                  n_steps=n_steps)
        section_memory = T.concatenate([l_h, r_h], axis=2)
        return section_memory


class Conv2d(object):

    def __init__(self, fshape, filter_size, pool_size, cf, prex=''):
        self.fshape = fshape
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.cf = cf
        self.prex = pp(prex, 'Convolotion 2D')
        self._init_params()

    def _init_params(self):
        self.W = init_param(param_type='weight',
                            param_size=self.filter_size,
                            param_name=pp(self.prex, 'weight'))
        self.b = init_param(param_type='bias',
                            param_size=self.filter_size[0],
                            param_name=pp(self.prex, 'conv_bias'))
        self.params = [self.W, self.b]

    def step_forward(self, fmap):
        conv_out = T.nnet.conv2d(input=fmap,
                                 filters=self.W,
                                 filter_shape=self.filter_size,
                                 input_shape=self.fshape)
        pooled_out = pool.pool_2d(input=conv_out,
                                  ds=self.pool_size,
                                  ignore_border=True
                                  )
        y = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return y

    def simple_step_forward(self, fmap, image_size):
        conv_out = T.nnet.conv2d(input=fmap,
                                 filters=self.W,
                                 filter_shape=self.filter_size,
                                 input_shape=[1] + image_size)
        pooled_out = pool.pool_2d(input=conv_out,
                                  ds=self.pool_size,
                                  ignore_border=True
                                  )
        y = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return y
