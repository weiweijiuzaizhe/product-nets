import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
import tensorflow as tf
import utils

dtype = utils.DTYPE


class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self, fetches, X=None, y=None, mode='train'):
            feed_dict = {}
            if type(self.X) is list:
                for i in range(len(X)):
                    feed_dict[self.X[i]] = X[i]
            else:
                feed_dict[self.X] = X
            if y is not None:
                feed_dict[self.y] = y
            if self.layer_keeps is not None:
                if mode == 'train':
                    feed_dict[self.layer_keeps] = self.keep_prob_train
                elif mode == 'test':
                    feed_dict[self.layer_keeps] = self.keep_prob_test
            return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)


class LR(Model):
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2, l2_weight=0,
                 random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w = self.vars['w']
            b = self.vars['b']
            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(xw + b, [-1])
            self.y_prob = tf.sigmoid(logits)
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)) + \
                        l2_weight * tf.nn.l2_loss(xw)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'tnormal', dtype),
                     ('v', [input_dim, factor_order], 'tnormal', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']
            """
            SparseTensor(values=[1, 2], indices=[[0, 0], [1, 2]], shape=[3, 4])
            [[1, 0, 0, 0]
            [0, 0, 2, 0]
            [0, 0, 0, 0]]
            http://www.jianshu.com/p/c233e09d2f5f
            """
            # 得到x*x的张量
            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, v))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(v)), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, w)

            logits = tf.reshape(xw + b + p, [-1])    # 预测出的目标值
            self.y_prob = tf.sigmoid(logits)         #

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class FNN(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        print(layer_sizes)     #  [[25, 439623, 36, 371, 4, 11029, 39491, 12, 7, 5, 4, 12, 2, 36, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8], 10, 1]

        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]

        for i in range(num_inputs):   # 初始化w,b  ,遍历从 0 到 num_input-1
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))  # w0_1,w0_2,...,w0_{num_input-1}
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))

        #以上是第一层
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))

        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))   # 下面就是i的下标
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))

        self.graph = tf.Graph()

        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)   # 'drop_out': [0, 0],
            self.keep_prob_test = np.ones_like(drop_out)    # Return an array of ones with the same shape and type as a given array.
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)   # 将之前初始化的那些参数都变成tf中的层
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]   # 返回一个list
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)  # 在维度1上进行连接
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),  # def activate(weights, activation_function):
                self.layer_keeps[0])

            for i in range(1, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(                 # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
                    utils.activate(                # def activate(weights, activation_function)
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.reshape(l, [-1])                # A Tensor of the same shape of x  ,相当于打散为1维
            self.y_prob = tf.sigmoid(l)            # 由此得到y,之后可以进行计算了

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]  # 整个网络中的w都会被加入正则化
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True   #使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

# Convolutional Click Prediction Model  CCPM
class CCPM(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        embedding_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = embedding_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('f1', [embedding_order, layer_sizes[2], 1, 2], 'tnormal', dtype))
        init_vars.append(('f2', [embedding_order, layer_sizes[3], 2, 2], 'tnormal', dtype))
        init_vars.append(('w1', [2 * 3 * embedding_order, 1], 'tnormal', dtype))
        init_vars.append(('b1', [1], 'zero', dtype))

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            l = tf.nn.dropout(
                utils.activate(
                    tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) + b0[i]
                               for i in range(num_inputs)], 1),
                    layer_acts[0]),
                self.layer_keeps[0])
            l = tf.transpose(tf.reshape(l, [-1, num_inputs, embedding_order, 1]), [0, 2, 1, 3])
            f1 = self.vars['f1']
            l = tf.nn.conv2d(l, f1, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]),
                    num_inputs / 2),
                [0, 1, 3, 2])
            f2 = self.vars['f2']
            l = tf.nn.conv2d(l, f2, [1, 1, 1, 1], 'SAME')
            l = tf.transpose(
                utils.max_pool_4d(
                    tf.transpose(l, [0, 1, 3, 2]), 3),
                [0, 1, 3, 2])
            l = tf.nn.dropout(
                utils.activate(
                    tf.reshape(l, [-1, embedding_order * 3 * 2]),
                    layer_acts[1]),
                self.layer_keeps[1])
            w1 = self.vars['w1']
            b1 = self.vars['b1']
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1,
                    layer_acts[2]),
                self.layer_keeps[2])

            l = tf.reshape(l, [-1])
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)

# Product-based Neural Networks for User Response Prediction
class PNN1(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, kernel_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [num_inputs, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                self.layer_keeps[0])

            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            p = tf.reduce_sum(
                tf.reshape(
                    tf.matmul(
                        tf.reshape(
                            tf.transpose(
                                tf.reshape(l, [-1, num_inputs, factor_order]),
                                [0, 2, 1]),
                            [-1, num_inputs]),
                        k1),
                    [-1, factor_order, layer_sizes[2]]),
                1)
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + b1 + p,
                    layer_acts[1]),
                self.layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.reshape(l, [-1])
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class PNN2(Model):
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, kernel_l2=None, init_path=None,
                 opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(layer_sizes[0])
        factor_order = layer_sizes[1]
        for i in range(num_inputs):
            layer_input = layer_sizes[0][i]
            layer_output = factor_order
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('k1', [factor_order * factor_order, layer_sizes[2]], 'tnormal', dtype))
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))
        for i in range(2, len(layer_sizes) - 1):
            layer_input = layer_sizes[i]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal',))
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]
            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)
            l = tf.nn.dropout(
                utils.activate(x, layer_acts[0]),
                self.layer_keeps[0])
            w1 = self.vars['w1']
            k1 = self.vars['k1']
            b1 = self.vars['b1']
            z = tf.reduce_sum(tf.reshape(l, [-1, num_inputs, factor_order]), 1)
            p = tf.reshape(
                tf.matmul(tf.reshape(z, [-1, factor_order, 1]),
                          tf.reshape(z, [-1, 1, factor_order])),
                [-1, factor_order * factor_order])
            l = tf.nn.dropout(
                utils.activate(
                    tf.matmul(l, w1) + tf.matmul(p, k1) + b1,
                    layer_acts[1]),
                self.layer_keeps[1])

            for i in range(2, len(layer_sizes) - 1):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.reshape(l, [-1])
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            if kernel_l2 is not None:
                self.loss += kernel_l2 * tf.nn.l2_loss(k1)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)


class deepFM(Model): 
    def __init__(self, layer_sizes=None, layer_acts=None, drop_out=None, layer_l2=None, init_path=None, opt_algo='gd',
                 learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        print(layer_sizes)                  # 形如 [[25, 439623, 36, 371, 4, 11029, 39491, 12, 7, 5,8], 10, 1]
        init_vars = []
        num_inputs = len(layer_sizes[0])    # 维度的类别
        factor_order = layer_sizes[1]       # k

        for i in range(num_inputs):          # 初始化w,b  ,遍历从 0 到 num_input - 1,num_inputs是field数目
            layer_input = layer_sizes[0][i]  # 这里输入的是one_hot之后的编码,是one_hot的数目
            layer_output = factor_order      # 每个field都被映射到了k个维度上
            init_vars.append(('w0_%d' % i, [layer_input, layer_output], 'tnormal', dtype))  # w0_1,w0_2,...,w0_{num_input-1}
            init_vars.append(('b0_%d' % i, [layer_output], 'zero', dtype))

        # 以上是第一层,产生的是一个embedding
        init_vars.append(('w1', [num_inputs * factor_order, layer_sizes[2]], 'tnormal', dtype))  # 直接和embedding层连接的那一层
        init_vars.append(('b1', [layer_sizes[2]], 'zero', dtype))

        # 原来的隐层
        for i in range(2, len(layer_sizes) - 1):  # 减一是因为后面layer_output的下标是 i + 1
            layer_input = layer_sizes[i]   # 这里最开始的输入是layer_size[2]
            layer_output = layer_sizes[i + 1]
            init_vars.append(('w%d' % i, [layer_input, layer_output], 'tnormal', dtype))   # 下面就是i的下标  # for var_name, var_shape, init_method, dtype in init_vars
            init_vars.append(('b%d' % i, [layer_output], 'zero', dtype))   #

        # FM的二阶层对应的变量值
        #for i in range(num_inputs):
        #    for j in range(i, num_inputs):
        #        init_vars.append(('fm_ww_%d_%d' % (i, j), [factor_order], 'tnormal', dtype))

        # FM的一阶层对应的变量值
        for i in range(num_inputs):
            init_vars.append(('fm_w_%d' % i, [layer_sizes[0][i],1], 'tnormal', dtype))


        self.graph = tf.Graph()

        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]  # X中有num_inputs个sparse_placeholder
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)   # 'drop_out': [0, 0],
            self.keep_prob_test = np.ones_like(drop_out)    # Return an array of ones with the same shape and type as a given array.
            self.layer_keeps = tf.placeholder(dtype)
            self.fm_ww = tf.placeholder(dtype)
            self.fm_w = tf.placeholder(dtype)

            self.vars = utils.init_var_map(init_vars, init_path)      # 至此,self.vars作为一个dict,里面存放的东西:name=>tensor
            w0 = [self.vars['w0_%d' % i] for i in range(num_inputs)]  # w0是一个list,元素是tensor,每个tensor是 input * k
            b0 = [self.vars['b0_%d' % i] for i in range(num_inputs)]  # b0是一个list ,维度是 1*k
            xw = [tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)]  # x的维度是1*input,返回一个list,内部的是tf的tensor,xw内部元素维度是1*k

            x = tf.concat([xw[i] + b0[i] for i in range(num_inputs)], 1)  # 这些tensor在维度1上进行连接,这里是把一堆的tensor变为一维的tensor,至此已经变为embedding的输出了

            fm_xx = [tf.reduce_sum(x[i]*x[j]) for i in range(num_inputs) for j in range(i, num_inputs)]  # FM的二阶值,一个list,每个元素是 k 维度,
            fm_x = [tf.sparse_tensor_dense_matmul(self.X[i] * self.vars['fm_w_%d' % i]) for i in range(num_inputs)]  # FM的二阶值

            fm_xx_con = tf.concat([fm_xx[i*num_inputs + j] for i in range(num_inputs) for j in range(i, num_inputs)], 1) # fm_xx[i*num_inputs + j] 是一个值
            fm_x_con = tf.concat([fm_x[i] for i in range(num_inputs)], 1)

            fm_xx_con = tf.reshape(fm_xx_con, [-1])
            fm_x_con = tf.reshape(fm_x_con, [-1])

            l = tf.nn.dropout(                     #
                utils.activate(x, layer_acts[0]),  # def activate(weights, activation_function):    'layer_acts': ['tanh', 'none'], 返回一个tensor
                self.layer_keeps[0])               # With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0

            for i in range(1, len(layer_sizes) - 1): #l2以及后面的hidden layer
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                l = tf.nn.dropout(                 # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
                    utils.activate(                # def activate(weights, activation_function)   定义激活函数,加入计算图
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.concat([1, fm_xx_con, fm_x_con], -1)  #进入l之前进行整合
            l = tf.reshape(l, [-1])                # A Tensor of the same shape of x  ,相当于打散为1维





            self.y_prob = tf.sigmoid(l)            # 由此得到y,之后可以进行计算了

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                # for i in range(num_inputs):
                self.loss += layer_l2[0] * tf.nn.l2_loss(tf.concat(xw, 1))
                for i in range(1, len(layer_sizes) - 1):
                    wi = self.vars['w%d' % i]  # 整个网络中的w都会被加入正则化
                    # bi = self.vars['b%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True   #使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)



