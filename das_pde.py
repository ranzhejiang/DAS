from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

import os
import shutil
import time

class dataflow(object):
    def __init__(self, x, buffersize, batchsize, y=None):
        self.x = x
        self.y = y
        self.buffersize = buffersize
        self.batchsize = batchsize

        if y is not None:
            dx = tf.data.Dataset.from_tensor_slices(x)
            dy = tf.data.Dataset.from_tensor_slices(y)
            self.dataset = tf.data.Dataset.zip((dx, dy))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(x)

        self.shuffled_batched_dataset = self.dataset.shuffle(buffersize).batch(batchsize)

    def get_shuffled_batched_dataset(self):
        return self.shuffled_batched_dataset

    def update_shuffled_batched_dataset(self):
        self.shuffled_batched_dataset = self.dataset.shuffle(self.buffersize).batch(self.batchsize)
        return self.shuffled_batched_dataset

    def get_n_batch_from_shuffled_batched_dataset(self, n):
        it = iter(self.shuffled_batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x

# Basic operation for neural networks: linear layers
class Linear(layers.Layer):
    def __init__(self, name, n_hidden=32, **kwargs):
        super(Linear, self).__init__(name=name, **kwargs)
        self.n_hidden = n_hidden


    def build(self, input_shape):
        # running a initialization (pass the data to a network) will get input_shape
        n_length = input_shape[-1]
        self.w = self.add_weight(name='w', shape=(n_length, self.n_hidden),
                                 initializer=nn_initializer(),
                                 dtype=tf.float32, trainable=True)
        self.b = self.add_weight(name='b', shape=(self.n_hidden,),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.float32, trainable=True)


    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Fully connected neural networks
class FCNN(tf.keras.Model):
    def __init__(self, name, n_out, depth, n_hidden, act='tanh', **kwargs):
        super(FCNN, self).__init__(name=name, **kwargs)
        self.n_out = n_out
        self.depth = depth
        self.n_hidden = n_hidden
        self.act = act
        self.hidden_layers = []
        for i in range(depth):
            self.hidden_layers.append(Linear(str(i), n_hidden))
        self.l_f = Linear('last',n_out)

    def call(self, inputs):
        x = inputs
        for i in range(self.depth):
            if self.act == 'relu':
                x = tf.nn.relu(self.hidden_layers[i](x))
            elif self.act == 'tanh':
                x = tf.nn.tanh(self.hidden_layers[i](x))
            # softplus function: log(1 + exp(x)) can be viewed as soft relu but it is smooth 
            elif self.act == 'softplus':
                x = tf.nn.softplus(self.hidden_layers[i](x))
            elif self.act == 'sin':
                x = sin_act(self.hidden_layers[i](x))
        x = self.l_f(x)

        return x

def nn_initializer():
    return tf.keras.initializers.GlorotUniform(seed=8)

def sin_act(x):
    return tf.math.sin(x)

def gen_square_domain(ndim, n_train, bd=1.0, unitcube=False, hyperuniform=False):
    # subfunction: generate samples uniformly at random in a ball
    def gen_nd_ball(n_sample, n_dim):
        x_g = np.random.randn(n_sample, n_dim)
        u_number = np.random.rand(n_sample, 1)
        x_normalized = x_g / np.sqrt(np.sum(x_g**2, axis=1, keepdims=True))
        x_sample = (u_number**(1/n_dim) * x_normalized).astype(np.float32)
        return x_sample

    if not unitcube:
        # if hyperuniform, half data points drawn from a unit ball and half from uniform distribution
        if hyperuniform:
            n_corner = n_train//2
            n_circle = n_train - n_corner
            # generate samples uniformly at random from a unit ball
            x_circle = gen_nd_ball(n_circle, ndim)
            # most of these samples are lies in the corner of a hypercube
            x_corner = np.random.uniform(-bd, bd, [n_corner, ndim]).astype(np.float32)
            x = np.concatenate((x_circle, x_corner), axis=0)
        else:
            x = np.random.uniform(-bd, bd, [n_train, ndim]).astype(np.float32)
    else:
        x = np.random.uniform(0, 1, [n_train, ndim]).astype(np.float32)
    return x

def gen_nd_cube_boundary(ndim, n_train, unitcube=False):
    if not unitcube:
        x = np.random.randn(n_train, ndim).astype(np.float32)
        x = (x.T / np.max(np.abs(x), axis=1)).T
    # [0,1]^d unit cube
    else:
        x = np.random.randn(n_train, ndim).astype(np.float32)
        x = (x.T / np.max(np.abs(x), axis=1)).T
        x = 0.5*x + 0.5
    return x

def gen_train_data(n_dim, n_sample, probsetup):
    if probsetup == 6:
        x = gen_square_domain(n_dim, n_sample)
        x_boundary = gen_nd_cube_boundary(n_dim, n_sample)
    else:
        raise ValueError('probsetup is not valid')

    return x, x_boundary

# exponential minus square norm function
def diffusion_exp(x):
    x_sum_square = np.sum(x**2, axis=1, keepdims=True)
    ux = np.exp(-10.0*x_sum_square)
    return ux

def diffusion_exp_boundary(x_boundary):
    x_sum_square = tf.reduce_sum(tf.math.square(x_boundary), axis=1, keepdims=True)
    u_boundary = tf.math.exp(-10.0*x_sum_square)
    return u_boundary

def boundary_loss_exp(model, x_boundary):
    fx_boundary = model(x_boundary)
    u_boundary = diffusion_exp_boundary(x_boundary)
    residual_boundary = tf.math.square(fx_boundary-u_boundary)
    return residual_boundary

def a_diff_v1(x):
    a = tf.ones([x.shape[0], 1], dtype=tf.float32)
    return a 

def compute_grads(f_out, x_inputs):
    grads = tf.gradients(f_out, [x_inputs])[0]
    return grads

def compute_div(f_out, x_inputs):
    div_qx = tf.stack([tf.gradients(tmp, [x_inputs])[0][:, idx] for idx, tmp in enumerate(tf.unstack(f_out, axis=1))], axis=1)
    div_qx = tf.reduce_sum(div_qx, axis=1, keepdims=True)
    return div_qx

def f_source_exp(x):
    n_dim = x.shape[-1]
    x_sum_square = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
    f_source = tf.math.exp(-10.0*x_sum_square) * (20.0*n_dim - 400.0*x_sum_square)
    return f_source

def residual_exp(model, x):
    fx = model(x)
    # constant coefficient function
    a_coeff = a_diff_v1(x)
    grads = compute_grads(fx, x)
    qx = a_coeff * grads
    # compute divergence 
    div_qx = compute_div(qx, x)

    f_source = f_source_exp(x)
    residual = tf.math.square(-div_qx - f_source)

    return residual

def load_valid_data(n_dim, probsetup):
    if probsetup == 6:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_exp_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = diffusion_exp(sample_valid)
    else:
        raise ValueError('probsetp is not valid')

    return sample_valid, u_true

class DAS():
    def __init__(self, args):
        self.args = args
        self._set()
        self.build_nn()
 

    def _set(self):
        args = self.args
        pde_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        flow_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        self.pde_optimizer = pde_optimizer
        self.flow_optimizer = flow_optimizer

        # this is for creating folder for save training and validation loss history, checkpoint and summary
        if os.path.exists(args.ckpts_dir):
            shutil.rmtree(args.ckpts_dir)
        os.mkdir(args.ckpts_dir)

        if os.path.exists(args.summary_dir):
            shutil.rmtree(args.summary_dir)
        os.mkdir(args.summary_dir)

        if os.path.exists(args.data_dir):
            shutil.rmtree(args.data_dir)
        os.mkdir(args.data_dir)

        self.pdeloss_vs_iter = []
        self.residualloss_vs_iter = []
        self.entropyloss_vs_iter = []
        self.approximate_error_vs_iter = []
        self.resvar_vs_iter = []

    def build_nn(self):
        args = self.args
        # create a neural network to approximate the solution of PDEs
        net_u = FCNN('FCNN', 1, args.netu_depth, args.n_hidden, args.activation)

        self.net_u = net_u 

    def get_pde_loss(self, x, x_boundary, stage_idx):
        args = self.args

        if args.probsetup == 6:
            residual = residual_exp(self.net_u, x)
            residual_boundary = boundary_loss_exp(self.net_u, x_boundary)
        else:
            raise ValueError('probsetup is not valid')

        pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)
        return pde_loss, residual

    @tf.function
    def train_pde(self, inputs, inputs_boundary, i, net_u_training_vars):
        # two neural networks: one for approximating PDE, and another for adaptive sampling
        with tf.GradientTape() as pde_tape:
            pde_loss, residual = self.get_pde_loss(inputs, inputs_boundary, i)

        grads_net_u = pde_tape.gradient(pde_loss, net_u_training_vars)
        self.pde_optimizer.apply_gradients(zip(grads_net_u, net_u_training_vars))
        return pde_loss, residual

    def solve_pde(self, train_dataset, stage_idx, sample_valid, u_true):
        """train a neural network to approximate the pde solution"""
        args = self.args
        n_epochs = args.n_epochs
        for k in tf.range(1, n_epochs+1):
            for step, train_batch in enumerate(train_dataset):
                batch_x = train_batch[:,:args.n_dim]
                batch_boundary = train_batch[:,args.n_dim:]
                pde_loss, residual = self.train_pde(batch_x, batch_boundary, stage_idx, self.net_u.trainable_weights)

                residual_loss = tf.reduce_mean(residual)
                variance_residual = tf.math.reduce_variance(residual)
                print('stage: %s, epoch: %s, iter: %s, residual_loss: %s, pde_loss: %s ' % 
                      (stage_idx, k.numpy(), step+1, residual_loss.numpy(), pde_loss.numpy()))

                self.pdeloss_vs_iter += [pde_loss.numpy()]
                self.residualloss_vs_iter += [residual_loss.numpy()]
                self.resvar_vs_iter += [variance_residual.numpy()]

                ####################################################
                # evalute model performance using load test data every iteration
                if args.probsetup == 0 or args.probsetup == 6:
                    ## Error on data points
                    u_pred = self.net_u(sample_valid)
                    approximate_error = tf.norm(u_true - u_pred, ord=2)/tf.norm(u_true, ord=2)

                else:
                    u_pred = self.net_u(sample_valid)
                    approximate_error = tf.reduce_mean(tf.math.square(u_true - u_pred))

                self.approximate_error_vs_iter += [approximate_error.numpy()]
                #####################################################

        # record the final five steps for the stopping criterion
        tol_pde = np.mean(np.array(self.pdeloss_vs_iter[-5:]))
        res_var = np.mean(np.array(self.resvar_vs_iter[-5:]))

        return u_pred, tol_pde, res_var

    def train(self):
        """training procedure"""
        args = self.args
        max_stage = args.max_stage

        sample_valid, u_true = load_valid_data(args.n_dim, args.probsetup)

        summary_writer = tf.summary.create_file_writer(args.summary_dir)

        print(' Quantity type for adaptive procedure: %s' % (args.quantity_type))
        print('====== Training process starting... ======')

        with summary_writer.as_default():

            # set random seed
            np.random.seed(23)
            tf.random.set_seed(23)

            # starting from uniform distribution
            x_data, x_boundary = gen_train_data(args.n_dim, args.n_train, args.probsetup)
            x = np.concatenate((x_data, x_boundary), axis=1)

            data_flow_pde = dataflow(x, buffersize=args.n_train, batchsize=args.batch_size)
            train_dataset_pde = data_flow_pde.get_shuffled_batched_dataset()

            data_flow_kr = dataflow(x_data, buffersize=args.n_train, batchsize=args.flow_batch_size)

            m = 1
            x_init_kr = data_flow_kr.get_n_batch_from_shuffled_batched_dataset(m)
            # pass data to networks to complete building process
            self.net_u(x_init_kr)

            solve_pde_time = 0
            for i in range(1, max_stage+1):

                solve_pde_start = time.time()
                u_pred, tol_pde, res_var = self.solve_pde(train_dataset_pde, i, sample_valid, u_true)
                solve_pde_end = time.time()
                solve_pde_time += (solve_pde_end - solve_pde_start)/3600
                if tol_pde < args.tol and res_var < args.tol and i > 1:
                    print('===== stoppping criterion satisfies, finish training =====')
                    break
            print('solve_pde_time is {:.4} hours'.format(solve_pde_time))
