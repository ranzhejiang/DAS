from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

import BR_lib.BR_data as BR_data
import pde_model

import os
import shutil
import time

from tensorflow.keras import layers

def nn_initializer():
    return tf.keras.initializers.GlorotUniform(seed=8)

def sin_act(x):
    """
    Sine activation function
    """
    return tf.math.sin(x)



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

def gen_train_data(n_dim, n_sample, probsetup):
    """
    generate training data for the first stage
    Args:
    -----
        n_dim: dimension
        n_sample: number of samples
        probsetup: type of problems, see das_train.py file 

    Returns:
    --------
        x, x_boundary
        data points for training, including interior data points and boundary data points
    """
    if probsetup == 3:
        x = BR_data.gen_square_domain(n_dim, n_sample)
        x_boundary = BR_data.gen_square_domain_boundary(n_dim, n_sample)

    elif probsetup == 6:
        x = BR_data.gen_square_domain(n_dim, n_sample)
        x_boundary = BR_data.gen_nd_cube_boundary(n_dim, n_sample)

    elif probsetup == 7:
        x = BR_data.gen_square_domain(n_dim, n_sample)
        x_boundary = BR_data.gen_square_domain_boundary(n_dim, n_sample)

    else:
        raise ValueError('probsetup is not valid')

    return x, x_boundary


def load_valid_data(n_dim, probsetup):
    """
    load validation data for performance evaluation
    Args:
    -----
        n_dim: data dimension
        probsetup: type of problems

    Returns:
    --------
        true function values at the validation set, numpy format
    """
    if probsetup == 3:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_square_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = pde_model.diffusion_peak(sample_valid)

    elif probsetup == 6:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_exp_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = pde_model.diffusion_exp(sample_valid)

    elif probsetup == 7:
        valid_dir = os.path.join('./dataset_for_validation', '{}d_square_problem.dat'.format(n_dim))
        sample_valid = np.loadtxt(valid_dir).astype(np.float32)
        u_true = pde_model.bimodal_exact(sample_valid)

    else:
        raise ValueError('probsetp is not valid')

    return sample_valid, u_true



class DAS():
    """
    Deep adaptive sampling (DAS) for partial differential  equations
    ------------------------------------------------------------------------------------
    Here is the deep adaptive sampling method to solve partial differential equations. 
    Solving PDEs using deep nueral networks needs to compute a loss function with sample generation. 
    In general, uniform samples are generated, but it is not an optimal choice to efficiently train models. 
    However, flow-based generative models provide an opportunity for efficient sampling, and this is exactly what DAS 
    does. 
    
    Args:
    -----
        args: input parameters
    """
    def __init__(self, args):
        self.args = args
        self._set()
        self.build_nn()
        # self.build_flow()
        self._restore()
 

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


    def _restore(self):
        args = self.args
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.pde_optimizer, net=self.net_u)
        self.manager = tf.train.CheckpointManager(self.ckpt, args.ckpts_dir, max_to_keep=5)

    def build_nn(self):
        args = self.args
        # create a neural network to approximate the solution of PDEs
        net_u = FCNN('FCNN', 1, args.netu_depth, args.n_hidden, args.activation)

        self.net_u = net_u 

    def get_pde_loss(self, x, x_boundary, stage_idx):
        args = self.args

        if args.probsetup == 3:
            residual = pde_model.residual_peak(self.net_u, x)
            residual_boundary = pde_model.boundary_loss_peak(self.net_u, x_boundary)

        elif args.probsetup == 6:
            residual = pde_model.residual_exp(self.net_u, x)
            residual_boundary = pde_model.boundary_loss_exp(self.net_u, x_boundary)

        elif args.probsetup == 7:
            residual = pde_model.residual_bimodal(self.net_u, x)
            residual_boundary = pde_model.boundary_loss_bimodal(self.net_u, x_boundary)

        else:
            raise ValueError('probsetup is not valid')

        # When replace_all = 0, DAS-G; DAS-R, else
        # importance sampling may be used if replace all samples
        if stage_idx == 1:
            pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

        else:

            if args.replace_all == 1:
                # importance sampling for computing residual
                # scaling to avoid numerical underflow issues
                if args.if_IS_residual == 0:
                    pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)
                else:
                    scaling = 1000.0
                    log_pdf = tf.clip_by_value(self.pdf_model(x), -23.02585, 5.0)
                    pdfx = tf.math.exp(log_pdf)
                    weight_residual = tf.math.divide(scaling*residual, scaling*pdfx)
                    pde_loss = tf.reduce_mean(weight_residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

            else:
                if args.if_IS_residual == 0:
                    pde_loss = tf.reduce_mean(residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

                else:
                    # importance sampling for computing residual
                    # scaling to avoid numerical underflow issues
                    scaling = 1000.0
                    log_pdf = tf.clip_by_value(self.pdf_model(x), -23.02585, 5.0)
                    pdfx = tf.math.exp(log_pdf)
                    weight_residual = tf.math.divide(scaling*residual, scaling*pdfx)
                    pde_loss = tf.reduce_mean(weight_residual) + args.lambda_bd*tf.reduce_mean(residual_boundary)

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
            # Save model
            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % args.ckpt_step == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

        # record the final five steps for the stopping criterion
        tol_pde = np.mean(np.array(self.pdeloss_vs_iter[-5:]))
        res_var = np.mean(np.array(self.resvar_vs_iter[-5:]))

        return u_pred, tol_pde, res_var

    def train(self):
        """training procedure"""
        args = self.args
        max_stage = args.max_stage

        #################################################
        #load test data for evaluating model
        sample_valid, u_true = load_valid_data(args.n_dim, args.probsetup)

        # summary
        summary_writer = tf.summary.create_file_writer(args.summary_dir)

        print(' Quantity type for adaptive procedure: %s' % (args.quantity_type))
        print('====== Training process starting... ======')

        with summary_writer.as_default():

            # set random seed
            np.random.seed(23)
            tf.random.set_seed(23)
            # In the first step, data points are generated uniformly since there is no prior information 

            # starting from uniform distribution
            x_data, x_boundary = gen_train_data(args.n_dim, args.n_train, args.probsetup)
            x = np.concatenate((x_data, x_boundary), axis=1)

            data_flow_pde = BR_data.dataflow(x, buffersize=args.n_train, batchsize=args.batch_size)
            train_dataset_pde = data_flow_pde.get_shuffled_batched_dataset()

            data_flow_kr = BR_data.dataflow(x_data, buffersize=args.n_train, batchsize=args.flow_batch_size)

            m = 1
            x_init_kr = data_flow_kr.get_n_batch_from_shuffled_batched_dataset(m)
            # pass data to networks to complete building process
            self.net_u(x_init_kr)

            solve_pde_time = 0
            solve_flow_time = 0
            for i in range(1, max_stage+1):

                solve_pde_start = time.time()
                u_pred, tol_pde, res_var = self.solve_pde(train_dataset_pde, i, sample_valid, u_true)
                solve_pde_end = time.time()
                solve_pde_time += (solve_pde_end - solve_pde_start)/3600
                if tol_pde < args.tol and res_var < args.tol and i > 1:
                    print('===== stoppping criterion satisfies, finish training =====')
                    break
            print('solve_pde_time is {:.4} hours'.format(solve_pde_time))
            print('solve_flow_time time is {:.8} hours'.format(solve_flow_time))

