import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from tqdm import tqdm


from scipy.stats import multivariate_normal as normal
class ControlSolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.LAMBDA = 1
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.train_config = config.train_config
        self.bsde = bsde
        self.comments =  config.eqn_config._comment

        self.loss_rec = []
        self.val_rec = []
        self.train_loss_rec = []

        if "alow" not in self.eqn_config:
            self.true_ld = 0.0
        else:
            self.true_ld = self.eqn_config.alow

       
        if "simulation" not in self.eqn_config:
            self.eqn_config.simulation = ""
        
        if "queue_example" not in self.eqn_config:
            self.queue_example = ""
        else:
            self.queue_example = self.eqn_config.queue_example
       
        if "control" in self.train_config:
            self.control = True

        self.model_critic = CriticModel(config, bsde)
        

        lr_schedule_critic = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries_critic, self.net_config.lr_values_critic)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_schedule_critic, epsilon=1e-8)
      

        self.gamma = self.eqn_config.discount
        if self.gamma < 1e-9:
            self.steady_flag = True  
        else:
            self.steady_flag = False
       
    def gen_samples(self, dump, load):
        start_time = time.time()


        num_sample = self.net_config.batch_size
        T = self.eqn_config.total_time_critic
        N = self.eqn_config.num_time_interval_critic
        Total_iterations = self.net_config.num_iterations
        simulation_method = self.eqn_config.simulation


        logging.info('mu: %.2f, R: %.2f, rho: %.2f, dim: %d, gamma: %.2f ' %(self.bsde.mu[0], self.eqn_config.R, self.eqn_config.rho, self.bsde.dim, self.gamma))
        logging.info('N: %d , T: %.2f, num_sample: %d, batch_size: %d' %(N,T,Total_iterations, self.net_config.batch_size) )
        print('pricing_flag:', self.eqn_config.pricing_flag)

        if not os.path.exists("data"):
            os.makedirs("data")
        
        filename_true = "data/"+self.bsde.name + "_" + str(self.bsde.mu[0])  + "_" + str(self.eqn_config.R) + "_" + str(self.eqn_config.rho) + "_" + str(num_sample) + "_" + str(T) + "_" + str(N) + "_" + str(Total_iterations) + "_dim="+str(self.bsde.dim) + "_" + str(self.train_config.epoch) + "pace=" + str(self.train_config.pace) + "_addcost=" + str(self.eqn_config.addcost[0]) + "_" + str(self.eqn_config.addcost[1]) + "_" + str(self.eqn_config.addcost[2])  

        if self.eqn_config.eqn_name == "extendedCrissCross":
            filename_true = filename_true + "_lr" +  str(self.net_config.lr_boundaries_critic[-1])     
            filename_true = filename_true + "_eta=" + str(self.eqn_config.eta_upper)
            filename_true = filename_true + "_pricing=" + self.eqn_config.pricing_flag  
            filename_true = filename_true + "_delay=" + str(self.eqn_config.delay_bound[0]) + "_" + str(self.eqn_config.delay_bound[1])


        filename = filename_true + simulation_method
        
        

        if load:
            self.dw_sample = np.load(filename+'_w.npy', allow_pickle = True)
            self.y_sample = np.load(filename+'_y.npy', allow_pickle = True)
            self.du_sample = np.load(filename+'_u.npy', allow_pickle = True)
            dw_shape = self.dw_sample.shape
            
            
            assert dw_shape[0] == num_sample and dw_shape[1] == self.bsde.dim and dw_shape[2] == N * (Total_iterations + 20)
        else:
            self.y_sample, self.dw_sample, self.du_sample = self.bsde.gen_samples(num_sample,T,N, Total_iterations + 20, simulation_method) 
        
        if dump:
            # np.save(filename+'_w', self.dw_sample)
            # np.save(filename+'_y', self.y_sample)
            # np.save(filename+'_u', self.du_sample)
            logging.info('Have dumped data. Time: %3u' % (time.time() - start_time))
        logging.info(self.y_sample.shape)


    def gen_filename(self):
        simulation_method = self.eqn_config.simulation
        T = self.eqn_config.total_time_critic
        N = self.eqn_config.num_time_interval_critic
        Total_iterations = self.net_config.num_iterations
        suffix = self.comments + "_" + self.eqn_config.eqn_name + self.queue_example + "_" + self.net_config.transformation + "_" + str(self.net_config.num_hiddens_critic[0]) +"_" + str(self.net_config.num_hiddens_critic[1]) + "_" + str(self.net_config.num_hiddens_critic[-1]) + "_" 
        suffix = suffix + "D" + str(len(self.net_config.num_hiddens_critic)) + "_"
        suffix = suffix + str (self.bsde.mu[0])+ "_" + str(self.eqn_config.R) + "_" + str (self.bsde.gamma) + "_" + str (self.bsde.a)+"_" + str(T) 
        suffix = suffix + "_" + str(N) + "_" + str(Total_iterations) +"_dim="+str(self.bsde.dim)  +  simulation_method + "_" + self.train_config.TD_type
        suffix = suffix + "_" + self.net_config.activation
        if self.eqn_config.eqn_name == "extendedCrissCross":
            suffix = suffix + "_" + str(self.train_config.epoch)  + "pace=" + str(self.train_config.pace)
            suffix = suffix + "_addcost=" + str(self.eqn_config.addcost[0]) + "_" + str(self.eqn_config.addcost[1]) + "_" + str(self.eqn_config.addcost[2])
            suffix = suffix  + "_lr" +  str(self.net_config.lr_boundaries_critic[-1]) 
            suffix = suffix + "_eta=" + str(self.eqn_config.eta_upper)
            suffix = suffix + "_pricing=" + self.eqn_config.pricing_flag
            suffix = suffix + "_delay=" + str(self.eqn_config.delay_bound[0]) + "_" + str(self.eqn_config.delay_bound[1])

        suffix = suffix.replace(" ","")
        suffix = suffix.replace(",","_")
        suffix = suffix.replace("[","")
        suffix = suffix.replace("]","")
        return suffix

    def train(self):
  
        
        start_time = time.time()

        # the pace of a lower bound converging to zero
        if "pace" in self.train_config:
            pace = self.train_config.pace
        
        else: pace = 0
        true_ld = self.true_ld;
        simulation_method = self.eqn_config.simulation
        T = self.eqn_config.total_time_critic
        N = self.eqn_config.num_time_interval_critic
        Total_iterations = self.net_config.num_iterations
        suffix = self.gen_filename()

        logging.info(suffix)
        logging.info("LAMBDA: %d" %(self.LAMBDA) )
        # validation data
        for step in range(self.net_config.num_iterations, self.net_config.num_iterations + 20 ):
            cur_start = step * N 
            cur_end = (step + 1) * N 
            cur_start_y = step * (N + 1) 
            cur_end_y = (step + 1) * (N + 1)
            if step == self.net_config.num_iterations:
                validation_w_sample = self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end]
                validation_y_sample = self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y]
                validation_u_sample = self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end]
            else:
                validation_w_sample = np.append(validation_w_sample, self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end], axis = 0)
                validation_y_sample = np.append(validation_y_sample, self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y], axis = 0)
                validation_u_sample = np.append(validation_u_sample, self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end], axis = 0)

        valid_data_critic = tf.constant(0.0, dtype=tf.float64), validation_w_sample, validation_y_sample, validation_u_sample

        #Total Epoch
        if "epoch" in self.train_config:
            epoch_total = self.train_config.epoch
        else:
            epoch_total= self.bsde.dim * 2 + self.bsde.a  + 20;
        
        logging.info('Epoch Total: %d' % (epoch_total))
        # begin sgd iteration
        for epoch in range(epoch_total):
            
            logging.info("Epoch: %d",epoch)
            if epoch > 0: 
                discard_num = 1000
            else: discard_num = 0 
            for step in range(self.net_config.num_iterations ):
                
                if epoch > 0 and step < discard_num:
                    continue
               
                cur_start = step * N 
                cur_end = (step + 1) * N 
                cur_start_y = step * (N + 1) 
                cur_end_y = (step + 1) * (N + 1)
                
                
                if pace == 0:
                    a_lowbound = max(true_ld,0.0)
                else:
                    a_lowbound = max(true_ld, (self.bsde.a * 0.8) - ( step + epoch * (self.net_config.num_iterations - discard_num)) / 40 / self.bsde.dim / pace);
                   
                a_lowbound = tf.cast(a_lowbound, dtype=tf.float64)
                
                # output training results
                if step % self.net_config.logging_frequency == 0 or  step == self.net_config.num_iterations  - 1:
                    
                    logging.info("lower_bound: %.3f",a_lowbound)
                    
                   

                    x0 = np.ones([self.net_config.batch_size, self.bsde.dim]) * 2
                    valid_loss_critic, err_value = self.loss_critic(valid_data_critic, training=False, use_NN=True)
                    valid_loss_critic = valid_loss_critic.numpy()
                    self.loss_rec.append(valid_loss_critic)
                    self.val_rec.append(err_value)
                    
                    cur_grad = self.compute_grad(x0).numpy()
                    cur_value = self.compute_value(x0).numpy()
         
                    elapsed_time = time.time() - start_time

     
                    train_data = a_lowbound, self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end], self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y], self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end]
                    train_loss_critic, train_err_value = self.loss_critic(train_data, training=False, use_NN=True)
            
                    self.train_loss_rec.append(train_loss_critic.numpy())
                
                    logging.info("validation loss: %.6f, training loss %.6f",   valid_loss_critic, train_loss_critic)
                    logging.info("step: %5u, Current Val: %.4f,  cur_V: %.4f,  cur_grad: %.4f, elapsed time: %3u" % (
                                step,  err_value,  cur_value, cur_grad, elapsed_time))
                   
                    fig_folder = "figs_" +  self.net_config.activation +"/"
                    log_folder = "logs_" +  self.net_config.activation + "/"
                    if not os.path.exists(fig_folder):
                        os.makedirs(fig_folder)
                    if not os.path.exists(log_folder):
                        os.makedirs(log_folder)
                    filename = "models/"+suffix

                    self.model_critic.NN_value.save(filename + ".keras")
                    self.model_critic.NN_value_grad.save(filename + "_grad.keras")

                
                # train
                train_data = a_lowbound, self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end], self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y], self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end]
                self.train_step_critic(train_data)
                

                if step == self.net_config.num_iterations  - 1:
                    filename = "models/"+suffix
                    self.model_critic.NN_value.save("models/" + suffix + ".keras")
                    np.savetxt(log_folder + suffix, np.stack([self.val_rec, self.loss_rec, self.train_loss_rec]))
              

    def loss_critic(self, inputs, training, use_NN):
        delta, cur_val, negative_loss = self.model_critic(inputs, training=training, use_NN=use_NN)
        
       
        loss = tf.math.reduce_variance(delta) / self.eqn_config.total_time_critic ** 2 
        return loss + self.LAMBDA * negative_loss, cur_val

    def grad_critic(self, inputs, training, use_NN):
        with tf.GradientTape(persistent=True) as tape:
            loss_critic, temp= self.loss_critic(inputs, training, use_NN)
            
        grad = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_critic(self, train_data):
        grad = self.grad_critic(train_data, training=False, use_NN = True)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        

    
    def compute_grad(self, x0):
        if self.bsde.name =="tandem":
            Z=self.model_critic.NN_value_grad(x0, training=False, need_grad=False)
            return  tf.reduce_mean(Z[:,0]-Z[:,1])
        return tf.reduce_mean(self.model_critic.NN_value_grad(x0, training=False, need_grad=False))

    
    def compute_value(self, x0):
        return tf.reduce_mean(self.model_critic.NN_value(x0, training=False, need_grad=False))
  
class CriticModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(CriticModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.train_config = config.train_config
        self.bsde = bsde
        self.NN_value = DeepNN(config, "critic")
        self.NN_value_grad = DeepNN(config, "critic_grad")
        self.gamma = config.eqn_config.discount

        if self.gamma < 1e-9:
            self.steady_flag = True
        else:
            self.steady_flag = False

        # Extract eqn_config fields as plain scalars to avoid Keras 3.x TrackedDict issue
        self.dim = config.eqn_config.dim
        self.total_time_critic = config.eqn_config.total_time_critic
        self.num_time_interval_critic = config.eqn_config.num_time_interval_critic
        self.eqn_name = config.eqn_config.eqn_name
        self.v1 = config.eqn_config.v1
        self.v2 = config.eqn_config.v2
        self.v3 = config.eqn_config.v3
        self.addcost = config.eqn_config.addcost

        if "control" in self.train_config:
            self.control = True
            
       
    def call(self, inputs, training, use_NN = True):
        negative_loss=0.0
        dim = self.dim
        a_lowbound, dw, y_sample, du = inputs
        flag_double_parametrization = True
       
        num_sample = np.shape(dw)[0]
       
        y = 0
        discount = 1 #broadcast to num_sample x 1
        temp_neg_loss2 = 0
        
        
        dt = self.total_time_critic / self.num_time_interval_critic

        for t in range(self.num_time_interval_critic):
            
            if flag_double_parametrization:
                #for parametrize G

                if self.eqn_name == "tandem":
                    i = 0
                    with tf.GradientTape() as g:
                        temp_x = tf.convert_to_tensor(y_sample[:,:,t])
                        g.watch(temp_x)
                        grad_t = self.NN_value_grad(temp_x, training=training, need_grad=False)
                        
                        hes = g.gradient(grad_t[:,i] - grad_t[:,i+1], temp_x)
                       
                        zero_grad0 = tf.math.minimum(tf.cast(0., tf.float64),hes[:,i])
                        
                        temp_neg_loss2 = temp_neg_loss2 + tf.reduce_sum(tf.square(zero_grad0)) 
                else:
                    
                    grad_t = self.NN_value_grad(y_sample[:,:,t], training=training, need_grad=False)

            else:
            # for only V

                with tf.GradientTape() as g:
                    temp_x = tf.convert_to_tensor(y_sample[:,:,t])
                    g.watch(temp_x)
                    grad_t = g.gradient(self.NN_value(temp_x, training=training, need_grad=False), temp_x)
           

            # ##############################
            # for F function
            # ##############################
            if self.control:
                if use_NN:
                    
                    w, temp_neg_loss = self.bsde.w_tf(y_sample[:,:,t], grad_t,a_lowbound) # running cost
                else:
                    w, temp_neg_loss = self.bsde.w_tf(y_sample[:,:,t], self.bsde.V_grad_true(y_sample[:,:,t])) # running cost
                negative_loss = negative_loss +  temp_neg_loss + temp_neg_loss2
            
            else:
                w, temp_neg_loss = self.bsde.w_tf(y_sample[:,:,t])
            
            # ##############################
            # integrand for drift
            # ##############################
            delta_y_drift = w * discount * dt
            y += delta_y_drift 
            
            # ##############################
            # integrand for diffusion 
            # ##############################
            # integrand for diffusion, sigma is num_sample x dim x dim_w, NN_value_grad is num_sample x dim, dw[:,:,t] is num_sample x dim_w
            delta_y_diffusion =  dw[:,:,t]
            if use_NN:
                delta_y_diffusion = tf.reduce_sum(delta_y_diffusion * grad_t, axis=1, keepdims=True)
            else:
                delta_y_diffusion = tf.reduce_sum(delta_y_diffusion * self.bsde.V_grad_true(y_sample[:,:,t]), axis=1, keepdims=True)
            delta_y_diffusion *= discount 
            #coef for diffusion
            y -= delta_y_diffusion 
            
            # ##############################
            # integrand for pushing control
            # ##############################
            delta_y_pushing = du[:,:,t]

            if self.eqn_name == "extendedCrissCross":

                boundary_cost_v = tf.concat([tf.constant(self.v1 + self.addcost[0], shape=[self.dim], dtype=tf.float64), tf.constant(self.v2 + self.addcost[1], shape=[self.dim], dtype=tf.float64), tf.constant(self.v3 + self.addcost[2], shape=[1], dtype=tf.float64)], 0)
                delta_y_pushing = tf.reduce_sum(delta_y_pushing * boundary_cost_v, axis=1, keepdims=True)

            delta_y_pushing *= discount 
            y += delta_y_pushing
            
            # we need to update the discount
            discount *= np.exp(-self.gamma * dt )

        y_final = y_sample[:,:,-1]

        if use_NN:    
            delta = self.NN_value(y_sample[:,:,0], training=training, need_grad=False) - y - self.NN_value(y_final, training=training, need_grad=False) * discount
        else:
            delta = self.bsde.V_true(y_sample[:,:,0]) - y - self.bsde.V_true(y_final) * discount
    
        if self.steady_flag:
            rtn_val = -tf.reduce_mean(delta) / self.total_time_critic
        else:
            rtn_val = -tf.reduce_mean(delta) / (1 - np.exp(-self.gamma * self.total_time_critic))


        return delta, rtn_val, negative_loss

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class DeepNN(tf.keras.Model):
    def __init__(self, config, AC, **kwargs):
        super(DeepNN, self).__init__(**kwargs)
        self.AC = AC
        self.eqn_config = config.eqn_config

        self.eqn = config.eqn_config.eqn_name
        
        dim = config.eqn_config.dim
       
        num_hiddens = config.net_config.num_hiddens_critic
       
        self.activation = config.net_config.activation

     

        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],use_bias=False,activation=None)for i in range(len(num_hiddens))]
       
        if AC == "critic":
            self.dense_layers.append(tf.keras.layers.Dense(1, activation=None, use_bias=True))

        elif AC == "critic_grad":
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None, use_bias=False))
   

    def call(self, x, training=False, need_grad=False):
        
        eps = 1e-4

        with tf.GradientTape() as g:
            if self.AC == "critic" and need_grad:
                g.watch(x)
           
            y = x;
            dim = x.shape[1];

            for i in range(len(self.dense_layers) - 1):
                y = self.dense_layers[i](y)
                
               
                if self.activation == "relu":
                    y =  tf.nn.relu(y)
                elif self.activation =="leaky":
                    y =  tf.nn.leaky_relu(y)
                elif self.activation =="elu":
                    y = tf.nn.elu(y)
                elif self.activation =="sigmoid":
                    y = y + tf.math.sigmoid(y) - 1/2
                    
            y = self.dense_layers[-1](y)
            
        if self.AC == "critic" and need_grad:
            return y, g.gradient(y, x)
        else:
            return y
