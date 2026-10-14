import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from tqdm import tqdm
import sys
import time
import logging

from rs_reflection import reflect_rs_closed_form, validate_rs_backup_configuration


class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
                
        self.dim = eqn_config.dim
        self.gamma = eqn_config.discount
        self.x0 = eqn_config.x0
        self.var = 1
        if "var" in eqn_config:
            self.var = eqn_config.var
        
        self.a = eqn_config.a
        

    def gen_samples(self, num_sample, T, N, Total_iterations, simulation_method):
       
        delta_t = T / N  
        
        sqrt_delta_t = np.sqrt(delta_t)
        dw_sample = np.random.normal(size=[num_sample, self.dim, N * Total_iterations])# * sqrt_delta_t
        dw_sigma_t = np.zeros(dw_sample.shape)
        x0 = np.ones(self.dim) * self.x0;
        
        y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])

        y_sample_array = []
        du_sample_array = []
        

        for i in range(N * Total_iterations):
            if i % 1000 == 0:
                
                logging.info(f"Generate Samples: {i}")
                # print("Generate Samples:",i)

            
            if i % N == 0:

                if simulation_method == "fixed":
                    y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])
                elif simulation_method == "uniform":
                    y_i = np.random.uniform (size = (num_sample, len(x0))) * 2
                y_sample_array.append(y_i)

            dw_sigma_t[:,:,i] = self.diffusion(dw_sample[:, :, i]) * sqrt_delta_t
           

            delta_x = self.drift() * delta_t + dw_sigma_t[:,:,i]

            x_i = y_i + delta_x
            y_i, du_i = self.Skorokod(x_i) 
            

            y_sample_array.append(y_i)
            du_sample_array.append(du_i)
           

        return np.stack(y_sample_array, axis = 2), dw_sigma_t, np.stack(du_sample_array, axis = 2)
    

    def Skorokod(self, X_full):
        """Skorokhod mapping in the SDE"""
        raise NotImplementedError


    def w_tf(self, x):
        """Running cost in control problems."""
        raise NotImplementedError

        
    def V_true(self, x):
        """True value function"""
        raise NotImplementedError

    def V_grad_true(self, x):
        raise NotImplementedError

    def c_true(self):
        """True cost"""
        raise NotImplementedError
    
        
    def sigma(self): #num_sample x dim x dim_w
        """diffusion coefficient"""
        raise NotImplementedError
        
    def drift(self):
        """drift in the SDE"""
        raise NotImplementedError
    
    def diffusion(self,  dw):
        """diffusion in the SDE"""
        raise NotImplementedError


    
class RybkoStolyar(Equation): 
    def __init__(self, eqn_config):

        super(RybkoStolyar, self).__init__(eqn_config)
        self.name = 'RybkoStolyar'
        self.R = np.identity(self.dim)
        
        self.mu = np.repeat(eqn_config.mu, self.dim)         
        self.a = self.a  
        self.pricing_multiply = eqn_config.pricing_multiply

        # ######################################################
        # The following parameters depend on model primitives
        # ######################################################

        # drift and variance of BM in workload problem
        self.realmu = np.repeat(-20.0/19, self.dim)
        self.realvariance = 20.0/19*np.array([[10.0/3, 4.0/3],[4.0/3, 10.0/3]])

        # drift control: trainingMatrix
        self.pricing_matrix = tf.constant([[50.0/57, 40.0/57],
                                            [40.0/57, 50.0/57]], dtype = tf.float64) * self.pricing_multiply
        self.pricing_flag = eqn_config.pricing_flag

        # singular controls: pushing direction
        self.m1 = 20.0/19
        self.m2 = 40.0/19
        self.m3 = 20.0/19
        self.m4 = 40.0/19
        self.eta_upper = validate_rs_backup_configuration(eqn_config.eta_upper)
        # singular controls: cost
        self.v1 = eqn_config.v1
        self.v2 = eqn_config.v2
        
        

    def Skorokod(self, X_full):
        """Vectorized three-step RS mapping with four control increments."""
        return reflect_rs_closed_form(X_full, self.eta_upper)

    def w_tf(self, x, grad_t, a_lowbound = 0): 
        
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        m4 = self.m4
        
        w = - tf.linalg.matvec(grad_t, self.mu - self.realmu)
        w = tf.reshape(w, [-1,1])
        
        # control (dynamic pricing)
        
        if (self.pricing_flag == "Y"):
            w = w - tf.reduce_sum(grad_t * tf.matmul(grad_t, self.pricing_matrix), axis=1, keepdims=True) / 4
        
        # control (admission control)
        
        max_zero_grad1 = tf.math.maximum(tf.cast(0., tf.float64), tf.linalg.matvec(grad_t,tf.cast([m1,m2],tf.float64)) - self.v1)
        max_zero_grad2 = tf.math.maximum(tf.cast(0., tf.float64), tf.linalg.matvec(grad_t,tf.cast([m4,m3],tf.float64)) - self.v2)

        max_zero_grad = max_zero_grad1 + max_zero_grad2
        
        w = w - tf.reshape(max_zero_grad, [-1,1]) * (self.a - a_lowbound) 
        
        min_zero_gradG12 = tf.math.minimum(tf.cast(0., tf.float64) , grad_t)
        w = w + tf.reduce_sum(min_zero_gradG12, 1, keepdims=True) * (self.a - a_lowbound) 


        zero_grad0 = tf.math.minimum(tf.cast(0., tf.float64), grad_t[:,0])
        zero_grad1 = tf.math.minimum(tf.cast(0., tf.float64), grad_t[:,1])
        return w, tf.reduce_sum(tf.square(zero_grad0)) + tf.reduce_sum(tf.square(zero_grad1))

    

    def sigma(self): # x is num_sample x dim, u is num_sample x dim_u, sigma is num_sample x dim x dim_w
        return np.linalg.cholesky(self.realvariance)


    def drift(self):
        return self.mu
    
    def diffusion(self,  dw): #sigma num_sample x dim x dim_w, dw is num_sample x dim_w
        return np.dot(self.sigma(), dw.transpose()).transpose() # num_sample x dim
    
    def const_control_optimal(self):
        pass
    
    def linear_control_optimal(self):
        pass
