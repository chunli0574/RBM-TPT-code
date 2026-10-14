import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from tqdm import tqdm
import sys
import time


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
       
        delta_t = T /N  
        R = self.R
        U_boundary = self.U_boundary
        A_boundary = self.A_boundary
        C_boundary = self.C_boundary
        
        sqrt_delta_t = np.sqrt(delta_t)
        dw_sample = np.random.normal(size=[num_sample, self.dim, N * Total_iterations])# * sqrt_delta_t
        dw_sigma_t = np.zeros(dw_sample.shape)
        x0 = np.ones(self.dim) * self.x0;
        # x0 = np.array(self.x0)
        
        y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])

        y_sample_array = []
        du_sample_array = []
        

        for i in range(N * Total_iterations):
            if i % 1000 == 0:
                
                print("Generate Samples:",i)

            
            if i % N == 0:

                if simulation_method == "fixed":
                    y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])
                elif simulation_method == "uniform":
                    y_i = np.random.uniform (size = (num_sample, len(x0))) * 2
                y_sample_array.append(y_i)

            dw_sigma_t[:,:,i] = self.diffusion(dw_sample[:, :, i]) * sqrt_delta_t
           

            delta_x = self.drift() * delta_t + dw_sigma_t[:,:,i]

            x_i = y_i + delta_x
            y_i, du_i = self.Skorokod(x_i, U_boundary, A_boundary, C_boundary) 
            

            y_sample_array.append(y_i)
            du_sample_array.append(du_i)
           

        return np.stack(y_sample_array, axis = 2), dw_sigma_t, np.stack(du_sample_array, axis = 2)
    

        
    def Skorokod(self, X_full, U, A, C):
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

    
    
class extendedCrissCross(Equation): 
    def __init__(self, eqn_config):

        super(extendedCrissCross, self).__init__(eqn_config)
        self.name = 'extendedCrissCross'
        self.R = np.identity(self.dim)
        
        self.Delta1 = eqn_config.Delta1
        self.Delta2 = eqn_config.Delta2

        self.v1 =  eqn_config.v1
        self.v2 =  eqn_config.v2
        self.v3 =  eqn_config.v3
        self.var_multiple = eqn_config.var_multiple

        self.h1 = tf.constant([self.Delta2,self.Delta1], dtype = tf.float64) 
        self.h2 = tf.constant([self.Delta1,self.Delta2], dtype = tf.float64) 
        self.mu = np.repeat(eqn_config.mu,self.dim) 
        self.mu = self.mu;
        
        self.a = self.a  
        self.pricing_flag = eqn_config.pricing_flag
        self.simu_u = int(eqn_config.simu_u)

        # b_h: upper bound for horizontal (scaled) queue length
        # b_v: upper bound for vertical (scaled) queue length
        self.b_h = eqn_config.delay_bound[0]*eqn_config.eta_upper
        self.b_v = eqn_config.delay_bound[1]*eqn_config.eta_upper

        # ######################################################
        # The following parameters depend on model primitives
        # ######################################################
        
        # drift and variance of BM in workload problem
        self.realmu = np.repeat(-20.0/19, self.dim)  
        self.realvariance = 20.0/19*(np.ones([self.dim,self.dim])*1.0/2 + np.diag(np.ones(self.dim))*3.0/2)

        # trainingMatrix
        H = np.diag([10.0/19] * (self.dim + 1)) * 8
        M_tilde = 20.0/19 * np.hstack((np.eye(self.dim, dtype=np.float64), np.ones((self.dim, 1), dtype=np.float64)))
        self.pricing_matrix = M_tilde @ np.linalg.inv(H) @ M_tilde.T
        # self.pricing_matrix = tf.ones((self.dim, self.dim), dtype=tf.float64) + tf.eye(self.dim, dtype=tf.float64)

        # pushing direction of outsourcing controls
        self.m_v = 20.0/19
        self.m_h = 20.0/19

    def gen_samples(self, num_sample, T, N, Total_iterations, simulation_method):
       
        delta_t = T / N  
        R = self.R
        
        sqrt_delta_t = np.sqrt(delta_t)
        dw_sample = np.random.normal(size=[num_sample, self.dim, N * Total_iterations])# * sqrt_delta_t
        dw_sigma_t = np.zeros(dw_sample.shape)
        x0 = np.ones(self.dim) * self.x0;
        
        y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])

        y_sample_array = []
        du_sample_array = []
        

        for i in range(N * Total_iterations):
            if i % 1000 == 0:
                
                print("Generate Samples:",i)

            
            if i % N == 0:

                if (simulation_method == "fixed") and (i % (self.simu_u*N) == 0):
                    y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])
                elif simulation_method == "uniform":
                    y_i = np.random.uniform (size = (num_sample, len(x0))) * 2
                elif (simulation_method == "binomial2") and (i % (self.simu_u*N) == 0):
                    y_i_index = np.random.randint(low = 0, high = len(x0) + 1, size=(num_sample))
                    upper_triangular_matrix = np.triu(np.ones((len(x0) + 1, len(x0)), dtype = int))
                    y_i = np.random.binomial(n=1, p=0.5, size=(num_sample, len(x0))) * self.b_v + upper_triangular_matrix[y_i_index,:] * self.b_h

                y_sample_array.append(y_i)

            dw_sigma_t[:,:,i] = self.diffusion(dw_sample[:, :, i]) * sqrt_delta_t
           

            delta_x = self.drift() * delta_t + dw_sigma_t[:,:,i]

            x_i = y_i + delta_x
            y_i, du_i = self.Skorokod(x_i) 
            

            y_sample_array.append(y_i)
            du_sample_array.append(du_i)
           

        return np.stack(y_sample_array, axis = 2), dw_sigma_t, np.stack(du_sample_array, axis = 2)
    

        
    def Skorokod(self, X_full): 
        
        eps = 1e-8
        num_samples = X_full.shape[0]
        d = self.dim
        b_h = self.b_h
        b_v = self.b_v
        
        Y_ret = np.zeros(X_full.shape)
        dU_ret = np.zeros([num_samples,(2*d+1)])
        
        for i in range(num_samples):
            X = X_full[i,:]
            Y = np.zeros(X.shape)
            dU_tmp = np.zeros(d*2+1)
            Y_max = -1
                    
            for j in range(d):
                if j == 0:
                    Y[j] = min(max(X[j],0), b_h + b_v)
                else:
                    Y[j] = min(max(X[j],max(0, Y_max - b_v)), b_h + b_v)
                if Y[j] > Y_max:
                    Y_max = Y[j]
                    
            dU_tmp[0:d] = np.maximum(0, Y - X)
            dU_tmp[d:2*d] = np.maximum(0, X - Y)
    
            Y_ret[i,:] = Y
            dU_ret[i,:] = dU_tmp

        return Y_ret, dU_ret
        
    
    def w_tf(self, x, grad_t, a_lowbound = 0): 

        w = - tf.linalg.matvec(grad_t, self.mu - self.realmu)
        w = tf.reshape(w, [-1,1])

        # control (dynamic pricing)
        if (self.pricing_flag == "Y"):
            w = w - tf.reduce_sum(grad_t * tf.matmul(grad_t, self.pricing_matrix), axis=1, keepdims=True) / 4
        
        # control (admission control)
        
        max_zero_grad1 = tf.math.maximum(tf.cast(0., tf.float64), (-1)*grad_t - self.v1)
        max_zero_grad2 = tf.math.maximum(tf.cast(0., tf.float64), (self.m_v)*grad_t - self.v2)
        max_zero_grad3 = tf.math.maximum(tf.cast(0., tf.float64), tf.linalg.matvec(grad_t,tf.cast((self.m_h)*np.ones(self.dim),tf.float64)) - self.v3)
        
        w = w - tf.reduce_sum(max_zero_grad1 + max_zero_grad2, 1, keepdims=True) * (self.a - a_lowbound) 
        w = w - tf.reshape(max_zero_grad3, [-1,1]) * (self.a - a_lowbound) 
        
        return w, 0


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
