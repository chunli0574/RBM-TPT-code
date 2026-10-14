import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import ops

from tqdm import tqdm  # Progress bar for iterations
from munch import Munch  # Dictionary with dot-access
from scipy.stats import multivariate_normal as normal

from rs_reflection import RS_CONTROL_ORDER

# ============================================================
# ControlSolver: main class for training the control / BSDE solver
# ============================================================
class ControlSolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        # ----------------------------------------------------
        # Regularization coefficient for loss function
        # ----------------------------------------------------
        self.LAMBDA = 1  # Ques

        # ----------------------------------------------------
        # Store configuration components
        # ----------------------------------------------------
        self.eqn_config = config.eqn_config     # equation-related parameters
        self.net_config = config.net_config     # neural network parameters
        self.train_config = config.train_config # training parameters
        self.raw_boundary_pushing_cost = tuple(self.eqn_config.boundary_pushing_cost_raw)

        # ----------------------------------------------------
        # BSDE / control problem instance
        # ----------------------------------------------------
        self.bsde = bsde


        # ----------------------------------------------------
        # Containers for recording training statistics
        # ----------------------------------------------------
        self.loss_rec = []          # validation loss  # Ques
        self.val_rec = []           # validation value error
        self.train_loss_rec = []    # training loss

        # ----------------------------------------------------
        # True lower bound for control (if provided)
        # ----------------------------------------------------
        if "alow" not in self.eqn_config:  # Ques
            self.true_ld = 0.0
        else:
            self.true_ld = self.eqn_config.alow

        # ----------------------------------------------------
        # Simulation method
        # ----------------------------------------------------
        if "simulation" not in self.eqn_config:
            self.eqn_config.simulation = ""

        # ----------------------------------------------------
        # Queueing example identifier (optional)
        # ----------------------------------------------------
        if "queue_example" not in self.eqn_config:
            self.queue_example = ""
        else:
            self.queue_example = self.eqn_config.queue_example

        # ----------------------------------------------------
        # Flag indicating whether control is enabled
        # ----------------------------------------------------
        if "control" in self.train_config:
            self.control = True

        # ----------------------------------------------------
        # Critic neural network model
        # ----------------------------------------------------
        self.model_critic = CriticModel(config, bsde)
        
        # ----------------------------------------------------
        # Learning rate schedule for critic optimizer
        # ----------------------------------------------------
        lr_schedule_critic = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries_critic, 
            self.net_config.lr_values_critic
        )

        # ----------------------------------------------------
        # Adam optimizer for critic network
        # ----------------------------------------------------
        self.optimizer_critic = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule_critic, 
            epsilon=1e-8
        )
      
        # ----------------------------------------------------
        # Discount factor (gamma)
        # ----------------------------------------------------
        self.gamma = self.eqn_config.discount

        # ----------------------------------------------------
        # Steady-state flag: true if gamma ~ 0
        # ----------------------------------------------------
        if self.gamma < 1e-9:
            self.steady_flag = True  
        else:
            self.steady_flag = False

    # ============================================================
    # Generate simulation samples
    # ============================================================
    def gen_samples(self):
        # Record starting time
        start_time = time.time()

        # ----------------------------------------------------
        # Read sampling and simulation parameters
        # ----------------------------------------------------
        num_sample = self.net_config.batch_size
        T = self.eqn_config.total_time_critic
        N = self.eqn_config.num_time_interval_critic
        Total_iterations = self.net_config.num_iterations
        simulation_method = self.eqn_config.simulation

        # ----------------------------------------------------
        # Log basic problem information
        # ----------------------------------------------------
        logging.info('mu: %.2f, R: %.2f, rho: %.2f, dim: %d, gamma: %.2f ' 
                     %(self.bsde.mu[0], self.eqn_config.R, 
                       self.eqn_config.rho, self.bsde.dim, self.gamma)
                    )
        logging.info('N: %d , T: %.2f, num_sample: %d, batch_size: %d' 
                     %(N, T, Total_iterations, self.net_config.batch_size) )

        self.y_sample, self.dw_sample, self.du_sample = self.bsde.gen_samples(
            num_sample, T, N, Total_iterations + 20, simulation_method
        )

        expected_steps = N * (Total_iterations + 20)
        expected_state_steps = (N + 1) * (Total_iterations + 20)
        expected_shapes = {
            "Brownian increments": (num_sample, self.bsde.dim, expected_steps),
            "states": (num_sample, self.bsde.dim, expected_state_steps),
            "control pushes": (num_sample, len(RS_CONTROL_ORDER), expected_steps),
        }
        arrays = {
            "Brownian increments": self.dw_sample,
            "states": self.y_sample,
            "control pushes": self.du_sample,
        }
        for label, array in arrays.items():
            if array.shape != expected_shapes[label]:
                raise ValueError(
                    f"{label} shape {array.shape} does not match {expected_shapes[label]}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{label} contains non-finite values")
        if not np.allclose(self.du_sample[:, 3, :], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("U2 backup push must be identically zero")

        logging.info('Generated samples. Time: %3u' % (time.time() - start_time))
        logging.info(self.y_sample.shape)


    # ============================================================
    # Generate filename suffix for logs and models
    # ============================================================
    def gen_filename(self):
        simulation_method = self.eqn_config.simulation
        T = self.eqn_config.total_time_critic
        N = self.eqn_config.num_time_interval_critic
        Total_iterations = self.net_config.num_iterations

        # Base suffix including architecture and equation parameters
        suffix = (
            self.eqn_config.eqn_name + 
            self.queue_example + "_" + 
            self.net_config.transformation + "_" + 
            str(self.net_config.num_hiddens_critic[0]) + "_"  
        )
        suffix += (
            str(self.bsde.mu[0]) + "_" + 
            str(self.eqn_config.R) + "_" + 
            str (self.bsde.gamma) + "_" + 
            str (self.bsde.a) + "_" + 
            str(T)
        )
        suffix += (
            "_" + str(N) + "_" + 
            str(Total_iterations) + "_dim=" + 
            str(self.bsde.dim) +  
            simulation_method + "_" + 
            self.train_config.TD_type + "_" +
            self.net_config.activation
        )

        # Additional parameters for Rybko–Stolyar
        if self.eqn_config.eqn_name == "RybkoStolyar":
            suffix += "_x0=" + str(self.eqn_config.x0)
            suffix += "_v1=" + str(self.eqn_config.v1) + "_v2=" + str(self.eqn_config.v2)
            add_cost_tag = "_".join(
                f"{float(value):.12g}" for value in self.raw_boundary_pushing_cost
            )
            suffix += "_addcost=" + add_cost_tag
            suffix += "_lr" +  str(self.net_config.lr_values_critic[0])
            suffix += "_eta=" + str(self.eqn_config.eta_upper)

        # Sanitize filename
        suffix = suffix.replace(" ", "").replace(",", "_").replace("[", "").replace("]", "")
        return suffix

    # ============================================================
    # Training loop for the critic network
    # ============================================================
    def train(self):
        start_time = time.time()

        # ----------------------------------------------------
        # Pace controls how fast the lower bound is relaxed
        # ----------------------------------------------------
        pace = self.train_config.pace if "pace" in self.train_config else 0
        
        true_ld = self.true_ld;
        simulation_method = self.eqn_config.simulation
        T = self.eqn_config.total_time_critic
        N = self.eqn_config.num_time_interval_critic
        Total_iterations = self.net_config.num_iterations
        suffix = self.gen_filename()
        os.makedirs("models", exist_ok=True)

        logging.info(suffix)
        logging.info("LAMBDA: %d" %(self.LAMBDA))
        
        # ----------------------------------------------------
        # Prepare validation dataset (last 20 trajectories)
        # ----------------------------------------------------
        for step in range(self.net_config.num_iterations, self.net_config.num_iterations + 20):
            cur_start = step * N 
            cur_end = (step + 1) * N 
            cur_start_y = step * (N + 1) 
            cur_end_y = (step + 1) * (N + 1)
            if step == self.net_config.num_iterations:
                # batch size * dim of state * length of time.
                validation_w_sample = self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end]
                validation_y_sample = self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y]
                validation_u_sample = self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end]
            else:
                validation_w_sample = np.append(
                    validation_w_sample, 
                    self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end], 
                    axis = 0
                )
                validation_y_sample = np.append(
                    validation_y_sample, 
                    self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y], 
                    axis = 0
                )
                validation_u_sample = np.append(
                    validation_u_sample, 
                    self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end], 
                    axis = 0
                )

        valid_data_critic = tf.constant(0.0, dtype=tf.float64), validation_w_sample, validation_y_sample, validation_u_sample

        # ----------------------------------------------------
        # Determine total number of epochs
        # ----------------------------------------------------
        if "epoch" in self.train_config:
            epoch_total = self.train_config.epoch
        else:
            epoch_total= self.bsde.dim * 2 + self.bsde.a + 20;
        
        logging.info('Epoch Total: %d' % (epoch_total))

        # ----------------------------------------------------
        # Main SGD training loop
        # ----------------------------------------------------
        for epoch in range(epoch_total):
            logging.info("Epoch: %d",epoch)
            
            discard_num = 1000 if epoch > 0 else 0
            
            for step in range(self.net_config.num_iterations ):
                if epoch > 0 and step < discard_num:
                    continue

                # Compute sample indices
                cur_start = step * N 
                cur_end = (step + 1) * N 
                cur_start_y = step * (N + 1) 
                cur_end_y = (step + 1) * (N + 1)
                
                # ------------------------------------------------
                # Compute lower bound for control
                # ------------------------------------------------
                if pace == 0:
                    a_lowbound = max(true_ld,0.0)
                else:
                    a_lowbound = max(
                        true_ld, 
                        (self.bsde.a * 0.8) 
                        - (step + epoch * (self.net_config.num_iterations - discard_num)) 
                        / 40 / self.bsde.dim / pace
                    )
                   
                a_lowbound = tf.cast(a_lowbound, dtype=tf.float64)
                
                # ------------------------------------------------
                # Logging and validation
                # ------------------------------------------------
                if (step % self.net_config.logging_frequency == 0) or (step == self.net_config.num_iterations - 1):
                    logging.info("lower_bound: %.3f",a_lowbound)

                    x0 = np.ones([self.net_config.batch_size, self.bsde.dim]) * 2
                    
                    valid_loss_critic, err_value = self.loss_critic(
                        valid_data_critic, training=False, use_NN=True
                    )
                    valid_loss_critic = valid_loss_critic.numpy()
                    self.loss_rec.append(valid_loss_critic)
                    self.val_rec.append(err_value)
                    
                    cur_grad = self.compute_grad(x0).numpy()
                    cur_value = self.compute_value(x0).numpy()
         
                    elapsed_time = time.time() - start_time

                    train_data = (
                        a_lowbound, 
                        self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end], 
                        self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y], 
                        self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end]
                    )
                    
                    train_loss_critic, _ = self.loss_critic(
                        train_data, training=False, use_NN=True
                    )
            
                    self.train_loss_rec.append(train_loss_critic.numpy())
                
                    logging.info(
                        "validation loss: %.6f, training loss %.6f",   
                        valid_loss_critic, train_loss_critic
                    )
                    logging.info(
                        "step: %5u, Current Val: %.4f,  cur_V: %.4f,  cur_grad: %.4f, elapsed time: %3u" 
                        % (step, err_value, cur_value, cur_grad, elapsed_time)
                    )

                    
                    fig_folder = "figs_" +  self.net_config.activation + "/"
                    log_folder = "logs_" +  self.net_config.activation + "/"
                    os.makedirs(fig_folder, exist_ok=True)
                    os.makedirs(log_folder, exist_ok=True)
                    
                    # Save model checkpoints
                    filename = "models/" + suffix
                    self.model_critic.NN_value.save(filename + ".keras")
                    self.model_critic.NN_value_grad.save(filename + "_grad.keras")

                
                # ------------------------------------------------
                # Perform one gradient update step
                # ------------------------------------------------
                train_data = (
                    a_lowbound, 
                    self.dw_sample[:self.net_config.batch_size, :, cur_start:cur_end], 
                    self.y_sample[:self.net_config.batch_size, :, cur_start_y:cur_end_y], 
                    self.du_sample[:self.net_config.batch_size, :, cur_start:cur_end]
                )
                self.train_step_critic(train_data)
                
                # ------------------------------------------------
                # Save final model and logs
                # ------------------------------------------------
                if (step == self.net_config.num_iterations - 1):
                    filename = "models/" + suffix
                    self.model_critic.NN_value.save("models/" + suffix + ".keras")
                    np.savetxt(
                        log_folder + suffix, 
                        np.stack([self.val_rec, self.loss_rec, self.train_loss_rec])
                    )

    # ============================================================
    # Critic loss function
    # ============================================================
    def loss_critic(self, inputs, training, use_NN):
        delta, cur_val, negative_loss = self.model_critic(inputs, training=training, use_NN=use_NN)
       
        loss = tf.math.reduce_variance(delta) / self.eqn_config.total_time_critic ** 2 
        return loss + self.LAMBDA * negative_loss, cur_val

    # ============================================================
    # Compute gradients of critic loss
    # ============================================================
    def grad_critic(self, inputs, training, use_NN):
        with tf.GradientTape(persistent=True) as tape:
            loss_critic, temp= self.loss_critic(inputs, training, use_NN)
            
        grad = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        del tape
        return grad

    # ============================================================
    # Single optimization step for critic network
    # ============================================================
    @tf.function
    def train_step_critic(self, train_data):
        grad = self.grad_critic(train_data, training=False, use_NN = True)
        self.optimizer_critic.apply_gradients(zip(grad, self.model_critic.trainable_variables))
        
    # ============================================================
    # Compute average gradient of value function
    # ============================================================
    def compute_grad(self, x0):
        return tf.reduce_mean(
            self.model_critic.NN_value_grad(x0, training=False, need_grad=False)
        )

    # ============================================================
    # Compute average value function at initial state
    # ============================================================
    def compute_value(self, x0):
        return tf.reduce_mean(
            self.model_critic.NN_value(x0, training=False, need_grad=False)
        )
  
class CriticModel(tf.keras.Model):
    """
    Critic model for BSDE-based control problems.
    This model evaluates the value function V and its gradient,
    and computes the BSDE residual used as the critic loss.
    """
    
    def __init__(self, config, bsde):
        # Initialize parent Keras model
        super(CriticModel, self).__init__()

        # Store configuration objects
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.train_config = config.train_config

        # Store BSDE object (contains dynamics, costs, true solutions, etc.)
        self.bsde = bsde

        # Neural network approximating the value function V(x)
        self.NN_value = DeepNN(config, "critic")
        
        # Neural network approximating the gradient of the value function ∇V(x)
        # This enables double parametrization (V and ∇V are learned separately)
        self.NN_value_grad = DeepNN(config, "critic_grad")

        # Discount factor gamma
        self.gamma = config.eqn_config.discount

        # Check whether this is a steady-state problem
        if self.gamma < 1e-9:
            self.steady_flag = True
        else:
            self.steady_flag = False

        # Extract eqn_config fields as plain scalars to avoid Keras 3.x TrackedDict issue
        self.dim = config.eqn_config.dim
        self.total_time_critic = config.eqn_config.total_time_critic
        self.num_time_interval_critic = config.eqn_config.num_time_interval_critic
        self.boundary_pushing_cost = tf.constant(
            config.eqn_config.boundary_pushing_cost, dtype=tf.float64
        )

        # Whether the problem involves control
        if "control" in self.train_config:
            self.control = True
            
       
    def call(self, inputs, training, use_NN = True):
        """
        Forward pass of the critic model.

        Args:
            inputs: tuple containing
                - a_lowbound: lower bound of control
                - dw: Brownian motion increments
                - y_sample: simulated state trajectories
                - du: boundary pushing control increments
            training: boolean indicating training or inference mode
            use_NN: whether to use neural network approximations
                    or true analytical solutions

        Returns:
            delta: BSDE residual
            rtn_val: critic loss value
            negative_loss: penalty loss (constraints / feasibility)
        """

        # Accumulated penalty loss
        negative_loss = 0.0 
        
        # State dimension
        dim = self.dim

        # Unpack inputs
        a_lowbound, dw, y_sample, du = inputs
        tf.debugging.assert_equal(
            tf.shape(du)[1],
            len(RS_CONTROL_ORDER),
            message="RS boundary pushes must use control order L1,L2,U1,U2",
        )

        # Whether to use double parametrization (V and ∇V separately)
        flag_double_parametrization = True

        # Number of Monte Carlo samples
        num_sample = np.shape(dw)[0]

        # Initialize BSDE integral value
        y = 0

        # Initialize discount factor (broadcasted over samples)
        discount = 1  

        # Time step size
        dt = (
            self.total_time_critic
            / self.num_time_interval_critic
        )

        # Time iteration for BSDE integration
        for t in range(self.num_time_interval_critic):

            # --------------------------------------------------
            # Compute gradient of value function ∇V
            # --------------------------------------------------
            if flag_double_parametrization:
                # Gradient is directly produced by a neural network
                grad_t = self.NN_value_grad(y_sample[:,:,t], training=training, need_grad=False)
            else:
                # Single parametrization: compute gradient via autodiff
                with tf.GradientTape() as g:
                    temp_x = tf.convert_to_tensor(y_sample[:,:,t])
                    g.watch(temp_x)
                    grad_t = g.gradient(self.NN_value(temp_x, training=training, need_grad=False), temp_x)
           

            # --------------------------------------------------
            # Running cost (generator F of the BSDE)
            # --------------------------------------------------
            if self.control:
                if use_NN:
                    # Use neural network gradient
                    w, temp_neg_loss = self.bsde.w_tf(
                        y_sample[:,:,t], grad_t, a_lowbound
                    ) 
                else:
                    # Use true gradient (for debugging / benchmarking)
                    w, temp_neg_loss = self.bsde.w_tf(
                        y_sample[:,:,t], self.bsde.V_grad_true(y_sample[:,:,t])
                    ) 

                # Accumulate penalty losses
                negative_loss += temp_neg_loss
            
            else:
                # No control case
                w, temp_neg_loss = self.bsde.w_tf(
                    y_sample[:,:,t]
                )

            # --------------------------------------------------
            # Drift term: integral of running cost
            # --------------------------------------------------
            delta_y_drift = w * discount * dt
            y += delta_y_drift 
            
            # --------------------------------------------------
            # Diffusion term: stochastic integral
            # --------------------------------------------------
            # sigma is num_sample x dim x dim_w, NN_value_grad is num_sample x dim, dw[:,:,t] is num_sample x dim_w
            delta_y_diffusion =  dw[:,:,t]
            if use_NN:
                delta_y_diffusion = tf.reduce_sum(
                    delta_y_diffusion * grad_t, 
                    axis=1, 
                    keepdims=True
                )
            else:
                delta_y_diffusion = tf.reduce_sum(
                    delta_y_diffusion * self.bsde.V_grad_true(y_sample[:, :, t]), 
                    axis=1, 
                    keepdims=True
                )
                
            delta_y_diffusion *= discount 
            y -= delta_y_diffusion 
            
            # --------------------------------------------------
            # Boundary pushing control term
            # --------------------------------------------------
            delta_y_pushing = du[:,:,t]
            delta_y_pushing = tf.reduce_sum(
                delta_y_pushing * self.boundary_pushing_cost,
                axis=1, keepdims=True
            )
            delta_y_pushing *= discount 
            y += delta_y_pushing
            
            # Update discount factor
            discount *= np.exp(-self.gamma * dt)

        # Terminal state
        y_final = y_sample[:,:,-1]

        # --------------------------------------------------
        # BSDE residual
        # --------------------------------------------------
        if use_NN:    
            delta = (
                self.NN_value(y_sample[:,:,0], training=training, need_grad=False)
                - y
                - self.NN_value(y_final, training=training, need_grad=False) * discount
            )
        else:
            delta = (
                self.bsde.V_true(y_sample[:,:,0]) 
                - y 
                - self.bsde.V_true(y_final) * discount
            )

        # --------------------------------------------------
        # Normalization of critic loss
        # --------------------------------------------------
        if self.steady_flag:
            # Steady-state normalization
            rtn_val = (
                -tf.reduce_mean(delta)
                / self.total_time_critic
            )
        else:
            # Discounted infinite-horizon normalization
            rtn_val = (
                -tf.reduce_mean(delta)
                / (1 - np.exp(-self.gamma * self.total_time_critic)))


        return delta, rtn_val, negative_loss

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class DeepNN(tf.keras.Model):
    """
    Deep fully-connected neural network used as:
    - critic: approximates the value function V(x)
    - critic_grad: approximates the gradient of the value function ∇V(x)

    This class supports Keras serialization for model saving/loading.
    """
    def __init__(self, config, AC, **kwargs):
        """
        Initialize the neural network.

        Args:
            config: configuration object (can be dict or Munch)
            AC: string flag indicating network role
                - "critic": value function network
                - "critic_grad": gradient network
        """
        super(DeepNN, self).__init__(**kwargs)

        # Actor/Critic identifier
        self.AC = AC
        
        # Convert Munch config to dictionary if necessary
        if isinstance(config, Munch):
            config = config.toDict()

        # Extract equation-related configuration
        self.eqn_config = config.get('eqn_config', {})
        self.eqn = self.eqn_config.get('eqn_name', '')

        # Dimension of the state space
        dim = self.eqn_config.get('dim', 1)

        # Hidden layer sizes for critic networks
        num_hiddens = config.get('net_config', {}).get('num_hiddens_critic', [])

        # Activation function name
        self.activation = config.get('net_config', {}).get(
            'activation', ''
        )

        # --------------------------------------------------
        # Build hidden layers (no activation here)
        # Activations are applied manually in `call`
        # --------------------------------------------------
        self.dense_layers = [
            tf.keras.layers.Dense(
                num_hiddens[i], 
                use_bias=False, 
                activation=None
            )
            for i in range(len(num_hiddens))
        ]

        # --------------------------------------------------
        # Output layer depends on network role
        # --------------------------------------------------
        if AC == "critic":
            # Value function V(x): scalar output
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    1, 
                    activation=None, 
                    use_bias=True
                )
            )
        elif AC == "critic_grad":
            # Gradient ∇V(x): vector output of dimension `dim`
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    dim, 
                    activation=None, 
                    use_bias=False
                )
            )

    def call(self, x, training=False, need_grad=False):
        """
        Forward pass of the network.

        Args:
            x: input tensor (state variable)
            training: whether the model is in training mode
            need_grad: whether to compute gradient of output w.r.t. input

        Returns:
            If AC == "critic" and need_grad:
                (output, gradient)
            Else:
                output only
        """

        # Small epsilon (not actively used here, but kept for stability/debugging)
        eps = 1e-4

        # Gradient tape is used only when gradient w.r.t. input is required
        with tf.GradientTape() as g:

            # Watch input x if gradient is needed
            if self.AC == "critic" and need_grad:
                g.watch(x)

            # Forward propagation
            y = x
            for i in range(len(self.dense_layers) - 1):
                # Linear transformation
                y = self.dense_layers[i](y)

                # Apply chosen activation function
                if self.activation == "relu":
                    y = tf.nn.relu(y)
                elif self.activation == "leaky":
                    y = tf.nn.leaky_relu(y)
                elif self.activation == "elu":
                    y = tf.nn.elu(y)
                elif self.activation == "sigmoid":
                    y = y + tf.math.sigmoid(y) - 1/2
                if self.activation == "gelu":
                    y = tf.nn.gelu(y)

            # Final linear layer (no activation)
            y = self.dense_layers[-1](y)

        # --------------------------------------------------
        # Return value and optionally gradient
        # --------------------------------------------------
        if self.AC == "critic" and need_grad:
            # Return value function and its gradient w.r.t input
            return y, g.gradient(y, x)
        else:
            # Return value only
            return y

    def get_config(self):
        """
        Required for Keras serialization.
        Stores all information needed to reconstruct the model.
        """
        config = super(DeepNN, self).get_config()
        config.update({
            "AC": self.AC,
            "eqn_config": self.eqn_config,
            "activation": self.activation,
            "dense_layers": [
                tf.keras.layers.serialize(layer) 
                for layer in self.dense_layers
            ],
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruct the model from a serialized config.
        """
        
        # Convert config to Munch if needed
        if isinstance(config, dict):
            config = Munch(config)

        # Deserialize dense layers
        dense_layers = [
            tf.keras.layers.deserialize(layer_config) 
            for layer_config in config.dense_layers
        ]

        # Recreate model instance
        instance = cls(config.eqn_config, config.AC)

        # Assign restored layers
        instance.dense_layers = dense_layers
        return instance
