"""
The main file to solve RBM control solver problems.
"""

import json
import munch
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import equation as eqn
from solver import ControlSolver
from rs_reflection import expand_rs_boundary_pushing_cost

# ------------------------------------------------------------
# Detect available physical GPUs using TensorFlow
# ------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')

# Print detected GPUs for debugging / verification
print("GPUs:", gpus)

# ------------------------------------------------------------
# Enable dynamic memory allocation on the first GPU (if exists)
# This prevents TensorFlow from allocating all GPU memory upfront
# ------------------------------------------------------------
if gpus:
    try:
        # Allow memory growth on GPU:0
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        # Ignore errors (e.g., GPU already initialized)
        pass

# ------------------------------------------------------------
# Log whether the program is running on GPU or CPU
# ------------------------------------------------------------
if tf.config.list_physical_devices('GPU'):    
    logging.info("Running with GPU acceleration")
else:
    logging.warning("Running on CPU only")

# ------------------------------------------------------------
# Command-line flags for experiment configuration
# ------------------------------------------------------------

# Path to the JSON configuration file
flags.DEFINE_string('config_path', 
                    'configs/RybkoStolyar.json',
                    """The path to load json file."""
                   )
# Name of the experiment (used as logging / output prefix)
flags.DEFINE_string('exp_name', None,
                    """The name of numerical experiments, prefix for logging""")

# ------------------------------------------------------------
# Optional command-line overrides for equation parameters
# These values override those in the JSON config if provided
# ------------------------------------------------------------
flags.DEFINE_float('eta_upper', None, "Value for eta_upper")
flags.DEFINE_float('mu', None, "Value for mu")
flags.DEFINE_float('discount', None, "Value for discount")
flags.DEFINE_float('v1', None, "Value for v1")
flags.DEFINE_float('v2', None, "Value for v2")
flags.DEFINE_float('pricing_multiply', None, "Value for pricing_multiply")

# Global FLAGS object containing all parsed command-line arguments
FLAGS = flags.FLAGS
# Directory where logs and outputs will be written
FLAGS.log_dir = './logs'

# ============================================================
# Main entry point for the experiment
# ============================================================
def main(argv):

    # argv is unused (flags are accessed via FLAGS)
    del argv

    # --------------------------------------------------------
    # Set experiment name
    # If not provided, use the config file name (without extension)
    # --------------------------------------------------------
    if FLAGS.exp_name is None: # use config name as exp_name
        FLAGS.exp_name = os.path.splitext(os.path.basename(FLAGS.config_path))[0]

    # --------------------------------------------------------
    # Load configuration from JSON file
    # --------------------------------------------------------
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)

    # Convert dictionary to Munch object for dot-access
    config = munch.munchify(config)

    # --------------------------------------------------------
    # Override config parameters using command-line flags
    # --------------------------------------------------------
    for name in ('eta_upper', 'mu', 'discount', 'v1', 'v2', 'pricing_multiply'):
        value = getattr(FLAGS, name)
        if value is not None:
            config.eqn_config[name] = float(value)

    # --------------------------------------------------------
    # Log final parameter values for reproducibility
    # --------------------------------------------------------
    logging.info(f"Updated eta_upper: {config.eqn_config.eta_upper}")
    logging.info(f"Updated mu: {config.eqn_config.mu}")
    logging.info(f"Updated discount: {config.eqn_config.discount}")
    logging.info(f"Updated v1: {config.eqn_config.v1}")
    logging.info(f"Updated v2: {config.eqn_config.v2}")
    logging.info(f"Updated pricing_multiply: {config.eqn_config.pricing_multiply}")

    # --------------------------------------------------------
    # Expand the raw two-parameter cost into the fixed four-control order.
    # --------------------------------------------------------
    add_cost = list(config.eqn_config.boundary_pushing_cost)
    config.eqn_config.boundary_pushing_cost_raw = add_cost
    config.eqn_config.boundary_pushing_cost = expand_rs_boundary_pushing_cost(
        add_cost, config.eqn_config.v1, config.eqn_config.v2
    )

    # Log updated boundary cost
    logging.info(f"Updated boundary_pushing_cost: {config.eqn_config.boundary_pushing_cost}")

    # --------------------------------------------------------
    # Instantiate the BSDE / control equation dynamically
    # --------------------------------------------------------
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)

    # --------------------------------------------------------
    # Set TensorFlow default floating-point precision
    # --------------------------------------------------------
    tf.keras.backend.set_floatx(config.net_config.dtype)

    # Dimension of the state space
    dim = config.eqn_config.dim

    # --------------------------------------------------------
    # Create log directory if it does not exist
    # --------------------------------------------------------
    os.makedirs(FLAGS.log_dir, exist_ok=True)

    # Prefix for experiment outputs; parameters in the name keep parallel jobs from overwriting each other
    path_prefix = os.path.join(
        FLAGS.log_dir,
        f"{FLAGS.exp_name}_eta={config.eqn_config.eta_upper}_mu={config.eqn_config.mu}",
    )

    # --------------------------------------------------------
    # Save final configuration for reproducibility
    # --------------------------------------------------------
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    # --------------------------------------------------------
    # Configure absl logging format and verbosity
    # --------------------------------------------------------
    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    # --------------------------------------------------------
    # Initialize and run the control solver
    # --------------------------------------------------------
    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)

    # Create control solver instance
    Control_solver = ControlSolver(config, bsde)

    Control_solver.gen_samples()

    # Train the model / control policy
    Control_solver.train()

# ============================================================
# Program entry point (absl-style)
# ============================================================
if __name__ == '__main__':
    app.run(main)
