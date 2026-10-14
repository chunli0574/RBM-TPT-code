"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).
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

# ------------------------------------------------------------
# Detect available physical GPUs using TensorFlow
# ------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')

print("GPUs:", gpus)

# ------------------------------------------------------------
# Enable dynamic memory allocation on the first GPU (if exists)
# ------------------------------------------------------------
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

# ------------------------------------------------------------
# Log whether the program is running on GPU or CPU
# ------------------------------------------------------------
if tf.config.list_physical_devices('GPU'):
    logging.info("Running with GPU acceleration")
else:
    logging.warning("Running on CPU only")

flags.DEFINE_string('config_path', 'configs/crissCross.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', None,
                    """The name of numerical experiments, prefix for logging""")

flags.DEFINE_boolean('dump', True,
                    """generate the data or load the existing data""")

flags.DEFINE_float('eta_upper', None, "Value for eta_upper")
flags.DEFINE_float('mu', None, "Value for mu")
flags.DEFINE_float('discount', None, "Value for discount")
flags.DEFINE_float('a', None, "Value for a")
flags.DEFINE_string('num_hiddens_critic', None, "Value for num_hiddens_critic (comma-separated)")
flags.DEFINE_float('pace', None, "Value for pace")


FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array

def main(argv):
    del argv
    if FLAGS.exp_name is None: # use config name as exp_name
        FLAGS.exp_name = os.path.splitext(os.path.basename(FLAGS.config_path))[0]
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)

    #
    if hasattr(FLAGS, 'eta_upper') and hasattr(FLAGS, 'mu') and hasattr(FLAGS, 'discount'):
        config.eqn_config.eta_upper = float(FLAGS.eta_upper)
        config.eqn_config.mu = float(FLAGS.mu)
        config.eqn_config.discount = float(FLAGS.discount)
        if FLAGS.a is not None:
            config.eqn_config.a = float(FLAGS.a)
        if FLAGS.num_hiddens_critic is not None:
            num_hiddens_value = int(FLAGS.num_hiddens_critic)
            config.net_config.num_hiddens_critic = [num_hiddens_value] * 4
        if FLAGS.pace is not None:
            pace = int(FLAGS.pace)
            config.train_config.pace = pace

    logging.info(f"Updated eta_upper: {config.eqn_config.eta_upper}")
    logging.info(f"Updated mu: {config.eqn_config.mu}")
    logging.info(f"Updated discount: {config.eqn_config.discount}")
    logging.info(f"Updated a: {config.eqn_config.a}")
    logging.info(f"Updated num_hiddens_critic: {config.net_config.num_hiddens_critic}")
    logging.info(f"Updated pace: {config.train_config.pace}")
    
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)


    tf.keras.backend.set_floatx(config.net_config.dtype)
    dim = config.eqn_config.dim


    
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')
            
    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    Control_solver = ControlSolver(config, bsde)
    
    dump_flag = FLAGS.dump
    Control_solver.gen_samples(dump = dump_flag, load = not dump_flag)

    Control_solver.train()

    
if __name__ == '__main__':
    app.run(main)
