import sys
import numpy as np
import random
import torch
import pickle
import pandas as pd
import logging
import time
import json
from datetime import timedelta
import os
from subprocess import *
import tensorflow as tf
from pathlib import Path
from numba.core.errors import NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)
logging.getLogger('numba').setLevel(logging.WARNING)

def ensure_path(path):
    path = Path(path)
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)
    return path
    
def run_system_command(cmd: str,
                       shell: bool = False,
                       err_msg: str = None,
                       verbose: bool = True,
                       split: bool = True,
                       stdout=None,
                       stderr=None) -> int:
    if verbose:
        sys.stdout.write("System cmd: {}\n".format(cmd))
    if split:
        cmd = cmd.split()
    rc = call(cmd, shell=shell, stdout=stdout, stderr=stderr)
    if err_msg and rc:
        sys.stderr.write(err_msg)
        exit(rc)
    return rc
    
def fcall(fun):
    """
    Convenience decorator used to measure the time spent while executing
    the decorated function.
    :param fun:
    :return:
    """
    def wrapper(*args, **kwargs):

        logging.info("[{}] ...".format(fun.__name__))

        start_time = time.perf_counter()
        res = fun(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        logging.info("[{}] Done! {}s\n".format(fun.__name__, timedelta(seconds=runtime)))
        return res

    return wrapper


def set_random_seed(args):
    if "torch" in sys.modules:
        torch.manual_seed(args["random_seed"])
    if "tf" in sys.modules:
        tf.random.set_seed(args["random_seed"])
    np.random.seed(int(args["random_seed"]))
    random.seed(args["random_seed"])

def setup_logging():
    
    level = {
        "info" : logging.INFO, 
        "debug" : logging.DEBUG,
        "critical" : logging.CRITICAL
    }
    
    msg_format = '%(asctime)s:%(levelname)s: %(message)s'
    formatter = logging.Formatter(msg_format, datefmt = '%H:%M:%S')

    logging.getLogger().addHandler(logging.StreamHandler())
    
    file_handler = logging.FileHandler("logs/run.log", mode = "w")
    file_handler.setLevel(level=level["info"])
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger()
    logger.setLevel(level["info"])

class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                name = args.run_name,
                config=args
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')

    def log_vectordb(self, output_dir):
        vectordb_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_vectordb", type ="vectordb"
        )

        vectordb_artifact.add_dir(output_dir)
        self._wandb.log_artifact(vectordb_artifact)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [value.decode("utf-8") if isinstance(value, bytes) else value for value in obj.tolist()]
        return json.JSONEncoder.default(self, obj)
    
@fcall
def load_data(path):
    path = str(path)
    
    if ".json" in path:
        with open(path, encoding = "utf-8") as f:
            if ".jsonl" in path:
                data = [json.loads(line) for line in f]
            elif ".json" in path:
                content = f.read()
                data = json.loads(content)
    elif ".pickle" in path:
        with open(path, "rb") as f :
            data = pickle.load(f)
    elif ".pt" in path:
        data = torch.load(path)
    elif ".csv" in path:
        data = pd.read_csv(path, index_col=0)
    else:
        raise NotImplementedError("Don't know how to load a dataset of this type")
    
    return data

def save_submission(url_lst):
    submission = []
    for _ , title in url_lst:
        pass