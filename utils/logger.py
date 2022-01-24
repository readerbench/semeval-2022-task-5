"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

helper for logging
NOTE: loggers are global objects use with caution
"""
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

_LOG_FMT = '%(asctime)s | %(levelname)s |   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')  # this is the global logger


def add_log_to_file(log_path):
    Path(log_path).parent.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


def log_tensorboard(writer, metric, epoch, step, total_steps, skip_validation=False):
    # train_loss: float
    # train_acc: float
    # train_prec: float
    # train_recall: float
    # train_f1: float
    # valid_loss: float
    # valid_acc: float
    # valid_prec: float
    # valid_recall: float
    # valid_f1: float
    if not isinstance(writer, SummaryWriter): 
        return
          

    writer.add_scalar('Train/Epoch_Loss', metric.train_loss, step, (epoch-1)*total_steps+step)
    writer.add_scalar('Train/F1', metric['train_f1'], step, (epoch-1)*total_steps+step)
    writer.add_scalar('Train/Precision', metric['train_prec'], step, (epoch-1)*total_steps+step)
    writer.add_scalar('Train/Recall', metric['train_recall'], step, (epoch-1)*total_steps+step)
    writer.add_scalar('Train/Accuracy', metric['train_acc'], step, (epoch-1)*total_steps+step)
    
    if not skip_validation:
        writer.add_scalar('Validation/Loss', metric.valid_loss, epoch)
        writer.add_scalar('Validation/F1', metric['valid_f1'], epoch)
        writer.add_scalar('Validation/Recall', metric['valid_recall'], epoch)
        writer.add_scalar('Validation/Precision', metric['valid_prec'], epoch)
        writer.add_scalar('Validation/Accuracy', metric['valid_acc'], epoch)
    