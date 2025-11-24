import logging
import os
from from_root import from_root
from datetime import datetime
from logging.handlers import RotatingFileHandler

log_folder_name = 'logs'
log_folder_path = os.path.join(from_root(),log_folder_name)
os.makedirs(log_folder_path, exist_ok = True)

log_file_name = '{}.log'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
log_file_path = os.path.join(log_folder_path,log_file_name)

def configure_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)

    file_logger = RotatingFileHandler(log_file_path,maxBytes= 5 * 1024 * 1024,backupCount= 3)
    file_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    console_logger.setFormatter(formatter)
    file_logger.setFormatter(formatter)

    logger.addHandler(console_logger)
    logger.addHandler(file_logger)

configure_logger()