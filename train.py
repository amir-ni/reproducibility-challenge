
from data_loader import get_dataset
from train_utils import train
from AGCRN import AGCRN
import numpy as np
import random
import torch
import os
import json
import logging


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = 'cuda:0'
torch.cuda.cudnn_enabled = False
torch.backends.cudnn.deterministic = True


def get_logger(log_directory, name, seed, embedding_dim):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logfile = os.path.join(log_directory, f'train-s{seed}-dim{embedding_dim}.log')
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':

    with open("config.json", "r") as f:
        config_list = json.load(f)

    for config in config_list:

        # set seed
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])

        # get dataset
        train_dataloader, validation_dataloader, _, std_scaler = get_dataset(
            config)

        # create model and optimizer
        model = AGCRN(config["num_node"], 1, config["hidden_dim"], 1,
                      config["horizon"], config["num_layers"], config["embed_dim"], config["k"])
        model = model.to(device)
        model.init_parameters()

        criterion = torch.nn.L1Loss().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.003)

        # make and setup directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_directory = os.path.join(current_dir, 'logs', config['dataset'])
        best_model_directory = os.path.join(
            current_dir, 'pretrained-model', config['dataset'])

        os.makedirs(log_directory, exist_ok=True)
        os.makedirs(best_model_directory, exist_ok=True)

        # logger
        logger = get_logger(log_directory, config['dataset'], config['seed'], config['embed_dim'])

        #best model path to save
        best_model_path = os.path.join(best_model_directory, f"best_model-s{config['seed']}-dim{config['embed_dim']}.pth")

        #losss save path
        loss_save_path = os.path.join(best_model_directory, f"train_loss-s{config['seed']}-dim{config['embed_dim']}.json")

        # Train & Validation loop
        train(100, model, criterion, optimizer, config['dataset'], train_dataloader, validation_dataloader,
              std_scaler, logger, best_model_path, loss_save_path)
