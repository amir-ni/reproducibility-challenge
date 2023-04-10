import json
import os
from data_loader import get_dataset

from AGCRN import AGCRN
import torch
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

    logfile = os.path.join(log_directory, f'test-s{seed}-dim{embedding_dim}.log')
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger



def metrics(pred, true):
    mae = torch.mean(torch.abs(true-pred))
    rmse = torch.sqrt(torch.mean((pred - true) ** 2))

    mask = torch.gt(true, 0)
    pred_mask = torch.masked_select(pred, mask)
    true_mask = torch.masked_select(true, mask)
    mape = torch.mean(
        torch.abs(torch.div((true_mask - pred_mask), true_mask)))

    return mae, rmse, mape


if __name__ == '__main__':

    with open("config.json","r") as f:
        config_list = json.load(f)

    for config in config_list:

        model = AGCRN(config["num_node"], 1, config["hidden_dim"], 1,
                    config["horizon"], config["num_layers"], config["embed_dim"], config["k"])

        model.load_state_dict(torch.load(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "pretrained-model", config['dataset'], f'best_model-s{config["seed"]}-dim{config["embed_dim"]}.pth')))

        model.eval()
        model = model.to(device)

        _, _, test_loader, scaler = get_dataset(config)


        # make and setup log directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_directory = os.path.join(current_dir, 'logs', config['dataset'])
        best_model_directory = os.path.join(current_dir, 'pretrained-model', config['dataset'])
        os.makedirs(log_directory, exist_ok=True)
        os.makedirs(best_model_directory, exist_ok=True)

        # logger
        logger = get_logger(log_directory,config['dataset'], config['seed'], config['embed_dim'])
        metics_path = os.path.join(best_model_directory, f'test_metrics-s{config["seed"]}-dim{config["embed_dim"]}.json')

        # Evaluation

        logger.info(f"===== Start Evaluation on Test Set  for {config['dataset']} =====")
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data = data[..., :1]
                label = target[..., :1]
                output = model(data)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = torch.cat(y_pred, dim=0)

        metrics_list = []
        
        for t in range(y_true.shape[1]):
            mae, rmse, mape = metrics(
                y_pred[:, t, ...], y_true[:, t, ...])
            
            # save metrics to list
            metrics_list.append({"horizon": t+1, "mae": mae.item(), "rmse": rmse.item(), "mape": mape.item() * 100})

            logger.info(f"Horizon {t+1}, MAE: {round(mae.item(),2)}, RMSE: {round(rmse.item(),2)}, MAPE: {round(mape.item()*100,2)}%")

        
        mae, rmse, mape = metrics(
            y_pred, y_true)
        
        # save metrics to list
        metrics_list.append({"horizon": "average", "mae": mae.item(), "rmse": rmse.item(), "mape": mape.item() * 100})

        logger.info(f">> Average Horizon, MAE: {round(mae.item(),2)}, RMSE: {round(rmse.item(),2)}, MAPE: {round(mape.item()*100,2)}%")
        
        # save metrics to json file
        with open(metics_path, 'w') as f:
            json.dump(metrics_list, f, indent=4)

