import os

import yaml
import argparse
import numpy as np

from models import *
import torch.backends.cudnn as cudnn
from training import Trainer
import os

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    parser.add_argument('--experiment_name', '-e',
                        dest="experiment_name",
                        help='experiment name',
                        default='test')

    parser.add_argument('--limit_data', '-l',
                        dest="limit_data",
                        help='Limit train and valid data',
                        type=float,
                        default=1.0)

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # tt_logger = TestTubeLogger(
    #     save_dir=config['logging_params']['save_dir'],
    #     name=config['logging_params']['name'],
    #     debug=False,
    #     create_git_tag=False,
    # )

    # For reproducibility
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[config['model_params']['name']](**config['model_params'])
    # experiment = VAEXperiment(model,
    #                           config['exp_params'])
    trainer = Trainer(vae_model=model, device=device, params=config, experiment_name=args.experiment_name,
                      limit_data=args.limit_data)
    trainer.train()

    # runner = Trainer(weights_save_path=f"{tt_logger.save_dir}",
    #                  min_epochs=1,
    #                  logger=tt_logger,
    #                  val_check_interval=100,
    #                  # train_percent_check=1.,
    #                  # val_percent_check=1.,
    #                  num_sanity_val_steps=5,
    #                  **config['trainer_params'])
    #
    # print(f"======= Training {config['model_params']['name']} =======")
    # runner.fit(experiment)
