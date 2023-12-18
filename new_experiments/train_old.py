import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

from model import *
from training_utils import *


logging.basicConfig(level=logging.INFO)
logging.info("Training Old Models")

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=800) # also 100 and 400
parser.add_argument("--save_path", type=str, default="models_eval")





# only use dim8 data
with open('data/datafile_dim8.pkl', "rb") as f:
    data_gen = pickle.load(f)
    
    
data_config = sim_config.dim8_config

obs_dim = data_config.obs_dim
latent_dim = data_config.latent_dim
action_dim = data_config.action_dim
t_max = data_config.t_max
step_size = data_config.step_size
encoder_latent_ratio = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = True
prior = ExponentialPrior.log_density
encoder_output_dim = latent_dim

logging.info(f"Encoder output dim: {encoder_output_dim}")

old_encoder = EncoderLSTM(
            obs_dim + action_dim,
            int(obs_dim * encoder_latent_ratio),
            encoder_output_dim,
            device=device,
            normalize=normalize,
        )

old_decoder = RocheExpertDecoder(
            obs_dim,
            encoder_output_dim,
            action_dim,
            14,
            1,
            roche=True,
            method="dopri5",
            device=device,
            ablate=False,
        )

old_vi = VariationalInference(old_encoder,
                                old_decoder,
                                prior_log_pdf=prior,
                                elbo=True,
                                )

params = (
            list(old_vi.encoder.parameters())
            + list(old_vi.decoder.output_function.parameters())
            + list(old_vi.decoder.ode.ml_net.parameters())
        )



if __name__ == "__main__":
    args = parser.parse_args()
    data_gen.set_device(device)
    data_gen.set_train_size(args.sample_size)
    logging.info(f"Sample size: {args.sample_size}")
    
    # create separate folders for each sample size
    path = os.path.join(args.save_path, str(args.sample_size))
    logging.info(f"Models will be saved to {path}")
    
    variational_training_loop(
        500,
        data_gen,
        old_vi,
        10,
        torch.optim.Adam(params, lr=1e-2),
        10,
        early_stop=10,
        print_future_mse=True,
        path=path,
        new = False,
    )