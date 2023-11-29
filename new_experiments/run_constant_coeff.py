import os
import torch
import pickle
import sim_config
from model import EncoderLSTM, ExponentialPrior
from new_models import NewEncoderLSTM
from new_models_cde import RocheCDEDecoder, CDEVariationalInference2, NewRocheCDE2
from training_utils import variational_training_loop, evaluate
import logging

# Set up logging
logging.basicConfig(filename="constant_coeff.log", level=logging.INFO)
logging.info("Starting constant coefficient experiment")


with open("data/datafile_dim8.pkl", "rb") as f:
    data_gen = pickle.load(f)

data_config = sim_config.dim8_config

obs_dim = data_config.obs_dim  # 40

latent_dim = data_config.latent_dim  # 8

action_dim = data_config.action_dim  # 1

t_max = data_config.t_max  # 14

step_size = data_config.step_size  # 1

encoder_latent_ratio = 2.0

encoder_output_dim = latent_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = True
prior = ExponentialPrior.log_density
ablate = False

sample_sizes = [800]


use_vanilla = False

training_iters = 500



if __name__ == "__main__":
    print("---" * 10)
    print("Running constant coefficient experiment")
    logging.info("Starting constant coefficient experiment")
    
    for sample_size in sample_sizes:
        dg = data_gen
        dg.set_device(device)
        dg.set_train_size(sample_size)
        logging.info("Sample size: {}".format(sample_size))

        model_name = "constant_coeff_CDE_{}".format(sample_size)
        vae2_path = os.path.join("new-models", model_name)
        
        encoder2 = NewEncoderLSTM(
                obs_dim + action_dim,
                int(obs_dim * encoder_latent_ratio),
                encoder_output_dim + 4,
                device=device,
                normalize=normalize,
            )

        decoder2 = RocheCDEDecoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim + 4,
            action_dim=action_dim,
            t_max=t_max,
            step_size=step_size,
            include_expert=True,
            ablate=False,
            use_beta=True,
            use_vanilla=use_vanilla,
            device=device,
            use_constant_coeff=True,
        )

        vae2 = CDEVariationalInference2(
            encoder=encoder2,
            decoder=decoder2,
            mc_size=100,
            kl_coeff1=1.0,  # 0.5 if slow to converge
            prior_log_pdf=prior,
            model_name=model_name,
        )
        
        # if model exists
        if os.path.exists(vae2_path):
            logging.info("Model exists, skipping")
        else:
            # do not include the parameters
            params2 = (
                list(vae2.encoder.parameters())
                + list(vae2.decoder.ode.ml_net.parameters())
                + list(vae2.decoder.output_function.parameters())
            )

            _, best_loss, time_elapsed = variational_training_loop(
                training_iters,
                dg,
                vae2,
                50,  # lower batch size for smaller number of samples
                torch.optim.Adam(params2, lr=1e-2),
                50,
                best_on_disk=1e9,
                early_stop=10,
                path="new-models/",
                print_future_mse=True,
                hybrid_cde=False,
            )
        
            logging.info("Training finished, taken {} seconds".format(time_elapsed))
        
        # load the best model
        logging.info("Loading best model...")
        
        vae2.encoder.load_state_dict(torch.load(vae2_path)['encoder_state_dict'])
        vae2.decoder.load_state_dict(torch.load(vae2_path)['decoder_state_dict'])

        logging.info("Evaluating...")
        output = evaluate(
            vae2,
         dg,
         50,
         5,
         mc_itr=50,
         real=False,
         hybrid_cde=False,
         )
        
        print("Final result: {}".format(output))
        
        print("---" * 10)
