import argparse
import logging
import torch
import pickle
import sim_config
from model import EncoderLSTM, ExponentialPrior
from new_models import NewEncoderLSTM
from new_models_cde import *
from training_utils import variational_training_loop



logging.basicConfig(level=logging.INFO)
logging.info("Training New Models")

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=800) # also 100 and 400
parser.add_argument("--use_vanilla", type=bool, default=False) # use vanilla CDE
parser.add_argument("--use_beta", type=bool, default=True) # use beta
parser.add_argument("--include_time", type=bool, default=False) # include time in CDE
parser.add_argument("--save_path", type=str, default="models_eval")
parser.add_argument("--num_restart", type=int, default=5) # restart as beta is random



        

if __name__ == "__main__":
    args = parser.parse_args()
    use_vanilla = args.use_vanilla
    use_beta = args.use_beta
    include_time = args.include_time
    
    logging.info(f"Using vanilla CDE: {use_vanilla}")
    logging.info(f"Using beta: {use_beta}")
    logging.info(f"Including time: {include_time}")
    
    for i in range(args.num_restart):
        logging.info(f"Restart {i}...")
        with open('data/datafile_dim8.pkl', "rb") as f:
            data_gen = pickle.load(f)
    
        data_config = sim_config.dim8_config

        obs_dim = data_config.obs_dim # 40

        latent_dim = data_config.latent_dim # 8

        action_dim = data_config.action_dim # 1

        t_max = data_config.t_max # 14

        step_size = data_config.step_size # 1

        encoder_latent_ratio = 2.0

        encoder_output_dim = latent_dim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        normalize = True
        prior = ExponentialPrior.log_density
        ablate = False
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
            use_beta=use_beta,
            use_vanilla=use_vanilla,
            include_time=include_time,
            use_constant_coeff=False,
            device=device)
        
        data_gen.set_device(device)
        data_gen.set_train_size(args.sample_size)
        
        logging.info("Sample size: {}".format(args.sample_size))
        
        path = os.path.join(args.save_path, f"new-models-{args.sample_size}")
        logging.info(f"Models will be saved to {path}")
        
        if use_beta:
            model_name = f"newVAE_CDE_beta_{args.sample_size}"
        else:
            model_name = f"newVAE_CDE_{args.sample_size}_no_beta"
            
        if use_vanilla:
            model_name += "_vanilla"
        
        if include_time:
            model_name += "_time"
            
        vae2 = CDEVariationalInference2(
        encoder=encoder2,
        decoder=decoder2,
        mc_size=100,
        kl_coeff1=1.0, #0.5 if slow to converge
        prior_log_pdf=prior,
        model_name=f"{model_name}",
        )

        params2 = (
            list(vae2.encoder.parameters())+
            list(vae2.decoder.ode.ml_net.parameters())+
            list(vae2.decoder.output_function.parameters())
        )

        if not use_vanilla:
            params2.append(vae2.decoder.ode.alpha)
            if use_beta:
                params2.append(vae2.decoder.ode.beta)
        
        variational_training_loop(
            500,
            data_gen,
            vae2,
            50, # lower batch size for smaller number of samples
            torch.optim.Adam(params2, lr=1e-2),
            10,
            best_on_disk=1e9,
            early_stop=1000,
            path=path,
            print_future_mse=True,
            hybrid_cde=False,
        )
        logging.info("#################### One Restart Done ####################")
