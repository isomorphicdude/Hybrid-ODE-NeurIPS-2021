"""Evaluate different models trained using 400 samples."""
import os
import torch
import pickle
import sim_config
from model import EncoderLSTM, ExponentialPrior
from model import RocheExpertDecoder, VariationalInference
from new_models import NewEncoderLSTM
from new_models_cde import RocheCDEDecoder, CDEVariationalInference2, NewRocheCDE2
from training_utils import variational_training_loop, evaluate
import logging

# Set up logging
logging.basicConfig(filename="constant_coeff.log", level=logging.INFO)
logging.info("Evaluating different models at times 2, 5, 10")

# Load data (only dim 8)
with open("data/datafile_dim8.pkl", "rb") as f:
    data_gen = pickle.load(f)
    
    
model_path_dict = {
    "LHM": "models_eval/VI_LSTMEncoder_HybridDecoder.pkl",
    "CDE_const_0.5": "models_eval/constant_coeff_CDE_400",
    "CDE_trainable": "models_eval/VAE_CDE2_400_new",
    "CDE_vanilla": "models_eval/VAE_vanillaCDE_include_time_400",
    "CDE_no_beta": "models_eval/VAE_CDE_include_time_nobeta_400",
}

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

####################

# LHM model (Qian et al. 2021)
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

old_vi.encoder.load_state_dict(torch.load(model_path_dict["LHM"])['encoder_state_dict'])
old_vi.decoder.load_state_dict(torch.load(model_path_dict["LHM"])['decoder_state_dict'])



####################
# CDE model (constant coefficient)
encoder_const = NewEncoderLSTM(
            obs_dim + action_dim,
            int(obs_dim * encoder_latent_ratio),
            encoder_output_dim + 4,
            device=device,
            normalize=normalize,
        )

decoder_const = RocheCDEDecoder(
    obs_dim=obs_dim,
    latent_dim=latent_dim + 4,
    action_dim=action_dim,
    t_max=t_max,
    step_size=step_size,
    include_expert=True,
    ablate=False,
    use_beta=True,
    use_vanilla=False,
    include_time=False,
    use_constant_coeff=True,
    device=device)

vae_const = CDEVariationalInference2(
    encoder=encoder_const,
    decoder=decoder_const,
    mc_size=100,
    kl_coeff1=1.0, #0.5 if slow to converge
    prior_log_pdf=prior,
)

vae_const.encoder.load_state_dict(torch.load(model_path_dict["CDE_const_0.5"])['encoder_state_dict'])
vae_const.decoder.load_state_dict(torch.load(model_path_dict["CDE_const_0.5"])['decoder_state_dict'])


####################
# CDE model (trainable coefficient)
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
    use_vanilla=False,
    include_time=False,
    device=device)

vae2 = CDEVariationalInference2(
    encoder=encoder2,
    decoder=decoder2,
    mc_size=100,
    kl_coeff1=1.0, #0.5 if slow to converge
    prior_log_pdf=prior,
)

vae2.encoder.load_state_dict(torch.load(model_path_dict["CDE_trainable"])['encoder_state_dict'])
vae2.decoder.load_state_dict(torch.load(model_path_dict["CDE_trainable"])['decoder_state_dict'])


####################
# CDE model (vanilla)
encoder_vanilla = NewEncoderLSTM(
            obs_dim + action_dim,
            int(obs_dim * encoder_latent_ratio),
            encoder_output_dim + 4,
            device=device,
            normalize=normalize,
        )

decoder_vanilla = RocheCDEDecoder(
     obs_dim=obs_dim,
    latent_dim=latent_dim + 4,
    action_dim=action_dim,
    t_max=t_max,
    step_size=step_size,
    include_expert=True,
    ablate=False,
    use_beta=False,
    use_vanilla=True,
    include_time=True,
    device=device)

vae_vanilla = CDEVariationalInference2(
    encoder=encoder_vanilla,
    decoder=decoder_vanilla,
    mc_size=100,
    kl_coeff1=1.0, #0.5 if slow to converge
    prior_log_pdf=prior,
)

vae_vanilla.encoder.load_state_dict(torch.load(model_path_dict["CDE_vanilla"])['encoder_state_dict'])
vae_vanilla.decoder.load_state_dict(torch.load(model_path_dict["CDE_vanilla"])['decoder_state_dict'])


####################
# CDE model (no beta)
encoder_no_beta = NewEncoderLSTM(
            obs_dim + action_dim,
            int(obs_dim * encoder_latent_ratio),
            encoder_output_dim + 4,
            device=device,
            normalize=normalize,
        )

decoder_no_beta = RocheCDEDecoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim + 4,
        action_dim=action_dim,
        t_max=t_max,
        step_size=step_size,
        include_expert=True,
        ablate=False,
        use_beta=False,
        use_vanilla=False,
        include_time=True,
        device=device)

vae_no_beta = CDEVariationalInference2(
    encoder=encoder_no_beta,
    decoder=decoder_no_beta,
    mc_size=100,
    kl_coeff1=1.0, #0.5 if slow to converge
    prior_log_pdf=prior,
)

vae_no_beta.encoder.load_state_dict(torch.load(model_path_dict["CDE_no_beta"])['encoder_state_dict'])
vae_no_beta.decoder.load_state_dict(torch.load(model_path_dict["CDE_no_beta"])['decoder_state_dict'])


times_to_eval = [2, 5, 10]

if __name__ == "__main__":
    print("---" * 10)
    print("Evaluating different models at times 2, 5, 10")
    logging.info("Evaluating different models at times 2, 5, 10")
    time_dict =  {"time2": {}, "time5": {}, "time10": {}}
    
    for time in times_to_eval:
        print("-----" * 10)
        logging.info("Evaluating at time {}".format(time))
        print("Evaluating at time {}".format(time))
        print("-----" * 10)
        
        logging.info("Evaluating LHM model")
        lhm_out = evaluate(old_vi, 
                                data_gen, 
                                50,
                                time,
                                mc_itr=50,
                                real=False,)
        print("-----" * 10)
        
        logging.info("Evaluating CDE model (constant coefficient)")
        cde_const_out = evaluate(vae_const, 
                                data_gen, 
                                50,
                                time,
                                mc_itr=50,
                                real=False,)
        print("-----" * 10)
        
        logging.info("Evaluating CDE model (trainable coefficient)")
        cde_trainable_out = evaluate(vae2, 
                                data_gen, 
                                50,
                                time,
                                mc_itr=50,
                                real=False,)
        print("-----" * 10)
        
        logging.info("Evaluating CDE model (vanilla)")
        cde_vanilla_out = evaluate(vae_vanilla, 
                                data_gen, 
                                50,
                                time,
                                mc_itr=50,
                                real=False,)
        print("-----" * 10)
        
        logging.info("Evaluating CDE model (no beta)")
        cde_no_beta_out = evaluate(vae_no_beta, 
                                data_gen, 
                                50,
                                time,
                                mc_itr=50,
                                real=False,)
        print("-----" * 10)
        
        # store in dict
        time_dict["time{}".format(time)]["LHM"] = lhm_out
        time_dict["time{}".format(time)]["CDE_const_0.5"] = cde_const_out
        time_dict["time{}".format(time)]["CDE_trainable"] = cde_trainable_out
        time_dict["time{}".format(time)]["CDE_vanilla"] = cde_vanilla_out
        time_dict["time{}".format(time)]["CDE_no_beta"] = cde_no_beta_out
        
    # save dict
    with open("eval_times.pkl", "wb") as f:
        pickle.dump(time_dict, f)
    