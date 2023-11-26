import os

# import torchcde
import torch
import numpy as np
import torch.distributions as dist
import torch.nn as nn

import torchsde
from torchdiffeq import odeint as dto

# from TorchDiffEqPack.odesolver import ode_solver
import sim_config
from global_config import DTYPE, get_device

from SDE_helper import make_y_net_CDE, make_w_net, expertODE
from torchbnn._impl import diffeq_layers, utils

import warnings 
# print detailed warnings
warnings.filterwarnings("default")

import logging
logging.getLogger().setLevel(logging.ERROR)

class HybridVariationalInference:
    epsilon = torch.finfo(DTYPE).eps
    def __init__(self, 
                 encoder,
                 decoder,
                 elbo=True,
                 prior_log_pdf=None,
                 use_full_mc_kl=False,
                 use_full_mc_lik=False,
                 mc_size=100,
                 lik_mc_size=10,
                 kl_coeff1=0.2,
                 kl_coeff2=0.2,
                 interval_length=1,
                 expert_encoder=None,
                #  augment_dim=4,
                 model_name=None):
        # extra expert encoder
        self.encoder = encoder
        self.decoder = decoder
        self.expert_encoder = expert_encoder
        self.prior_log_pdf = prior_log_pdf
        self.mc_size = mc_size
        self.kl_coeff1 = kl_coeff1
        self.kl_coeff2 = kl_coeff2
        self.interval_length = interval_length
        self.elbo = elbo
        self.use_full_mc_kl = use_full_mc_kl
        self.use_full_mc_lik = use_full_mc_lik
        self.lik_mc_size = lik_mc_size
        # self.augment_dim = augment_dim
        
        if model_name is None:
            self.model_name = "VI_{}_{}.pkl".format(encoder.model_name, decoder.model_name)
        else:
            self.model_name = model_name


    def save(self, path, itr, best_loss):
        path = path + self.model_name
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "itr": itr,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "best_loss": best_loss,
            },
            path,
        )

    def loss(self, data):
        """
        Compute the loss function.
        
        Should be comprised of:
        1. Likelihood loss
        2. KL divergence loss for VAE
        3. KL divergence loss for Bayes SDE
        """  
        
        x = data["measurements"]
        a = data["actions"]
        mask = data["masks"]

        self.x = x
        self.a = a
        self.mask = mask

        # q
        mu, log_var = self.encoder(x, a, mask)
        
        # q of expert
        _mu_expert, _log_var_expert = self.expert_encoder(x, a, mask)

        self.mu = mu
        self.log_var = log_var
        
        self._mu_expert = _mu_expert
        self._log_var_expert = _log_var_expert

        # B, D
        if self.elbo:
            z = self.encoder.reparameterize(mu, log_var)
            _z_expert = self.expert_encoder.reparameterize(_mu_expert, _log_var_expert)
        else:
            raise NotImplementedError("Not implemented yet")
            z = mu
        self.z = z
        
        all_z = torch.cat((z, _z_expert), dim=-1)
        # print(f"all_z shape: {all_z.shape}")
        
        # pass through decoder
        x_hat, kl_of_w = self.decoder(all_z, a)
        
        self.x_hat = x_hat
        # self.h_hat = h_hat

        # average over B (samples in mini batch)
        if not self.use_full_mc_lik:
            lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]
        else:
            lik = torch.mean(self.mc_lik(), dim=0)
            
        if not self.elbo:
            raise NotImplementedError("Not implemented for no ELBO")


        if self.prior_log_pdf is None:
            # if no prior, use standard normal
            # analytic KL
            raise NotImplementedError("Not implemented yet for analytic KL")
            # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        else:
            # monte carlo KL over batch
            kld_loss_vae = torch.mean(self.mc_kl_vae(mu, log_var), dim=0)
            
            if not self.use_full_mc_kl:
                kld_loss_sde = kl_of_w
            else:
                kld_loss_sde = torch.mean(self.mc_kl_sde(), dim=0)
        # print(f"Likelihood loss: {lik}")
        # print(f"KL loss for VAE: {kld_loss_vae}")
        # print(f"KL loss for SDE: {kld_loss_sde}")
        loss = lik + kld_loss_vae * self.kl_coeff1 + kld_loss_sde * self.kl_coeff2 / self.interval_length
        return loss

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
    

    def mc_kl_vae(self, mu, log_var):
        mc_samples = list()

        for i in range(self.mc_size):
            # sample from q(z)
            z = self.encoder.reparameterize(mu, log_var)
            
            z[z <= 0.0] = self.epsilon
            # log p(z)
            log_p = self.prior_log_pdf(z)
            # log q(z)
            log_q = self.encoder.log_density(mu, log_var, z)
            mc_samples.append(log_q - log_p)

        mc_tensor = torch.stack(mc_samples, dim=-1)
        mc_mean = torch.mean(mc_tensor, dim=-1)
        return mc_mean
    
    def mc_kl_sde(self):
        mc_samples = list()
        input_z = self.z
        for i in range(self.mc_size):
            # sample from q(z)
            
            input_z[input_z <= 0.0] = self.epsilon
            
            # pass through decoder
            _, kl_of_w = self.decoder(input_z, self.a)
            mc_samples.append(kl_of_w)
            
        mc_tensor = torch.stack(mc_samples, dim=-1)
        mc_mean = torch.mean(mc_tensor, dim=-1)
        return mc_mean
    
    def mc_lik(self):
        mc_samples = list()
        # against the ground truth
        input_x = self.x
        
        input_a = self.a
        input_mask = self.mask
        
        # two loops
        for i in range(self.lik_mc_size):
            # pass through encoder
            mu, log_var = self.encoder(input_x, input_a, input_mask)
            # sample from q(z)
            z = self.encoder.reparameterize(mu, log_var)
            # pass through decoder
            for j in range(self.lik_mc_size):
                input_x_hat, _ = self.decoder(z, input_a)
                # compute likelihood
                lik = torch.sum((input_x - input_x_hat) ** 2 * input_mask) / input_x.shape[1]
                mc_samples.append(lik)
                
        mc_tensor = torch.stack(mc_samples, dim=-1)
        mc_mean = torch.mean(mc_tensor, dim=-1)
        return mc_mean



class SDENet4(torchsde.SDEStratonovich):
    """
    Initializes the Bayes SDE model with expert ODE.
    
    In contrast to SDENet3, the expert ODE is not part of the model
    it is a separate dynamical system, which is used to control the
    value of the ODE directly, instead of the weights.
    
    In addition, we introduce extra augmentation to the state space
    initializing 
    """

    def __init__(self,
                 input_size=(6,), # single latent vector shape (dim,)
                 y_net_hidden_size=2,
                 weight_network_sizes=(64,),
                 latent_dim=6,
                 action_dim=1,
                 augmented_dim=4,
                 sigma=0.1,
                 obs_dim=20,
                 t_max=14,
                 step_size=1,
                 dt=0.1,
                 batch_size=32,
                 adaptive=False,
                 device=None,
                 scaling_factor = 1,
                 ablate=False):
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # SDE configurations
        super(SDENet4, self).__init__(noise_type="diagonal")
        self.dt = dt
        self.batch_size = batch_size
        self.sigma = sigma
        self.adaptive = adaptive
        
        # expert configurations
        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.step_size = step_size
        self.roche = True
        self.ablate = ablate
        self.expert_dim = 4
        self.augmented_dim = augmented_dim
        self.scaling_factor = scaling_factor
        
        # action configurations
        dc = sim_config.RochConfig()
        self.action_times = None
        self.dosage = None
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=DTYPE))
        
        # include expert dim as augmentation        
        self.input_size = input_size #(D, )
        
        print(f"Running ablation study: {self.ablate}")
        
        self.model_name = "SDENet4"
            
        
        
        # times to output predictions
        # TODO: find way to transform to 0,1 interval
        self.prediction_times = torch.arange(0,
                                             t_max + step_size,
                                             step_size,
                                             device=self.device,
                                             dtype=DTYPE) / self.scaling_factor
        
        
        # Create network evolving state.
        self.y_net, self.output_size = make_y_net_CDE(
            input_size=(input_size[0], ),
            control_vec_dim=self.expert_dim,
            hidden_size=y_net_hidden_size,
        )
        
        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)
        
        
        # assign initial params as nn.Parameter
        self.flat_initial_params = nn.Parameter(flat_initial_params, 
                                                requires_grad=True)
        self.params_size = flat_initial_params.numel()
        print(f"Number of parameters of y-net: {self.params_size}")
        
        
        self.unravel_params = unravel_params
        
        # Weights governed by a normal SDE
        self.w_net = make_w_net(
                    in_features=self.params_size,
                    out_features=self.params_size,
                    hidden_sizes=weight_network_sizes,
                    activation="tanh",)
        _initial_w_params = self.w_net.make_initial_params()
        _flat_initial_w_params, _unravel_w_params = utils.ravel_pytree(_initial_w_params)
        print(f"Number of parameters of w-net: {_flat_initial_w_params.numel()}")
        
               
        # Final decoding layer to observation space.
        self.projection = nn.Sequential(
            nn.Linear(self.input_size[0], 
                      obs_dim,
                      bias=True)
            )

        # time-interval
        # TODO: change back to 0,1 if numerical issues
        self.register_buffer('ts', torch.tensor([0., 1.]))
        self.sigma = sigma
        self.nfe = 0
    
    def set_action(self, action):
        """Set the treatment action times."""
        # T, B, D (Time, Batch, Dose)
        self.dosage = torch.max(action[..., 0], dim=0)[0] # Batch

        time_list = []
        for i in range(action.shape[1]):
            # indices of non-zero dosage
            time = torch.where(action[..., 0][:, i] != 0)[0]
            time = time * self.step_size # step is 1
            time_list.append(time)

        # B, N_DOSE
        self.action_times = torch.stack(time_list, dim=0)
        
    def dose_at_time(self, t):
        """Computes the dosage at time t."""
        return self.dosage * torch.sum(
            torch.exp(self.kel * \
                (self.action_times - t * self.scaling_factor) * \
                    (self.scaling_factor * t >= self.action_times)) * \
                        (self.scaling_factor * t >= self.action_times), dim=-1
        )
    
        

    def f(self, t, Y: torch.Tensor):
        """
        Implements the augmented drift function.
        - Y: (hidden_state, extra_input, weight, logqp)
        here hidden_state is latent plus redundant(possibly)
        """
        input_y = Y
        # print(f"input_y: {input_y.shape}")
        self.nfe += 1
        
        # those are still flattened
        y, y_expert, w, _ = Y.split(
            split_size=(
                Y.numel() - self.params_size - 1 - self.batch_size * self.expert_dim,
                self.batch_size * self.expert_dim,
                self.params_size, 
                1), 
            dim=1)

        # evolve the hidden state (TODO: include expert ODE and actions)
        y = y.reshape(self.batch_size, -1)
        y_expert = y_expert.reshape(self.batch_size, -1)
        
        # dim 4 flattened fy
        fy = self.y_net(t, 
                        y, 
                        self.unravel_params(w.reshape(-1))).reshape(-1)
        

        Disease = y_expert[:, 0]
        ImmuneReact = y_expert[:, 1]
        Immunity = y_expert[:, 2]
        Dose2 = y_expert[:, 3]
        
        # TODO: transform t to 0,1 interval
        fy_expert = expertODE(t,
                              Disease,
                              ImmuneReact,
                              Immunity,
                              Dose2,
                              self.dose_at_time,
                              device=self.device,
                              in_SDE=True,
                              scaling_factor=self.scaling_factor)
        
        # same size as fy        
        ncde_fy = torch.einsum("ijk,ik->ij", 
                               fy.reshape(-1, y.shape[1], self.expert_dim),
                               y_expert).reshape(-1)
        
        
        hybrid_fy = torch.cat((ncde_fy,
                               fy_expert.reshape(-1)),
                              dim=-1)
        
        nn = self.w_net(t, w)
        
        fw = nn - w  # hardcoded OU prior on weights w
        
        
        fl = (nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)
        
        return torch.cat((hybrid_fy[None], fw, fl), dim=-1)



    def g(self, t, Y: torch.Tensor):
        
        self.nfe += 1
        
        # state diffusion coefficient, include expert dim
        gy = torch.zeros(size=(Y.numel() - self.params_size - 1,), 
                         device=Y.device)
        
        # weight diffusion coefficient
        gw = torch.full(size=(self.params_size,), 
                        fill_value=self.sigma, 
                        device=Y.device)
        
        
        gl = torch.tensor([0.], device=Y.device)
        
        ret = torch.cat((gy, gw, gl), dim=-1)
        
        if len(ret.shape) == 1:
            ret = ret[None]
        
        return ret


    def make_initial_params(self):
        return self.y_net.make_initial_params()
    
    
    def forward(self, 
                Y: torch.Tensor, # initial state
                action,
                adjoint=False,
                adjoint_adaptive=False, 
                method="midpoint"):
        self.nfe = 0    
        
        # set treatment actions
        self.set_action(action)
        
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        # print(f"Y shape: {Y.shape}")
        
        y, y_expert = Y.split(split_size=(self.input_size[0],
                                            self.expert_dim),
                              dim=1)
        # print(f"y shape: {y.shape}")
        # print(f"y_expert shape: {y_expert.shape}")
        
        # set the aug dim to zeros
        y_l, y_r = y.split(split_size=(y.shape[1] - self.augmented_dim,
                                       self.augmented_dim),
                           dim=1)
        y_r = torch.zeros_like(y_r)
        
        y = torch.cat((y_l, y_r), dim=1)
        # print(f"y shape after concat: {y.shape}")
        
        aug_y = torch.cat((y.reshape(-1), 
                           y_expert.reshape(-1),
                           self.flat_initial_params, 
                           torch.tensor([0.], device=Y.device)))
        
        # print(f"aug_y shape: {aug_y.shape}")
        
        aug_y = aug_y[None]
        assert len(aug_y.shape) >= 2, "Batch dim for Diffusion must exist!"
        
        bm = torchsde.BrownianInterval(
            t0=self.prediction_times[0], 
            t1=self.prediction_times[-1], 
            size=aug_y.shape,
            dtype=aug_y.dtype,
            device=aug_y.device,
            cache_size=45 if adjoint else 30  # If not adjoint, don't really need to cache.
        )
 
        
        
        sde_output = sdeint(self, 
                            aug_y,
                            self.prediction_times,
                            bm=bm,
                            method=method,
                            dt=self.dt,
                            adaptive=self.adaptive)
        
        # Only return the hidden part of the state. (_ is the weight)
        y1 = sde_output[:, :, :y.numel()].flatten().reshape(len(self.prediction_times), 
                                                            *tuple(y.size())) # the last bit is removed
        # y1 shape (time, batch, dim)
        
        # Projection layer.
        measurements = self.projection(y1)

        logqp = .5 * sde_output[-1, :, -1] # KL divergence
        
        return measurements, logqp
        

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None


class SDENet5(torchsde.SDEStratonovich):
    """
    Initializes the Bayes SDE model with expert ODE.
    
    Same as SDENet4, but includes the expert ODE as part of the model.
    
    In addition, we introduce extra augmentation to the state space
    initializing 
    """

    def __init__(self,
                 input_size=(6,), # single latent vector shape (dim,)
                 y_net_hidden_size=2,
                 weight_network_sizes=(64,),
                 latent_dim=6,
                 action_dim=1,
                 augmented_dim=4,
                 sigma=0.1,
                 obs_dim=20,
                 t_max=14,
                 step_size=1,
                 dt=0.1,
                 batch_size=32,
                 adaptive=False,
                 device=None,
                 ablate=False):
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # SDE configurations
        super(SDENet5, self).__init__(noise_type="diagonal")
        self.dt = dt
        self.batch_size = batch_size
        self.sigma = sigma
        self.adaptive = adaptive
        
        # expert configurations
        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.step_size = step_size
        self.roche = True
        self.ablate = ablate
        self.expert_dim = 4
        self.augmented_dim = augmented_dim
        
        # action configurations
        dc = sim_config.RochConfig()
        self.action_times = None
        self.dosage = None
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=DTYPE))
        
        # include expert dim as augmentation        
        self.input_size = input_size #(D, )
        
        print(f"Running ablation study: {self.ablate}")
        
        self.model_name = "SDENet5"
            
        
        
        # times to output predictions
        # TODO: find way to transform to 0,1 interval
        self.prediction_times = torch.arange(0,
                                             t_max + step_size,
                                             step_size,
                                             device=self.device,
                                             dtype=DTYPE) / self.t_max
        
        
        # Create network evolving state.
        self.y_net, self.output_size = make_y_net_CDE(
            input_size=(input_size[0], ),
            control_vec_dim=self.expert_dim,
            hidden_size=y_net_hidden_size,
        )
        
        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)
        
        
        # assign initial params as nn.Parameter
        self.flat_initial_params = nn.Parameter(flat_initial_params, 
                                                requires_grad=True)
        self.params_size = flat_initial_params.numel()
        print(f"Number of parameters of y-net: {self.params_size}")
        
        
        self.unravel_params = unravel_params
        
        # Weights governed by a normal SDE
        self.w_net = make_w_net(
                    in_features=self.params_size,
                    out_features=self.params_size,
                    hidden_sizes=weight_network_sizes,
                    activation="tanh",)
        
        _initial_w_params = self.w_net.make_initial_params()
        _flat_initial_w_params, _unravel_w_params = utils.ravel_pytree(_initial_w_params)
        print(f"Number of parameters of w-net: {_flat_initial_w_params.numel()}")
        
               
        # Final decoding layer to observation space.
        self.projection = nn.Sequential(
            nn.Linear(self.input_size[0], 
                      obs_dim,
                      bias=True)
            )

        # time-interval
        # TODO: change back to 0,1 if numerical issues
        self.register_buffer('ts', torch.tensor([0., 1.]))
        self.sigma = sigma
        self.nfe = 0
    
    def set_action(self, action):
        """Set the treatment action times."""
        # T, B, D (Time, Batch, Dose)
        self.dosage = torch.max(action[..., 0], dim=0)[0] # Batch

        time_list = []
        for i in range(action.shape[1]):
            # indices of non-zero dosage
            time = torch.where(action[..., 0][:, i] != 0)[0]
            time = time * self.step_size # step is 1
            time_list.append(time)

        # B, N_DOSE
        self.action_times = torch.stack(time_list, dim=0)
        
    def dose_at_time(self, t):
        """Computes the dosage at time t."""
        return self.dosage * torch.sum(
            torch.exp(self.kel * \
                (self.action_times - t * self.t_max) * \
                    (self.t_max * t >= self.action_times)) * \
                        (self.t_max * t >= self.action_times), dim=-1
        )
    
        

    def f(self, t, Y: torch.Tensor):
        """
        Implements the augmented drift function.
        - Y: (hidden_state, extra_input, weight, logqp)
        here hidden_state is latent plus redundant(possibly)
        """
        input_y = Y
        # print(f"input_y: {input_y.shape}")
        self.nfe += 1
        
        # those are still flattened
        y, y_expert, w, _ = Y.split(
            split_size=(
                Y.numel() - self.params_size - 1 - self.batch_size * self.expert_dim,
                self.batch_size * self.expert_dim,
                self.params_size, 
                1), 
            dim=1)

        # evolve the hidden state (TODO: include expert ODE and actions)
        y = y.reshape(self.batch_size, -1)
        y_expert = y_expert.reshape(self.batch_size, -1)
        
        # dim 4 flattened fy
        fy = self.y_net(t, 
                        y, 
                        self.unravel_params(w.reshape(-1))).reshape(-1)
        

        Disease = y_expert[:, 0]
        ImmuneReact = y_expert[:, 1]
        Immunity = y_expert[:, 2]
        Dose2 = y_expert[:, 3]
        
        # TODO: transform t to 0,1 interval
        fy_expert = expertODE(t,
                              Disease,
                              ImmuneReact,
                              Immunity,
                              Dose2,
                              self.dose_at_time,
                              device=self.device,
                              in_SDE=True)
        
        # same size as fy        
        ncde_fy = torch.einsum("ijk,ik->ij", 
                               fy.reshape(-1, y.shape[1], self.expert_dim),
                               y_expert).reshape(-1)
        
        
        hybrid_fy = torch.cat((ncde_fy,
                               fy_expert.reshape(-1)),
                              dim=-1)
        
        nn = self.w_net(t, w)
        
        fw = nn - w  # hardcoded OU prior on weights w
        
        
        fl = (nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)
        
        return torch.cat((hybrid_fy[None], fw, fl), dim=-1)



    def g(self, t, Y: torch.Tensor):
        
        self.nfe += 1
        
        # state diffusion coefficient, include expert dim
        gy = torch.zeros(size=(Y.numel() - self.params_size - 1,), 
                         device=Y.device)
        
        # weight diffusion coefficient
        gw = torch.full(size=(self.params_size,), 
                        fill_value=self.sigma, 
                        device=Y.device)
        
        
        gl = torch.tensor([0.], device=Y.device)
        
        ret = torch.cat((gy, gw, gl), dim=-1)
        
        if len(ret.shape) == 1:
            ret = ret[None]
        
        return ret


    def make_initial_params(self):
        return self.y_net.make_initial_params()
    
    
    def forward(self, 
                Y: torch.Tensor, # initial state
                action,
                adjoint=False,
                adjoint_adaptive=False, 
                method="midpoint", 
                rtol=1e-4,
                atol=1e-3):
        self.nfe = 0    
        
        # set treatment actions
        self.set_action(action)
        
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        # print(f"Y shape: {Y.shape}")
        
        y, y_expert = Y.split(split_size=(self.input_size[0],
                                            self.expert_dim),
                              dim=1)
        # print(f"y shape: {y.shape}")
        # print(f"y_expert shape: {y_expert.shape}")
        
        # set the aug dim to zeros
        y_l, y_r = y.split(split_size=(y.shape[1] - self.augmented_dim,
                                       self.augmented_dim),
                           dim=1)
        y_r = torch.zeros_like(y_r)
        
        y = torch.cat((y_l, y_r), dim=1)
        # print(f"y shape after concat: {y.shape}")
        
        aug_y = torch.cat((y.reshape(-1), 
                           y_expert.reshape(-1),
                           self.flat_initial_params, 
                           torch.tensor([0.], device=Y.device)))
        
        # print(f"aug_y shape: {aug_y.shape}")
        
        aug_y = aug_y[None]
        assert len(aug_y.shape) >= 2, "Batch dim for Diffusion must exist!"
        
        bm = torchsde.BrownianInterval(
            t0=self.prediction_times[0], 
            t1=self.prediction_times[-1], 
            size=aug_y.shape,
            dtype=aug_y.dtype,
            device=aug_y.device,
            cache_size=45 if adjoint else 30  # If not adjoint, don't really need to cache.
        )
 
        
        if adjoint_adaptive:
            sde_output = sdeint(self, 
                               aug_y,
                               self.prediction_times,
                               bm=bm,
                               method=method,
                               dt=self.dt,
                               adaptive=self.adaptive,
                               adjoint_adaptive=adjoint_adaptive,
                               rtol=rtol,
                               atol=atol)
        else:
            sde_output = sdeint(self, 
                               aug_y,
                               self.prediction_times,
                               bm=bm,
                               method=method,
                               dt=self.dt,
                               adaptive=self.adaptive,
                               rtol=rtol,
                               atol=atol)
        
        # Only return the hidden part of the state. (_ is the weight)
        y1 = sde_output[:, :, :y.numel()].flatten().reshape(len(self.prediction_times), 
                                                            *tuple(y.size())) # the last bit is removed
        # y1 shape (time, batch, dim)
        
        # Projection layer.
        measurements = self.projection(y1)

        logqp = .5 * sde_output[-1, :, -1] # KL divergence
        
        return measurements, logqp
        

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None


