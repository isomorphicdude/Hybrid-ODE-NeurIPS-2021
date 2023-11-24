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

from SDE_helper import make_y_net, make_w_net, expertODE
from torchbnn._impl import diffeq_layers, utils

import warnings 
# print detailed warnings
warnings.filterwarnings("default")

import logging
logging.getLogger().setLevel(logging.ERROR)



class GaussianReparam:
    """Independent Gaussian variational posterior with re-parameterization trick."""
    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def log_density(mu, log_var, z):
        n = dist.normal.Normal(mu, torch.exp(0.5 * log_var))
        log_p = torch.sum(n.log_prob(z), dim=-1)
        return log_p
    
class ExponentialPrior:
    @staticmethod
    def log_density(z):
        n = dist.exponential.Exponential(rate=torch.tensor([100.0]).to(z))
        return torch.sum(n.log_prob(z), dim=-1)
    
    

# The same encoder structure
class NewEncoderLSTM(nn.Module, GaussianReparam):
    """Is used in the simulation setting. """
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True, device=None):
        # output dim is the dim of initial condition
        # input dim is observation and action

        super(NewEncoderLSTM, self).__init__()

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.model_name = "LSTMEncoder"

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim,
                            ).to(self.device)

        # The linear layer that maps from hidden state space to output space: predict mean
        self.lin = nn.Linear(hidden_dim, 
                             output_dim).to(self.device)

        self.log_var = nn.Linear(hidden_dim, 
                                 output_dim).to(self.device)

    def forward(self, x, a, mask):
        # y and t are the first k observations

        t_max = x.shape[0]

        x = x.squeeze()

        y_in = torch.cat([x, a], dim=-1) # concatenate observation and action
        mask_in = torch.cat([mask, torch.ones_like(a)], dim=-1)

        hidden = None

        for t in reversed(range(t_max)):
            obs = y_in[t : t + 1, ...] * mask_in[t : t + 1, ...]
            out, hidden = self.lstm(obs, hidden) 
            
        out_linear = self.lin(out)
        log_var = self.log_var(out)

        # B, D
        mu = out_linear[0, ...]
        log_var = log_var[0, ...]

        if self.normalize:
            # scale mu
            mu = torch.exp(mu) / 10
            # mask = torch.zeros_like(mu)
            # mask[:, 0] = 1
            # mu = mu * mask

            # scale var
            log_var = log_var - 5.0

        return mu, log_var
    

# Decoder should be a Bayes SDE mixed with the expert ODE
class SDENet(torchsde.SDEIto):
    """
    Initializes the Bayes SDE model.
    
    - input_size: (D, ) where D is the dimension of the latent vector
    - blocks: (1, ) for now
    - batch_size: batch size
    - weight_network_sizes: (2, 2) for now
    - sigma: diffusion coefficient
    - hidden_width: hidden width of the y net
    - obs_dim: observation dimension
    - latent_dim: latent dimension
    - action_dim: action dimension
    - t_max: maximum time
    - step_size: step size of time interval
    - device: device
    - condition_w: whether to condition on all expert ODEs
    """
    # chain rule holds df(X_t) = f'(X_t) dX_t for Stratonovich SDEs
    # limit of sum 0.5*(X_ti + X_{ti-1})(S_{ti} - S_{ti-1})
    
    def __init__(self,
                 input_size, # single latent vector shape (dim,)
                 y_net_hidden_width=2,
                #  batch_size=32,
                 weight_network_sizes=(64,),
                 control_network_width=64,
                 sigma=0.1,
                 obs_dim=20,
                 latent_dim=6,
                 action_dim=1,
                 t_max=14,
                 step_size=1,
                 dt=0.1,
                 device=None,
                 condition_w=True,
                 condition_w_only=False,
                 ablate=False):
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # SDE configurations
        super(SDENet, self).__init__(noise_type="diagonal")
        self.dt = dt
        self.input_size = input_size #(D, )
        self.aug_input_size = (input_size[0], 
                               *input_size[1:])
        # self.aug_zeros_size = (*input_size[1:],) # (D,)
        # self.register_buffer('aug_zeros', 
        #                      torch.zeros(size=(1, *self.aug_zeros_size)))
        
        # expert configurations
        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.step_size = step_size
        self.roche = True
        self.ablate = ablate
        print(f"Running ablation study: {self.ablate}")
        
        self.condition_w_only = condition_w_only
        print(f"Conditioning on w only: {self.condition_w_only}")
        if condition_w_only:
            assert condition_w, "Must condition on all expert ODEs"
        
        if self.roche:
            # include expert dim
            if latent_dim == 4:
                self.model_name = "ExpertDecoder"
            else:
                self.model_name = "HybridDecoder"
        else:
            self.model_name = "NeuralODEDecoder"
            
        # action configurations
        dc = sim_config.RochConfig()
        self.action_times = None
        self.dosage = None
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=DTYPE))

        # if self.ablate:
        #     self.model_name = self.model_name + "Ablate"
        #     print("Running ablation study")
        
        

        # times to output predictions
        # TODO: find way to transform to 0,1 interval
        self.prediction_times = torch.arange(0,
                                             t_max + step_size,
                                             step_size,
                                             device=self.device,
                                             dtype=DTYPE) / self.t_max
        
        
        # Create network evolving state.
        self.y_net, self.output_size = make_y_net(
            input_size=(input_size[0] - 4, ), # exclude expert dim
            hidden_width=y_net_hidden_width,
        )
        
        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()  # w0.
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)
        
        
        # assign initial params as nn.Parameter
        self.flat_initial_params = nn.Parameter(flat_initial_params, 
                                                requires_grad=True)
        self.params_size = flat_initial_params.numel()
        print(f"Number of parameters of y net: {self.params_size}")
        
        
        self.unravel_params = unravel_params
        
        # Weight network, incoporating the Expert ODE
        self.condition_w = condition_w
        print(f"Conditioning on expert ODEs: {self.condition_w}")
        
        if not self.condition_w:
            self.w_net = make_w_net(
                in_features=self.params_size,
                out_features=self.params_size,
                hidden_sizes=weight_network_sizes,
                # activation="tanh",
            )
        else:
            if not self.condition_w_only:
                print(f"Conditioning expert ODEs only {self.condition_w_only}")
                # self.w_net = make_w_net(
                #     in_features=self.params_size + 4 * batch_size,
                #     out_features=self.params_size,
                #     hidden_sizes=weight_network_sizes,
                #     activation="tanh",
                # )
                self.w_net = make_w_net(
                    in_features=self.params_size,
                    out_features=self.params_size,
                    hidden_sizes=weight_network_sizes,
                    # activation="tanh",
                )
                # Create control term.
                self.control_net = make_w_net(
                    in_features=4,
                    out_features=self.params_size,
                    hidden_sizes=(control_network_width,),
                    # activation="tanh",
                )
            else:
                # print("Conditioning on expert ODE only")
                raise NotImplementedError("Not implemented for conditional ODE")
                self.w_net = make_w_net(
                    in_features=self.params_size,
                    out_features=self.params_size,
                    hidden_sizes=weight_network_sizes,
                    # activation="tanh",
                )
            
            
        # Final decoding layer to observation space.
        if not self.ablate:
            self.projection = nn.Sequential(
                nn.Linear(self.input_size[0], 
                        obs_dim,
                        bias=True),
            )
        else:
            # drop the expert dim
            self.projection = nn.Sequential(
                nn.Linear(self.input_size[0] - 4, 
                        obs_dim,
                        bias=True),
            )

        # time-interval
        # TODO: change back to 0,1 if numerical issues
        self.register_buffer('ts', torch.tensor([0., 1.]))
        self.sigma = sigma
        self.nfe = 0
        
        # for Neural CDE
        # self.controlled_w = make_w_net(
        #             in_features=self.params_size,
        #             out_features=self.params_size * 4,
        #             hidden_sizes=weight_network_sizes,
        #             activation="tanh",)
        
        
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
        
    # def dose_at_time(self, t):
    #     """Computes the dosage at time t."""
    #     # print("not using modifed")
    #     return self.dosage * torch.sum(
    #         torch.exp(self.kel * \
    #             (self.action_times - t) * \
    #                 (t >= self.action_times)) * \
    #                     (t >= self.action_times), dim=-1
    #     )
    

    def f(self, t, Y: torch.Tensor):
        """
        Implements the augmented drift function.
        - y: (hidden_state, weight, logqp)
        """
        input_y = Y
        # print(f"input_y: {input_y.shape}")
        self.nfe += 1
        # those are still flattened
        y, w, _ = Y.split(split_size=(Y.numel() - self.params_size - 1,
                                      self.params_size, 
                                      1), 
                          dim=1)

        # evolve the hidden state (TODO: include expert ODE and actions)
        y = y.reshape(-1, *self.aug_input_size)
        
        # split y into first 4 expert and the rest
        y_expert, y_ml = y[:, :4], y[:, 4:]
        fy = self.y_net(t, 
                        y_ml, 
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
                              in_SDE=True).reshape(-1)

        hybrid_fy = torch.cat([fy_expert, 
                               fy], dim=-1)
        
        if len(hybrid_fy.shape) == 1:
            hybrid_fy = hybrid_fy[None]
        
        if not self.condition_w:
            nn = self.w_net(t, w)
        else:
            # include everything
            # hybrid_w = torch.cat((y_expert.reshape(1, -1), w), dim=-1)
            # nn = self.w_net(t, hybrid_w)
            
            # # take mean over batch of y_expert
            if not self.ablate:
                if not self.condition_w_only:
                    # does not enter computation graph
                    # viewed as external observation for control
                    y_expert_copy = y_expert.clone().detach()
                    control_term = self.control_net(t, y_expert_copy).mean(dim=0)
                    nn = self.w_net(t, w) + control_term
                else:
                    raise NotImplementedError("Not implemented for conditional ODE")
                    # y_expert_copy = y_expert.clone().detach()
                    # nn = self.control_net(y_expert_copy).mean(dim=0)  
                    # nn = nn[None]
            else:
                # use some vector of the same size in ablative study
                # control_term = self.control_net(torch.ones_like(y_expert)).mean(dim=0)
                # nn = self.w_net(t, w) + control_term
                
                # exclude expert dim
                nn = self.w_net(t, w)
            
        
        hybrid_fw = nn - w  # hardcoded OU prior on weights w
        
        fl = (nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)
        
        # print(f"The drift f has shape: {torch.cat((hybrid_fy, hybrid_fw, fl), dim=-1).shape}")
        assert input_y.shape == torch.cat((hybrid_fy, hybrid_fw, fl), dim=-1).shape, \
            f"Want: {input_y.shape} Got: {torch.cat((hybrid_fy, hybrid_fw, fl)).shape}. Check nblocks for dataset divisibility.\n"
        
        return torch.cat((hybrid_fy, hybrid_fw, fl), dim=-1)



    def g(self, t, Y: torch.Tensor):
        self.nfe += 1
        # state diffusion coefficient
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
        
        # print(f"The diffusion g has shape: {ret.shape}")
        return ret


    def make_initial_params(self):
        return self.y_net.make_initial_params()
    
    
    def forward(self, 
                y: torch.Tensor, 
                action,
                adjoint=False,
                adaptive=False,
                adjoint_adaptive=False, 
                method="euler", 
                rtol=1e-4,
                atol=1e-3):
        # Note: This works correctly, as long as we are requesting the nfe after each gradient update.
        # There are obviously cleaner ways to achieve this.
        self.nfe = 0    
        
        # set treatment actions
        self.set_action(action)
        
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        # if self.aug_zeros.numel() > 0:  # Add zero channels.
        #     aug_zeros = self.aug_zeros.expand(y.shape[0], *self.aug_zeros_size)
        #     y = torch.cat((y, aug_zeros), dim=1)
            
        aug_y = torch.cat((y.reshape(-1), 
                           self.flat_initial_params, 
                           torch.tensor([0.], device=y.device)))
        
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
                               adaptive=adaptive,
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
                               adaptive=adaptive,
                               rtol=rtol,
                               atol=atol)
            # print(f"nfe: {self.nfe}")
            # print(f"sde_output shape: {sde_output}")
        
        # Only return the hidden part of the state. (_ is the weight)
        y1 = sde_output[:, :, :y.numel()].flatten().reshape(len(self.prediction_times), 
                                                            *tuple(y.size())) # the last bit is removed
        # y1 shape (time, batch, dim)
        
        # Projection layer.
        if not self.ablate:
            measurements = self.projection(y1)
            
        else:
            # drop the expert dim
            y2 = y1[:, :, 4:]
            measurements = self.projection(y2)
        logqp = .5 * sde_output[-1, :, -1] # KL divergence
        
        
        return measurements, logqp
        

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None




class NewVariationalInference:
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
                 model_name=None):
        
        self.encoder = encoder
        self.decoder = decoder
        self.prior_log_pdf = prior_log_pdf
        self.mc_size = mc_size
        self.kl_coeff1 = kl_coeff1
        self.kl_coeff2 = kl_coeff2
        self.interval_length = interval_length
        self.elbo = elbo
        self.use_full_mc_kl = use_full_mc_kl
        self.use_full_mc_lik = use_full_mc_lik
        self.lik_mc_size = lik_mc_size
        
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

        self.mu = mu
        self.log_var = log_var

        # B, D
        if self.elbo:
            z = self.encoder.reparameterize(mu, log_var)
        else:
            z = mu
        self.z = z

        x_hat, kl_of_w = self.decoder(z, a)
        
        self.x_hat = x_hat
        # self.h_hat = h_hat

        # average over B (samples in mini batch)
        if not self.use_full_mc_lik:
            lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]
        else:
            lik = torch.mean(self.mc_lik(), dim=0)
            
        if not self.elbo:
            raise NotImplementedError("Not implemented yet")


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
    
    


class SDENet2(torchsde.SDEStratonovich):
    """
    Vanilla SDE weights for Neural ODE.
    """
    
    def __init__(self,
                 input_size, # single latent vector shape (dim,)
                 y_net_hidden_width=2,
                #  batch_size=32,
                 weight_network_sizes=(64,),
                 sigma=0.1,
                 obs_dim=20,
                 t_max=14,
                 step_size=1,
                 dt=0.1,
                 device=None,
                 ablate=False):
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # SDE configurations
        super(SDENet2, self).__init__(noise_type="diagonal")
        self.dt = dt
        self.input_size = input_size #(D, )
        self.aug_input_size = (input_size[0], 
                               *input_size[1:])
        self.t_max = t_max
        self.model_name = "PureODE"

        
        self.ablate = ablate
        print(f"Running ablation study: {self.ablate}")
        
        
        # times to output predictions
        # TODO: find way to transform to 0,1 interval
        self.prediction_times = torch.arange(0,
                                             t_max + step_size,
                                             step_size,
                                             device=self.device,
                                             dtype=DTYPE) / self.t_max
        
        
        # Create network evolving state.
        self.y_net, self.output_size = make_y_net(
            input_size=(input_size[0], ),
            hidden_width=y_net_hidden_width,
        )
        
        # Create network evolving weights.
        initial_params = self.y_net.make_initial_params()  # w0.
        flat_initial_params, unravel_params = utils.ravel_pytree(initial_params)
        
        
        # assign initial params as nn.Parameter
        self.flat_initial_params = nn.Parameter(flat_initial_params, 
                                                requires_grad=True)
        self.params_size = flat_initial_params.numel()
        print(f"Number of parameters of y net: {self.params_size}")
        
        
        self.unravel_params = unravel_params
        
        
        self.w_net = make_w_net(
            in_features=self.params_size,
            out_features=self.params_size,
            hidden_sizes=weight_network_sizes,
            # activation="tanh",
        )
        _initial_w_params = self.w_net.make_initial_params()
        _flat_initial_w_params, _unravel_w_params = utils.ravel_pytree(_initial_w_params)
        print(f"Number of parameters of w-net: {_flat_initial_w_params.numel()}")
            
        # Final decoding layer to observation space.
        self.projection = nn.Sequential(
            nn.Linear(self.input_size[0], 
                    obs_dim,
                    bias=True),
        )

        # time-interval
        # TODO: change back to 0,1 if numerical issues
        self.register_buffer('ts', torch.tensor([0., 1.]))
        self.sigma = sigma
        self.nfe = 0
    

    def f(self, t, Y: torch.Tensor):
        """
        Implements the augmented drift function.
        - y: (hidden_state, weight, logqp)
        """
        input_y = Y
        
        self.nfe += 1
        
        # those are still flattened
        y, w, _ = Y.split(split_size=(Y.numel() - self.params_size - 1,
                                      self.params_size, 
                                      1), 
                          dim=1)

        # evolve the hidden state
        y = y.reshape(-1, *self.aug_input_size)
        
        fy = self.y_net(t, 
                        y, 
                        self.unravel_params(w.reshape(-1))).reshape(-1)

        if len(fy.shape) == 1:
            fy = fy[None]
        
        nn = self.w_net(t, w)
        
        fw = nn - w  # hardcoded OU prior on weights w
        
        fl = (nn ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)
        
        return torch.cat((fy, fw, fl), dim=-1)



    def g(self, t, Y: torch.Tensor):
        self.nfe += 1
        
        # state diffusion coefficient
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
        
        # print(f"The diffusion g has shape: {ret.shape}")
        return ret


    def make_initial_params(self):
        return self.y_net.make_initial_params()
    
    
    def forward(self, 
                y: torch.Tensor, 
                action,
                adjoint=False,
                adaptive=False,
                adjoint_adaptive=False, 
                method="midpoint", 
                rtol=1e-4,
                atol=1e-3):
        # Note: This works correctly, as long as we are requesting the nfe after each gradient update.
        # There are obviously cleaner ways to achieve this.
        self.nfe = 0    
        
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
            
        aug_y = torch.cat((y.reshape(-1), 
                           self.flat_initial_params, 
                           torch.tensor([0.], device=y.device)))
        
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
                               adaptive=adaptive,
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
                               adaptive=adaptive,
                               rtol=rtol,
                               atol=atol)
        
        # Only return the hidden part of the state. (_ is the weight)
        #NOTE: This has extremely high memory usage.
        y1 = sde_output[:, :, :y.numel()].flatten().reshape(len(self.prediction_times), 
                                                            *tuple(y.size())) # the last bit is removed

        # Projection layer.
        measurements = self.projection(y1)
            
        logqp = .5 * sde_output[-1, :, -1] # KL divergence
        
        return measurements, logqp
        

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None
        
        
        
class SDENet3(torchsde.SDEStratonovich):
    """
    Initializes the Bayes SDE model with expert ODE.
    
    The dt from expert controls the SDE drfit term in the 
    variational posterior. Similar to the hybrid model.
    But include uncertainty, as the influence of expert ODE
    can be controlled.
    Need to investigate under model misspecification, this
    should be more robust, as expert is included "optionally", 
    in the sense that it only impacts the weights of the ODE.
    """

    def __init__(self,
                 input_size=(6,), # single latent vector shape (dim,)
                 y_net_hidden_width=2,
                 weight_network_sizes=(64,),
                 latent_dim=6,
                 action_dim=1,
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
        super(SDENet3, self).__init__(noise_type="diagonal")
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
        
        # include expert dim as augmentation        
        self.input_size = input_size #(D, )
        
        print(f"Running ablation study: {self.ablate}")
        
        if self.roche:
            # include expert dim
            if latent_dim == 4:
                self.model_name = "ExpertDecoder"
            else:
                self.model_name = "HybridDecoder"
        else:
            self.model_name = "NeuralODEDecoder"
            
        # action configurations
        dc = sim_config.RochConfig()
        self.action_times = None
        self.dosage = None
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=DTYPE))
        
        
        # times to output predictions
        # TODO: find way to transform to 0,1 interval
        self.prediction_times = torch.arange(0,
                                             t_max + step_size,
                                             step_size,
                                             device=self.device,
                                             dtype=DTYPE) / self.t_max
        
        
        # Create network evolving state.
        self.y_net, self.output_size = make_y_net(
            input_size=(input_size[0] - self.expert_dim, ),
            hidden_width=y_net_hidden_width,
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
        
        # Weights governed by a Neural CDE
        self.controlled_w = make_w_net(
                    in_features=self.params_size,
                    out_features=self.params_size * self.expert_dim,
                    hidden_sizes=weight_network_sizes,
                    activation="tanh",)
        _initial_w_params = self.controlled_w.make_initial_params()
        _flat_initial_w_params, _unravel_w_params = utils.ravel_pytree(_initial_w_params)
        print(f"Number of parameters of w-net: {_flat_initial_w_params.numel()}")
        
               
        # Final decoding layer to observation space.
        self.projection = nn.Sequential(
            nn.Linear(self.input_size[0] - self.expert_dim, 
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
        - y: (hidden_state, weight, logqp)
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

        hybrid_fy = torch.cat([fy, 
                               fy_expert.view(-1)], dim=-1)
        
        if len(hybrid_fy.shape) == 1:
            hybrid_fy = hybrid_fy[None]
        

        nn = torch.matmul(self.controlled_w(t, w).view(self.params_size, self.expert_dim),
                            fy_expert.mean(dim=0))
        
        hybrid_fw = nn - w  # hardcoded OU prior on weights w
        
        
        fl = (nn[None] ** 2).sum(dim=1, keepdim=True) / (self.sigma ** 2)
        

        return torch.cat((hybrid_fy, hybrid_fw, fl), dim=-1)



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
                Y: torch.Tensor, 
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
        
        y, y_expert = Y.split(split_size=(self.input_size[0] - self.expert_dim,
                                            self.expert_dim),
                              dim=1)
        
        aug_y = torch.cat((y.reshape(-1), 
                           y_expert.reshape(-1),
                           self.flat_initial_params, 
                           torch.tensor([0.], device=Y.device)))
        
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
    