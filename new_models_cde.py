"""Implements a special form of controlled differential equation (CDE) model."""
import os

# import torchcde
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from torchdiffeq import odeint as dto


import sim_config
from global_config import DTYPE, get_device


class NewRocheCDE1(nn.Module):
    """
    Controlled Differential Equation for Roche model.
    int alpha * [I 0] + [1-alpha, 1] * ml_net(y) dXe
     where I is 4 x 4 identity matrix
    so the first 4 dimensions are the expert model weighted by alpha
    the last 2 are the ml_dim
    """
    def __init__(self, 
                 latent_dim, 
                 action_dim, 
                 t_max, 
                 step_size, 
                 ablate=False, 
                 include_expert=False,
                 device=None, 
                 extra_dim=0,
                 use_beta = False,
                 dtype=DTYPE):
        super().__init__()

        assert action_dim == 1

        self.action_dim = action_dim
        self.latent_dim = int(latent_dim)
        self.extra_dim = extra_dim
        
        # fixed expert dim
        self.expert_dim = int(4)
        
        # the rest is governed by neural ODE
        self.ml_dim = self.latent_dim - self.expert_dim
        
        self.include_expert = include_expert
        if self.include_expert:
            print("Including expert model")
            self.ml_dim = self.latent_dim - 2 * self.expert_dim
        
        
        print(f"latent_dim: {self.latent_dim}")
        print(f"expert_dim: {self.expert_dim}")
        print(f"ml_dim: {self.ml_dim}")
        
        # use hybrid when expanded
        self.expanded = True if self.ml_dim > 0 else False
        self.ablate = ablate

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t_max = t_max
        self.step_size = step_size

        # parameters for the expert ODE
        dc = sim_config.RochConfig()
        self.HillCure = nn.Parameter(torch.tensor(dc.HillCure, device=self.device, dtype=dtype))
        self.HillPatho = nn.Parameter(torch.tensor(dc.HillPatho, device=self.device, dtype=dtype))
        self.ec50_patho = nn.Parameter(torch.tensor(dc.ec50_patho, device=self.device, dtype=dtype))
        self.emax_patho = nn.Parameter(torch.tensor(dc.emax_patho, device=self.device, dtype=dtype))
        self.k_dexa = nn.Parameter(torch.tensor(dc.k_dexa, device=self.device, dtype=dtype))
        self.k_discure_immunereact = nn.Parameter(
            torch.tensor(dc.k_discure_immunereact, device=self.device, dtype=dtype)
        )
        self.k_discure_immunity = nn.Parameter(torch.tensor(dc.k_discure_immunity, device=self.device, dtype=dtype))
        self.k_disprog = nn.Parameter(torch.tensor(dc.k_disprog, device=self.device, dtype=dtype))
        self.k_immune_disease = nn.Parameter(torch.tensor(dc.k_immune_disease, device=self.device, dtype=dtype))
        self.k_immune_feedback = nn.Parameter(torch.tensor(dc.k_immune_feedback, device=self.device, dtype=dtype))
        self.k_immune_off = nn.Parameter(torch.tensor(dc.k_immune_off, device=self.device, dtype=dtype))
        self.k_immunity = nn.Parameter(torch.tensor(dc.k_immunity, device=self.device, dtype=dtype))
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=dtype))
        
        if self.ablate:
            # ablation uses mis-specified parameters
            self.theta_1 = torch.tensor(1, device=self.device, dtype=dtype)
            self.theta_2 = torch.tensor(2, device=self.device, dtype=dtype)

        if not self.include_expert:
            self.ml_net = nn.Sequential(nn.Linear(self.latent_dim, 
                                                    self.expert_dim * self.latent_dim), 
                                        nn.Tanh())
        else:
            self.ml_net = nn.Sequential(nn.Linear(self.latent_dim-self.expert_dim,
                                                  (self.latent_dim-self.expert_dim) * self.expert_dim),
                                        nn.Tanh())

        
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.device, dtype=dtype),
                                    requires_grad=True)
        # self.alpha = nn.Parameter(torch.rand(1, dtype=dtype, device=self.device))
        self.use_beta = use_beta
        if self.use_beta:
            # use another parameter to weight the expert
            self.beta = nn.Parameter(torch.rand(1, dtype=dtype, device=self.device))
            
        
        self.expert_mat = nn.Parameter(torch.cat([torch.eye(self.expert_dim,),
                                        torch.zeros(self.expert_dim,
                                                    self.ml_dim,)],
                                        dim=0),
                                            requires_grad=True)
            
            
        # parameters for the action
        self.times = None
        self.dosage = None

    def set_action(self, action):
        self.dosage = torch.max(action[..., 0], dim=0)[0] 

        time_list = []
        for i in range(action.shape[1]):
            # indices of non-zero dosage
            time = torch.where(action[..., 0][:, i] != 0)[0]
            time = time * self.step_size # step is 1
            time_list.append(time)

        # B, N_DOSE
        self.times = torch.stack(time_list, dim=0)

    def dose_at_time(self, t):
        return self.dosage * torch.sum(
            torch.exp(self.kel * (self.times - t) * \
                (t >= self.times)) * (t >= self.times), dim=-1
        )

    def forward(self, t, y):
        # y: B, D
        # length of B
        # the expert variables are the last 4
        y_expert = y[:, -self.expert_dim:]
        Disease = y_expert[:, 0]
        ImmuneReact = y_expert[:, 1]
        Immunity = y_expert[:, 2]
        Dose2 = y_expert[:, 3]

        if not self.ablate:
            Dose = self.dose_at_time(t)

            dxdt1 = (
                Disease * self.k_disprog
                - Disease * Immunity ** self.HillCure * self.k_discure_immunity
                - Disease * ImmuneReact * self.k_discure_immunereact
            )

            dxdt2 = (
                Disease * self.k_immune_disease
                - ImmuneReact * self.k_immune_off
                + Disease * ImmuneReact * self.k_immune_feedback
                + (ImmuneReact ** self.HillPatho * self.emax_patho)
                / (self.ec50_patho ** self.HillPatho + ImmuneReact ** self.HillPatho)
                - Dose2 * ImmuneReact * self.k_dexa
            )

            dxdt3 = ImmuneReact * self.k_immunity

            dxdt4 = self.kel * Dose - self.kel * Dose2
        else:
            # mis-specified parameters
            dxdt1 = ImmuneReact
            dxdt2 = -1.0 * Disease * self.theta_1 * 100
            dxdt3 = Dose2
            dxdt4 = -1.0 * Immunity * self.theta_2 * 100
        
        dmldt = self.ml_net(y[:, :-self.expert_dim])
        
        # separate the expert and the neural ODE
        _dexpdt = torch.cat([dxdt1[..., None], 
                            dxdt2[..., None], 
                            dxdt3[..., None], 
                            dxdt4[..., None], 
                            ], 
                            dim=-1)
        
        # weight the expert opinions by beta
        # clamped_beta = torch.clamp(self.beta, 0, 1)
        # clamped_beta = torch.tanh(self.beta)
        # clamped_beta = torch.sigmoid(self.beta)
        
        if self.use_beta:
            clamped_beta = torch.clamp(self.beta, 0, 1)
            dexpdt = clamped_beta * _dexpdt + (1 - clamped_beta) * torch.ones_like(_dexpdt)
        else:
            dexpdt = _dexpdt
        # first term is the expert
        expert_term = torch.einsum("ij,bj->bi", self.expert_mat, dexpdt)
                                    
        clamped_alpha = torch.clamp(self.alpha, 0, 1)
        
        weighted_expert = torch.multiply(clamped_alpha, expert_term)
        
        # second term is the ml
        if not self.include_expert:
            ml_term = torch.einsum("bij,bj->bi",
                                    dmldt.reshape(-1, self.latent_dim, self.expert_dim),
                                    dexpdt)
            # weight ml_term by alpha
            weight_exp = torch.ones_like(dexpdt) * (1 - clamped_alpha)
            
            weight_ml = torch.ones((dexpdt.shape[0], self.latent_dim - self.expert_dim))
            
            weight_ml = torch.cat([weight_exp, 
                                    weight_ml], dim=-1)
            
            weighted_ml = weight_ml * ml_term
        else:
            ml_term = torch.einsum("bij,bj->bi",
                                    dmldt.reshape(-1, 
                                                  self.latent_dim-self.expert_dim, 
                                                  self.expert_dim),
                                    dexpdt)
            
            weight_exp = torch.ones_like(dexpdt) * (1 - clamped_alpha)
            
            weight_ml = torch.ones((dexpdt.shape[0], self.latent_dim - 2 * self.expert_dim))
            
            weight_ml = torch.cat([weight_exp, 
                                    weight_ml], dim=-1)
            
            weighted_ml = weight_ml * ml_term
        
        
        # sum the two terms
        dxdt = weighted_expert + weighted_ml
        
        return torch.cat([
            dxdt,
            dexpdt
        ], dim=-1)
        
        
class RocheCDEDecoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        latent_dim,
        action_dim,
        t_max,
        step_size,
        roche=True,
        ablate=False,
        method="dopri5",
        ode_step_size=None,
        include_expert = False,
        device=None,
        dtype=DTYPE,
        use_beta=False,
    ):
        super().__init__()

        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.step_size = step_size
        self.roche = roche
        self.ablate = ablate
        self.model_name = "RocheExpertDecoderCDE"
        self.include_expert = include_expert
        self.use_beta = use_beta

        if self.ablate:
            self.model_name = self.model_name + "Ablate"
            print("Running ablation study")

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t = torch.arange(0, 
                              t_max + step_size,
                              step_size, 
                              device=self.device, dtype=dtype)

        options = {}
        options.update({"method": method})
        options.update({"h": ode_step_size})
        options.update({"t0": 0.0})
        options.update({"t1": t_max + step_size})
        options.update({"rtol": 1e-7})
        options.update({"atol": 1e-8})
        options.update({"print_neval": True})
        options.update({"neval_max": 1000000})
        options.update({"safety": None})
        options.update({"t_eval": self.t})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})

        self.options = options
        
        self.ode = NewRocheCDE1(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            t_max=self.t_max,
            step_size=self.step_size,
            ablate=self.ablate,
            device=self.device,
            dtype=dtype,
            include_expert=self.include_expert,
            use_beta=self.use_beta,
        ).to(self.device)

        if not self.include_expert:
            self.output_function = nn.Sequential(
                nn.Linear(self.latent_dim, 
                        self.obs_dim, 
                        bias=True),
            ).to(self.device)
        else:
            self.output_function = nn.Sequential(
                nn.Linear(self.latent_dim - self.ode.expert_dim, 
                        self.obs_dim, 
                        bias=True),
            ).to(self.device)
            # print(f"{self.output_function}")


    def forward(self, init, a):
        self.ode.set_action(a)
        # solve ode
        
        h = dto(
            self.ode, 
            init, 
            self.t, 
            rtol=self.options["rtol"], 
            atol=self.options["atol"], 
            method=self.options["method"]
        )

        # generate output
        x_hat = self.output_function(h[:, :, :-self.ode.expert_dim])
        
        return x_hat, h
    
    
class CDEVariationalInference:
    epsilon = torch.finfo(DTYPE).eps
    def __init__(self, 
                 encoder,
                 decoder,
                 elbo=True,
                 prior_log_pdf=None,
                 mc_size=100,
                 kl_coeff1=0.2,
                 expert_encoder=None,
                 model_name=None):
        # extra expert encoder
        self.encoder = encoder
        self.decoder = decoder
        self.expert_encoder = expert_encoder
        self.prior_log_pdf = prior_log_pdf
        self.mc_size = mc_size
        self.kl_coeff1 = kl_coeff1
        self.elbo = elbo
        
        
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
        lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]
            
        if not self.elbo:
            raise NotImplementedError("Not implemented for no ELBO")


        if self.prior_log_pdf is None:
            # if no prior, use standard normal
            # analytic KL
            raise NotImplementedError("Not implemented yet for analytic KL")
        else:
            # monte carlo KL over batch
            kld_loss_vae = torch.mean(self.mc_kl_vae(mu, log_var), dim=0)
            
        # print(f"Likelihood loss: {lik}")
        # print(f"KL loss for VAE: {kld_loss_vae}")
        # print(f"KL loss for SDE: {kld_loss_sde}")
        loss = lik + kld_loss_vae * self.kl_coeff1
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
    
    
    

    
    
class CDEVariationalInference2:
    """Same as above but the expert model is included in the encoder dimension,
    so no extra argument of expert encoder is needed."""
    epsilon = torch.finfo(DTYPE).eps
    def __init__(self, 
                 encoder,
                 decoder,
                 elbo=True,
                 prior_log_pdf=None,
                 mc_size=100,
                 kl_coeff1=0.2,
                 model_name=None):
        # extra expert encoder
        self.encoder = encoder
        self.decoder = decoder
        self.prior_log_pdf = prior_log_pdf
        self.mc_size = mc_size
        self.kl_coeff1 = kl_coeff1
        self.kl = None
        self.elbo = elbo
        
        
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

        self.mu = mu
        self.log_var = log_var
        # print(self.decoder.ode.beta)
        # B, D
        if self.elbo:
            z = self.encoder.reparameterize(mu, log_var)
        else:
            raise NotImplementedError("Not implemented yet")
            z = mu
        self.z = z
        
        all_z = z
        # print(f"all_z shape: {all_z.shape}")
        
        # pass through decoder
        x_hat, kl_of_w = self.decoder(all_z, a)
        
        self.x_hat = x_hat
        # self.h_hat = h_hat

        # average over B (samples in mini batch)
        lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]
            
        if not self.elbo:
            raise NotImplementedError("Not implemented for no ELBO")


        if self.prior_log_pdf is None:
            # if no prior, use standard normal
            # analytic KL
            raise NotImplementedError("Not implemented yet for analytic KL")
        else:
            # monte carlo KL over batch
            kld_loss_vae = torch.mean(self.mc_kl_vae(mu, log_var), dim=0)
            
        # print(f"Likelihood loss: {lik}")
        # print(f"KL loss for VAE: {kld_loss_vae}")
        # print(f"KL loss for SDE: {kld_loss_sde}")
        loss = lik + kld_loss_vae * self.kl_coeff1
        
        self.kl = kld_loss_vae
        
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