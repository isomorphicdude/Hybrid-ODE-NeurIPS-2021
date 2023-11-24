import numpy as np
import torch
import torchsde
from torch import nn
from torchbnn._impl import diffeq_layers, utils

import sim_config
from global_config import DTYPE, get_device


def make_y_net(input_size,
               verbose=False,
               hidden_width=2,
               explicit_params=True,
               ):
    """Evolving state network."""
    _input_size = (input_size[0],) + input_size[1:]
    # print(f"y_net input size: {_input_size}")
    layers = []
    if hidden_width is not None:
        layers.extend(
            # use linear layers for now
            [
                diffeq_layers.Linear(_input_size[0], hidden_width),
                diffeq_layers.DiffEqWrapper(nn.ReLU()),
                diffeq_layers.Linear(hidden_width, _input_size[0]),
                diffeq_layers.DiffEqWrapper(nn.Tanh()),
                # diffeq_layers.Linear(hidden_width, _input_size[0]),
                # diffeq_layers.DiffEqWrapper(nn.Tanh()),
            ]
        )
    else:
        layers.extend(
            [
                diffeq_layers.Linear(_input_size[0], _input_size[0]),
                diffeq_layers.DiffEqWrapper(nn.Tanh()),
            ]
        )
            # if i == 1:
                # print(f"y_net (augmented) input size: {_input_size}")
            # layers.append(diffeq_layers.Print(name=f"group: {i}, block: {j}"))

    # explicit parameters for the evolving weights
    y_net = diffeq_layers.DiffEqSequential(*layers, 
                                           explicit_params=explicit_params)
    
    # return augmented input size b/c y net should have same input / output
    return y_net, _input_size



def make_y_net_CDE(input_size,
                   control_vec_dim,
                   hidden_size=(1, 64, 1),
                   explicit_params=True,
               ):
    """
    Evolving state network, returns a matrix 
    shape (input_size[0], input_size[0] x control_vec_dim)
    whose output can be reshaped to (input_size[0], control_vec_dim)
    """
    
    _input_size = (input_size[0],) + input_size[1:]
    # print(f"y_net input size: {_input_size}")
    layers = []
    
    #TODO: use for loop to create hidden layers
    
    if hidden_size is not None and len(hidden_size) > 1:
        for i in range(len(hidden_size)-1):
            layers.extend(
                # use linear layers for now
                [
                    diffeq_layers.Linear(_input_size[0], hidden_size[i]),
                    diffeq_layers.DiffEqWrapper(nn.ReLU()),
                ]
            )
        layers.extend(
            [
                diffeq_layers.Linear(hidden_size[-1], _input_size[0] * control_vec_dim),
                diffeq_layers.DiffEqWrapper(nn.Tanh()),
            ]
        )
    else:
        layers.extend(
            [
                diffeq_layers.Linear(_input_size[0], _input_size[0] * control_vec_dim),
                diffeq_layers.DiffEqWrapper(nn.Tanh()),
            ]
        )

    # explicit parameters for the evolving weights
    y_net = diffeq_layers.DiffEqSequential(*layers, 
                                           explicit_params=explicit_params)
    return y_net, _input_size




def make_w_net(in_features, 
               out_features,
               hidden_sizes=(2, ),
               activation="relu"
               ):
    """Evolving weights network."""
    activation = utils.select_activation(activation)
    all_sizes = (in_features,) + tuple(hidden_sizes) + (out_features,)

    layers = []
    for i, (in_size, out_size) in enumerate(zip(all_sizes[:-1], all_sizes[1:]), 1):
        layers.append(nn.Linear(in_size, 
                                out_size))
        if i + 1 < len(all_sizes):
            layers.append(activation())
        else:  # Last layer needs zero initialization.
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            layers.append(utils.select_activation("tanh")())
            
    return diffeq_layers.DiffEqWrapper(nn.Sequential(*layers))



# create ode function
def expertODE(t, 
              Disease, 
              ImmuneReact, 
              Immunity, 
              Dose2,
              dose_func,
              device=None,
              in_SDE=False):
    """
    Computes the RHS of the expert ODE.
    
    Parameters:   
        - t: times
        - Disease: disease state
        - ImmuneReact: immune reaction state
        - Immunity: immunity state
        - Dose2: drug dose
        - dose_func: dose function at time t
        - device: device to use
        - in_SDE: if the ODE is used in an SDE
    """
    # parameters for the expert ODE
    dtype = DTYPE
    if device is None:
        device = get_device()
    dc = sim_config.RochConfig()
    HillCure = nn.Parameter(torch.tensor(dc.HillCure, device=device, dtype=dtype))
    HillPatho = nn.Parameter(torch.tensor(dc.HillPatho, device=device, dtype=dtype))
    ec50_patho = nn.Parameter(torch.tensor(dc.ec50_patho, device=device, dtype=dtype))
    emax_patho = nn.Parameter(torch.tensor(dc.emax_patho, device=device, dtype=dtype))
    k_dexa = nn.Parameter(torch.tensor(dc.k_dexa, device=device, dtype=dtype))
    k_discure_immunereact = nn.Parameter(
        torch.tensor(dc.k_discure_immunereact, device=device, dtype=dtype)
    )
    k_discure_immunity = nn.Parameter(torch.tensor(dc.k_discure_immunity, device=device, dtype=dtype))
    k_disprog = nn.Parameter(torch.tensor(dc.k_disprog, device=device, dtype=dtype))
    k_immune_disease = nn.Parameter(torch.tensor(dc.k_immune_disease, device=device, dtype=dtype))
    k_immune_feedback = nn.Parameter(torch.tensor(dc.k_immune_feedback, device=device, dtype=dtype))
    k_immune_off = nn.Parameter(torch.tensor(dc.k_immune_off, device=device, dtype=dtype))
    k_immunity = nn.Parameter(torch.tensor(dc.k_immunity, device=device, dtype=dtype))
    kel = nn.Parameter(torch.tensor(dc.kel, device=device, dtype=dtype))
    
    Dose = dose_func(t)

    dxdt1 = (
        Disease * k_disprog
        - Disease * Immunity ** HillCure * k_discure_immunity
        - Disease * ImmuneReact * k_discure_immunereact
    )

    dxdt2 = (
        Disease * k_immune_disease
        - ImmuneReact * k_immune_off
        + Disease * ImmuneReact * k_immune_feedback
        + (ImmuneReact ** HillPatho * emax_patho)
        / (ec50_patho ** HillPatho + ImmuneReact ** HillPatho)
        - Dose2 * ImmuneReact * k_dexa
    )

    dxdt3 = ImmuneReact * k_immunity

    dxdt4 = kel * Dose - kel * Dose2
    
    if not in_SDE:
        return torch.cat([
            dxdt1[..., None],
            dxdt2[..., None],
            dxdt3[..., None],
            dxdt4[..., None]
        ], dim=-1)
    else:
        return torch.cat([
            dxdt1[..., None],
            dxdt2[..., None],
            dxdt3[..., None],
            dxdt4[..., None]
        ], dim=-1) * 14.0
    
    