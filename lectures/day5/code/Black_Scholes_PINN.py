#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Black–Scholes PDE for a European Call using PINNs

We solve the PDE:
    V_t + 0.5*sigma^2 * S^2 * V_SS + r*S*V_S - r*V = 0
with
    Terminal condition at t=T: V(S,T) = max(S-K, 0),
    Boundary conditions:
        V(0,t) = 0
        V(Smax,t) = Smax - K*exp(-r*(T-t))   (for large Smax)

We'll train a neural network V_theta(S,t) and use:
    - PDE residual in the interior of (S in [0,Smax], t in [0,T])
    - Boundary condition losses
    - Terminal condition loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters and PDE constants
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
K = 50.0        # strike
T = 1.0         # maturity
S_max = 100.0   # max price

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class BSNet(nn.Module):
    """
    Neural network for V(S,t).
    Input = (S,t), output = V(S,t).
    We'll use 3 hidden layers with Tanh activation.
    """
    def __init__(self, hidden_units=50):
        super(BSNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)

def bs_residual(model, S, t):
    """
    Compute PDE residual:
        V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V
    """
    # enable grads
    S.requires_grad_(True)
    t.requires_grad_(True)

    X = torch.cat([S, t], dim=1)  # shape (N,2)
    V = model(X)                  # shape (N,1)

    # Partial derivatives via autograd
    V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                              create_graph=True)[0]
    V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                              create_graph=True)[0]
    V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S),
                               create_graph=True)[0]

    residual = V_t + 0.5*sigma**2 * S**2 * V_SS + r*S*V_S - r*V
    return residual

def sampler(N_int, N_bc, N_term):
    """
    Return:
       - interior collocation points (S in [0,Smax], t in [0,T])
       - boundary points for S=0, S=Smax
       - terminal points for t=T
    """
    # interior
    S_int = torch.rand(N_int,1, device=device)*S_max
    t_int = torch.rand(N_int,1, device=device)*T

    # boundary S=0
    S_bc0 = torch.zeros(N_bc,1, device=device)
    t_bc0 = torch.rand(N_bc,1, device=device)*T

    # boundary S=Smax
    S_bc1 = torch.ones(N_bc,1, device=device)*S_max
    t_bc1 = torch.rand(N_bc,1, device=device)*T

    # terminal t=T
    S_term = torch.rand(N_term,1, device=device)*S_max
    t_term = torch.ones(N_term,1, device=device)*T

    return (S_int, t_int), (S_bc0, t_bc0), (S_bc1, t_bc1), (S_term, t_term)

def main():
    model = BSNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5000

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample points
        (S_int, t_int), (S_bc0, t_bc0), (S_bc1, t_bc1), (S_term, t_term) = sampler(2000, 200, 200)

        # PDE residual in interior
        r_int = bs_residual(model, S_int, t_int)
        loss_pde = torch.mean(r_int**2)

        # BC: V(0,t)=0
        X_bc0 = torch.cat([S_bc0, t_bc0], dim=1)
        V_bc0 = model(X_bc0)
        loss_bc0 = torch.mean(V_bc0**2)

        # BC: V(S_max,t) = S_max - K*exp(-r*(T-t))
        X_bc1 = torch.cat([S_bc1, t_bc1], dim=1)
        V_bc1 = model(X_bc1)
        bc1_target = S_max - K*torch.exp(-r*(T - t_bc1))
        loss_bc1 = torch.mean((V_bc1 - bc1_target)**2)

        # Terminal condition: V(S,T)=max(S-K,0)
        X_term = torch.cat([S_term, t_term], dim=1)
        V_term = model(X_term)
        term_target = torch.maximum(S_term - K, torch.zeros_like(S_term))
        loss_term = torch.mean((V_term - term_target)**2)

        # total loss
        loss = loss_pde + loss_bc0 + loss_bc1 + loss_term
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6e}")

    print("Training complete. One can evaluate at t=0 and compare to Black–Scholes closed-form.")

if __name__ == "__main__":
    main()
