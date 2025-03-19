#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 2D Laplace Equation with Non-Zero BCs using PINNs

We solve:
    u_xx + u_yy = 0,   (x,y) in (0,1)x(0,1)
with boundary conditions:
    u(0,y) = 0,  u(1,y) = 1,  for y in [0,1]
    u(x,0) = 0,  u(x,1) = 1,  for x in [0,1]

Analytical solution (since it's linear from 0 to 1 on both edges):
    u(x,y) = (x + y)/2.

PINN approach:
    - A neural network u_theta(x,y).
    - PDE residual: u_xx + u_yy = 0 in the interior.
    - BC: matches the non-zero boundary values.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class LaplaceNet(nn.Module):
    """
    Neural network that takes (x,y) as input and outputs scalar u(x,y).
    We'll use a few hidden layers with Tanh activation.
    """
    def __init__(self, hidden_units=20):
        super(LaplaceNet, self).__init__()
        self.layer1 = nn.Linear(2, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, 1)
        self.activation = nn.Tanh()

    def forward(self, xy):
        # xy shape: (N,2) => columns [x, y]
        out = self.activation(self.layer1(xy))
        out = self.activation(self.layer2(out))
        out = self.layer3(out)
        return out

def laplace_residual(model, x, y):
    """
    PDE residual: u_xx + u_yy
    """
    # Enable gradients
    x.requires_grad_(True)
    y.requires_grad_(True)

    inp = torch.cat([x, y], dim=1)  # shape (N,2)
    u = model(inp)

    # First partial derivatives
    dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    dudy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]

    # Second partial derivatives
    d2udx2 = torch.autograd.grad(dudx, x, grad_outputs=torch.ones_like(dudx),
                                 create_graph=True)[0]
    d2udy2 = torch.autograd.grad(dudy, y, grad_outputs=torch.ones_like(dudy),
                                 create_graph=True)[0]

    # Laplace PDE residual = u_xx + u_yy
    res = d2udx2 + d2udy2
    return res

def boundary_condition(x, y):
    """
    Return the exact boundary value for (x,y).

    Boundary rules:
      - x=0 => u=0
      - x=1 => u=1
      - y=0 => u=0
      - y=1 => u=1
    """
    bc_val = torch.zeros_like(x)  # default 0
    tol = 1e-6

    # Right side x=1 => u=1
    mask_right = (x > 1.0 - tol)
    bc_val[mask_right] = 1.0

    # Top side y=1 => u=1
    mask_top = (y > 1.0 - tol)
    bc_val[mask_top] = 1.0

    # Left side x=0 => 0, bottom side y=0 => 0
    # already 0 by bc_val default

    return bc_val

def sample_boundary(N):
    """
    Sample boundary points on x=0, x=1, y=0, y=1
    Return a tensor of shape (4N, 2).
    """
    x_rand = torch.rand((N,1), device=device)
    y_rand = torch.rand((N,1), device=device)

    # top (y=1), bottom (y=0)
    top = torch.cat([x_rand, torch.ones_like(x_rand)], dim=1)
    bottom = torch.cat([x_rand, torch.zeros_like(x_rand)], dim=1)
    # left (x=0), right (x=1)
    left = torch.cat([torch.zeros_like(y_rand), y_rand], dim=1)
    right = torch.cat([torch.ones_like(y_rand), y_rand], dim=1)

    boundary = torch.cat([top, bottom, left, right], dim=0)
    return boundary

def main():
    model = LaplaceNet(hidden_units=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5000

    # Interior collocation points
    N_interior = 2000
    xy_interior = torch.rand((N_interior, 2), device=device)
    x_in = xy_interior[:,0:1]
    y_in = xy_interior[:,1:2]

    # Boundary points
    boundary_xy = sample_boundary(200)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 1) PDE residual in interior
        res = laplace_residual(model, x_in, y_in)
        loss_pde = torch.mean(res**2)

        # 2) Boundary condition
        x_b = boundary_xy[:,0:1]
        y_b = boundary_xy[:,1:2]
        u_b_pred = model(boundary_xy)            # predicted u on boundary
        u_b_true = boundary_condition(x_b, y_b)  # exact boundary value
        loss_bc = torch.mean((u_b_pred - u_b_true)**2)

        # Total loss
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | PDE loss: {loss_pde.item():.4e}, BC loss: {loss_bc.item():.4e}")

    # --- Evaluate at a grid and compare with exact solution (x+y)/2 ---
    Nx = 50
    Ny = 50
    x_space = torch.linspace(0,1,Nx, device=device)
    y_space = torch.linspace(0,1,Ny, device=device)
    XX, YY = torch.meshgrid(x_space, y_space, indexing='xy')  # shape Nx x Ny
    XY = torch.cat([XX.reshape(-1,1), YY.reshape(-1,1)], dim=1)  # shape (Nx*Ny,2)

    # PINN prediction on a grid
    u_pred = model(XY).detach().cpu().numpy().reshape(Nx, Ny)
    # Analytical solution
    x_np = XX.cpu().numpy()
    y_np = YY.cpu().numpy()
    u_true = 0.5*(x_np + y_np)  # (x+y)/2

    # --- Compute MSE in the domain
    mse_domain = np.mean((u_pred - u_true)**2)
    print("MSE in domain:", mse_domain)

    # --- Plot results side by side
    # We'll use imshow with subplots: (1) PINN solution, (2) Analytical, (3) Difference
    # Because 'indexing=xy', the shape is (Nx, Ny), where Nx is the # of x-points and Ny is # of y-points.
    # imshow expects the first dimension to be the vertical (rows) and the second dimension to be horizontal (columns).
    # We can either transpose or accept that x is along vertical axis. Here, we transpose so x is horizontal, y is vertical.

    u_pred_T = u_pred.T         # shape (Ny, Nx)
    u_true_T = u_true.T         # shape (Ny, Nx)
    diff_T   = (u_pred - u_true).T

    plt.figure(figsize=(12, 4))

    # 1) PINN solution
    plt.subplot(1, 3, 1)
    plt.imshow(u_pred_T, extent=[0,1,0,1], origin='lower', aspect='auto')
    plt.colorbar()
    plt.title("PINN solution")

    # 2) Analytical
    plt.subplot(1, 3, 2)
    plt.imshow(u_true_T, extent=[0,1,0,1], origin='lower', aspect='auto')
    plt.colorbar()
    plt.title("Analytical solution")

    # 3) Difference
    plt.subplot(1, 3, 3)
    plt.imshow(diff_T, extent=[0,1,0,1], origin='lower', aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.title("Difference (PINN - Analytical)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
