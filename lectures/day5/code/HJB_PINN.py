#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Simple 1D Hamilton–Jacobi–Bellman (HJB) Equation using PINNs

We solve a toy HJB:
    -r*V(x) + max_{a in [-1,1]} [ (x+a)*V'(x) - alpha*a^2 ] = 0
on x in [0,1], with boundary conditions:
    V(0) = 0,
    V(1) = 1.

We approximate the "max" over a discrete set of controls in [-1,1].
PINN approach:
    - A small network V(x).
    - Residual = -r*V + max_{a in [-1,1]} [ (x+a)*V'(x) - alpha*a^2 ].
    - Then add boundary constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

r = 0.05
alpha = 0.1  # penalty coefficient for a^2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class HJBNet(nn.Module):
    """
    Neural net for V(x).
    We'll keep it simple: input x -> some hidden layers -> output V.
    """
    def __init__(self, hidden_units=20):
        super(HJBNet, self).__init__()
        self.layer1 = nn.Linear(1, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.layer1(x))
        out = self.activation(self.layer2(out))
        out = self.layer3(out)
        return out

def hjb_residual(model, x):
    """
    HJB PDE residual:
      -r*V(x) + max_{a in [-1,1]} [ (x+a)*V'(x) - alpha*a^2 ] = 0.
    We'll approximate max by sampling discrete a values, e.g. a in {-1, -0.5, 0, 0.5, 1}.
    """
    x.requires_grad_(True)
    V = model(x)
    dVdx = torch.autograd.grad(
        V, x,
        grad_outputs=torch.ones_like(V),
        create_graph=True
    )[0]

    # Sample discrete controls in [-1,1].
    A = torch.linspace(-1.0, 1.0, steps=5).to(device)  # e.g., -1, -0.5, 0, 0.5, 1

    # Evaluate expression for each a
    values = []
    for a in A:
        # expression = (x+a)*V'(x) - alpha*a^2
        expr = (x + a)*dVdx - alpha*(a**2)
        values.append(expr)

    # Combine across a dimension => shape (N, 5)
    stacked_vals = torch.stack(values, dim=-1)
    # We approximate the max over a by taking the maximum along dim=-1
    max_expr, _ = torch.max(stacked_vals, dim=-1, keepdim=True)

    # PDE residual = -r*V + max_expr
    res = -r*V + max_expr
    return res

def main():
    model = HJBNet(hidden_units=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5000

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Sample interior points
        x_interior = torch.rand((100,1), device=device)  # x in (0,1)

        # PDE residual
        res = hjb_residual(model, x_interior)
        loss_pde = torch.mean(res**2)

        # Boundary conditions: V(0)=0, V(1)=1
        x0 = torch.tensor([[0.0]], device=device)
        x1 = torch.tensor([[1.0]], device=device)
        V0 = model(x0)
        V1 = model(x1)
        loss_bc = torch.mean((V0 - 0.0)**2) + torch.mean((V1 - 1.0)**2)

        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

    print("HJB training complete. Evaluate or refine for your specific control sets.")

if __name__ == "__main__":
    main()
