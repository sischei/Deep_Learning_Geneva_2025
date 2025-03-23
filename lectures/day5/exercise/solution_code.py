import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set to True for hard boundary enforcement, False for soft enforcement
use_hard_boundary = True  # Change to False to run the soft boundary method

# Define the domain boundaries
x0 = 0.0
x1 = np.pi / 2

# Define the neural network (for soft method, it approximates u(x); for hard, it approximates N(x))
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        out = self.net(x)
        if use_hard_boundary:
            # Hard method: Construct the trial solution that satisfies u(0)=0 and u(x1)=1
            # We choose the ansatz: u(x) = (2*x/π) + x*(x1 - x)*out
            trial = (2 * x / np.pi) + x * (x1 - x) * out
            return trial
        else:
            # Soft method: directly output u(x)
            return out

# Function to compute the ODE residual: u''(x) + u(x) = 0
def ode_residual(model, x):
    # Create a new tensor that is detached from the current computation graph and requires grad
    x = x.detach().requires_grad_()
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx + u

# Create the model instance
model = PINN()

# Set the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create collocation points in the interior of the domain
n_collocation = 1000
x_collocation = torch.linspace(x0, x1, n_collocation, requires_grad=True).view(-1, 1)

# Training loop
n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    residual = ode_residual(model, x_collocation)
    loss_pde = torch.mean(residual**2)
    
    if not use_hard_boundary:
        # Soft method: add boundary loss terms
        # Boundary condition at x = 0: u(0) = 0
        x_left = torch.tensor([[x0]], requires_grad=True)
        u_left = model(x_left)
        loss_bc_left = (u_left - 0.0)**2
        
        # Boundary condition at x = x1: u(x1) = 1
        x_right = torch.tensor([[x1]], requires_grad=True)
        u_right = model(x_right)
        loss_bc_right = (u_right - 1.0)**2
        
        loss = loss_pde + loss_bc_left + loss_bc_right
    else:
        # Hard method: the trial solution automatically satisfies the boundaries
        loss = loss_pde
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the trained model on a fine grid and compare with the analytical solution
x_test = torch.linspace(x0, x1, 100).view(-1, 1)
u_pred = model(x_test).detach().numpy()
x_test_np = x_test.detach().numpy()
u_analytical = np.sin(x_test_np)

print("\nx\tAnalytical u(x)\tPINN u(x)")
for xi, ua, up in zip(x_test_np, u_analytical, u_pred):
    print(f"{xi[0]:.4f}\t{ua[0]:.4f}\t\t{up[0]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(x_test_np, u_analytical, label='Analytical (sin(x))')
plt.plot(x_test_np, u_pred, 'o', label='PINN Prediction', markersize=4)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title("ODE Solution using PINN: u''(x) + u(x) = 0")
plt.show()
