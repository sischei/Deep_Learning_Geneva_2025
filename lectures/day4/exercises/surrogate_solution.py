import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) True function definition
###############################################################################
def true_function(a, x):
    """
    f(a, x) = (x - a)^3 + (x - a)
    """
    return (x - a)**3 + (x - a)


###############################################################################
# 2) Generate training data
###############################################################################
def generate_training_data(N=40, seed=123):
    """
    Randomly sample N points (a_i, x_i) in [1,2]x[0,4] and compute f(a_i, x_i).
    """
    torch.manual_seed(seed)
    # 'a' in [1,2]
    training_A = 1.0 + torch.rand(N) * (2.0 - 1.0)
    # 'x' in [0,4]
    training_X = 0.0 + torch.rand(N) * 4.0
    training_Y = true_function(training_A, training_X)
    return training_A, training_X, training_Y


###############################################################################
# 3) Define the GP Model in GPyTorch
###############################################################################
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Use constant mean + RBF kernel
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


###############################################################################
# 4) Train the GP
###############################################################################
def train_gp(model, likelihood, train_x, train_y, training_iter=50):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")


###############################################################################
# 5) Surrogate predictions
###############################################################################
def predict_surrogate(model, likelihood, a_grid, x_grid):
    """
    Predict the GP's mean and variance on a 2D grid of (a, x).
    """
    model.eval()
    likelihood.eval()

    Agrid, Xgrid = torch.meshgrid(a_grid, x_grid, indexing='ij')  # shape = [len(a_grid), len(x_grid)]
    eval_points = torch.stack([Agrid.reshape(-1), Xgrid.reshape(-1)], dim=-1)  # [len(a_grid)*len(x_grid), 2]

    with torch.no_grad():
        pred = likelihood(model(eval_points))
        predicted_mean = pred.mean
        predicted_var = pred.variance

    # Reshape to match the 2D shape
    predicted_mean = predicted_mean.reshape(len(a_grid), len(x_grid))
    predicted_var = predicted_var.reshape(len(a_grid), len(x_grid))
    return predicted_mean, predicted_var


###############################################################################
# 6a) Find best parameter a via a simple grid search
###############################################################################
def find_best_parameter_grid(model, likelihood, y=1.0,
                             a_search=torch.linspace(1.0, 2.0, 100),
                             x_search=torch.linspace(0.0, 4.0, 50)):
    """
    We compute mismatch(a) = sum_{x in x_search} [f_hat(a,x) - y]^2
    and pick the a that minimizes mismatch.
    """
    with torch.no_grad():
        mean_matrix, _ = predict_surrogate(model, likelihood, a_search, x_search)
        # mean_matrix shape: [#a_search, #x_search]
        mismatch = torch.sum((mean_matrix - y)**2, dim=1)  # sum over x dimension
        best_idx = torch.argmin(mismatch)
        best_a = a_search[best_idx]
    return best_a


###############################################################################
# 6b) Find best parameter a via BFGS optimization
###############################################################################
def find_best_parameter_bfgs(model, likelihood, y=1.0,
                             init_a=1.5,
                             x_search=torch.linspace(0.0, 4.0, 50),
                             max_iter=20):
    """
    Use PyTorch's LBFGS to minimize mismatch(a) = sum_{x in x_search} [f_hat(a,x) - y]^2
    w.r.t. 'a'. We let 'a' be a learnable parameter with gradient.
    """
    # We'll treat 'a' as a single parameter in [1,2].
    param_a = torch.tensor(init_a, dtype=torch.float32, requires_grad=True)

    # Define the LBFGS optimizer
    optimizer = torch.optim.LBFGS([param_a], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()

        # Expand param_a into shape [#x_search] so we can stack with x_search
        # Alternatively, we can repeat or unsqueeze
        a_vec = param_a.repeat(x_search.shape[0])  # shape [50]
        points = torch.stack([a_vec, x_search], dim=-1)  # [50,2]

        # Evaluate GP model at these points
        model.eval()
        likelihood.eval()
        f_pred = likelihood(model(points))
        f_mean = f_pred.mean  # shape [50]

        # Mismatch is sum of squared errors vs y
        # The smaller, the better
        mismatch = torch.sum((f_mean - y)**2)

        mismatch.backward()  # compute gradients w.r.t. param_a
        return mismatch

    # Run BFGS
    optimizer.step(closure)

    # After finishing, param_a is our best guess
    best_a = param_a.detach().clone()
    # Optionally clamp to [1,2] if we want to ensure it's in the domain
    best_a = torch.clamp(best_a, 1.0, 2.0)

    return best_a


###############################################################################
# PLOTTING HELPER 1:
# Compare true f(a, x) vs. GP for a single slice (e.g. x=2, vary a in [1,2])
###############################################################################
def plot_comparison_slice(model, likelihood, fixed_x=2.0):
    """
    We fix x, vary a in [1..2], and plot:
      1) The true function f(a, fixed_x)
      2) The GP's predicted mean
      3) The predicted 2-sigma confidence region
    This gives a 1D line plot to see how well the GP is matching the truth.
    """
    a_grid = torch.linspace(1.0, 2.0, 200)
    x_vals = torch.full_like(a_grid, fixed_x)
    test_points = torch.stack([a_grid, x_vals], dim=-1)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(test_points))
        mean = pred.mean
        lower, upper = pred.confidence_region()

    # True function
    f_true = true_function(a_grid, x_vals)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(a_grid, f_true, 'k--', label='True function')
    plt.plot(a_grid, mean, 'b', label='GP mean')
    plt.fill_between(a_grid.numpy(), lower.numpy(), upper.numpy(),
                     color='blue', alpha=0.2, label='GP ±2σ')
    plt.title(f"Slice at x={fixed_x}")
    plt.xlabel("a")
    plt.ylabel("f(a, x)")
    plt.legend()
    plt.tight_layout()
    plt.show()


###############################################################################
# PLOTTING HELPER 2:
# Plot mismatch(a) = sum_{x}(f_hat(a,x)-y)^2 vs. a, also show BFGS optimum
###############################################################################
def plot_mismatch_function(model, likelihood, y=1.0,
                           a_grid=torch.linspace(1.0, 2.0, 200),
                           x_grid=torch.linspace(0.0, 4.0, 50),
                           best_a=None, method=""):
    """
    Plots mismatch(a) = sum_{x in x_grid} [GP mean f(a,x) - y]^2
    across a in [1,2]. If best_a is provided, it is shown on the plot.
    """
    with torch.no_grad():
        mean_matrix, _ = predict_surrogate(model, likelihood, a_grid, x_grid)
        mismatch_vals = torch.sum((mean_matrix - y)**2, dim=1)  # shape [len(a_grid)]

    plt.figure(figsize=(6,4))
    plt.plot(a_grid, mismatch_vals, 'r-', label='Mismatch vs a')
    if best_a is not None:
        best_mismatch = np.interp(best_a.item(), a_grid.numpy(), mismatch_vals.numpy())
        plt.plot([best_a.item()], [best_mismatch], 'bo', label=f'Best a ({method})')
    plt.title("Mismatch Function vs. a")
    plt.xlabel("a")
    plt.ylabel(r"$\sum_x (f_{GP}(a,x) - 1)^2$")
    plt.legend()
    plt.tight_layout()
    plt.show()


###############################################################################
# MAIN
###############################################################################
def main():
    # 1) Generate training data
    training_A, training_X, training_Y = generate_training_data(N=40, seed=123)

    # 2) Convert to shape [N,2] for GP
    train_x = torch.stack([training_A, training_X], dim=-1)  # [N,2]
    train_y = training_Y  # [N]

    # 3) Create model + likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # 4) Train the GP
    train_gp(model, likelihood, train_x, train_y, training_iter=50)

    # 5) Plot a slice to compare GP vs. true function
    #    (Here we fix x=2.0 and vary a from [1..2])
    plot_comparison_slice(model, likelihood, fixed_x=2.0)

    # 6a) Grid search for best parameter
    best_a_grid = find_best_parameter_grid(model, likelihood, y=1.0)
    print(f"[GRID SEARCH] Best parameter a that matches y=1.0 is: {best_a_grid.item():.4f}")

    # 6b) BFGS search for best parameter
    best_a_bfgs = find_best_parameter_bfgs(model, likelihood, y=1.0,
                                          init_a=1.5, max_iter=20)
    print(f"[BFGS]        Best parameter a that matches y=1.0 is: {best_a_bfgs.item():.4f}")

    # 7) Plot mismatch function vs. a, highlight both solutions
    a_fine = torch.linspace(1.0, 2.0, 200)
    x_fine = torch.linspace(0.0, 4.0, 50)

    plot_mismatch_function(
        model, likelihood, y=1.0,
        a_grid=a_fine, x_grid=x_fine,
        best_a=best_a_grid, method="Grid"
    )

    plot_mismatch_function(
        model, likelihood, y=1.0,
        a_grid=a_fine, x_grid=x_fine,
        best_a=best_a_bfgs, method="BFGS"
    )

    # Optionally, you could show both on the same plot (you would simply
    # call plot_mismatch_function once and pass in both points,
    # but for clarity we separated them above).


if __name__ == "__main__":
    main()
