import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the analytical function (expensive model)
def model_function(theta):
    # Example analytical function with parameter theta
    return (theta - 2)**2 + 0.5 * np.sin(5 * theta)

# Hypothetical observation
y_obs = 1.0

# Define the misfit function: squared difference between model output and observation
def misfit(theta):
    return (model_function(theta) - y_obs)**2

# Generate sample data for theta in the domain [0,4]
theta_samples = np.linspace(0, 4, 50)
misfit_samples = misfit(theta_samples)

# Fit a surrogate model to the misfit_samples using polynomial regression (degree 4)
poly_coeffs = np.polyfit(theta_samples, misfit_samples, deg=4)
surrogate_model = np.poly1d(poly_coeffs)

# Define a function to optimize based on the surrogate model
def surrogate_objective(theta):
    return surrogate_model(theta)

# Find the local optimum of the surrogate model using a local optimizer
initial_guess = [2.0]  # initial guess for theta
result = minimize(surrogate_objective, initial_guess, bounds=[(0, 4)])
theta_opt = result.x[0]
misfit_opt = misfit(theta_opt)

print("Optimal theta (surrogate):", theta_opt)
print("Misfit at optimal theta:", misfit_opt)

# For comparison: determine the optimum by evaluating the true misfit on a fine grid
theta_fine = np.linspace(0, 4, 200)
misfit_fine = misfit(theta_fine)
idx_min = np.argmin(misfit_fine)
theta_true_opt = theta_fine[idx_min]
misfit_true_opt = misfit_fine[idx_min]

print("Optimal theta (grid search):", theta_true_opt)
print("Misfit at optimal theta (true):", misfit_true_opt)

# Plot the true misfit and the surrogate model
theta_plot = np.linspace(0, 4, 200)
plt.figure(figsize=(8, 5))
plt.plot(theta_plot, misfit(theta_plot), label='True Misfit')
plt.plot(theta_plot, surrogate_model(theta_plot), '--', label='Surrogate Model')
plt.scatter(theta_opt, misfit(theta_opt), color='red', label='Optimal (Surrogate)')
plt.xlabel('theta')
plt.ylabel('Misfit')
plt.legend()
plt.title("Surrogate Model Optimization")
plt.show()
