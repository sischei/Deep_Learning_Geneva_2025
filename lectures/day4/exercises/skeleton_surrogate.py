import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the analytical function (expensive model)
def model_function(theta):
    # Example analytical function with parameter theta
    return (theta - 2)**2 + 0.5*np.sin(5*theta)

# Hypothetical observation
y_obs = 1.0

# Define the misfit function: squared difference between the model and the observation
def misfit(theta):
    return (model_function(theta) - y_obs)**2

# Generate sample data for theta in the domain [0,4]
theta_samples = np.linspace(0, 4, 50)
misfit_samples = misfit(theta_samples)

# %%% Fill in: Fit a surrogate model to the misfit_samples (e.g., using polynomial regression of degree 4) %%%
# Hint: Use np.polyfit to obtain polynomial coefficients and np.poly1d to build the surrogate function.
poly_coeffs = None  # Replace None with your code
surrogate_model = np.poly1d(poly_coeffs)

# %%% Fill in: Find the local optimum of the surrogate model using a local optimizer %%%
# Hint: You can use scipy.optimize.minimize on the surrogate model function.
result = None  # Replace None with your optimization call

theta_opt = result.x[0]
misfit_opt = misfit(theta_opt)

print("Optimal theta (surrogate):", theta_opt)
print("Misfit at optimal theta:", misfit_opt)

# Plot the true misfit and the surrogate model
theta_plot = np.linspace(0, 4, 200)
plt.figure(figsize=(8, 5))
plt.plot(theta_plot, misfit(theta_plot), label='True Misfit')
plt.plot(theta_plot, surrogate_model(theta_plot), '--', label='Surrogate Model')
plt.scatter(theta_opt, misfit(theta_opt), color='red', label='Optimal theta')
plt.xlabel('theta')
plt.ylabel('Misfit')
plt.legend()
plt.title("Surrogate Model Optimization")
plt.show()
