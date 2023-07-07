import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Set file path
file_path = f'{os.getcwd()}/MNIST/data/'

# Read data from files
df = pd.read_csv(file_path + 'data.dat', sep=r'\s{3}', engine='python', header=None)
labeldf = pd.read_csv(file_path + 'label.dat', sep=r'\s{3}', engine='python', header=None)

# Read labels from label.dat file

true_labels = labeldf.values.flatten()
# Preprocess dataset
feature_means = np.mean(df, axis=0)
feature_stds = np.std(df, axis=0)

X = (df - feature_means) / feature_stds

# Perform PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X.T) # (1990,4)

# EM algorithm

# Function to initialize GMM parameters
def initialize_parameters(X, num_components):
    _, num_features = X.shape
    weights = np.ones(num_components) / num_components
    means = np.random.randn(num_components, num_features)
    covariances = np.zeros((num_components, num_features, num_features))
    for k in range(num_components):
        covariances[k] = np.eye(num_features)
    return weights, means, covariances

# Function to compute Gaussian PDF
def gaussian_pdf(X, mean, covariance):
    _, num_features = X.shape
    diff = X - mean
    cov_inv = np.linalg.inv(covariance)
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    denominator = np.sqrt(np.nan_to_num((2 * np.pi) ** num_features * np.linalg.det(covariance)))
    return np.nan_to_num(np.exp(exponent)) / denominator


# Expectation step
def expectation_step(X, weights, means, covariances,epsilon=1e-6):
    num_samples, num_components = X.shape[0], weights.shape[0]
    responsibilities = np.zeros((num_samples, num_components))
    for k in range(num_components):
        covariance = covariances[k]
        covariance += np.eye(covariance.shape[0]) * epsilon  # Add regularization to the covariance matrix
        responsibilities[:, k] = weights[k] * gaussian_pdf(X, means[k], covariances[k])
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True) + epsilon
    return responsibilities

# Maximization step
def maximization_step(X, responsibilities):
    _, num_components, num_features = X.shape[0], responsibilities.shape[1], X.shape[1]
    weights = np.mean(responsibilities, axis=0)
    means = responsibilities.T @ X / np.sum(responsibilities, axis=0, keepdims=True).T
    covariances = np.zeros((num_components, num_features, num_features))
    for k in range(num_components):
        diff = X - means[k]
        weighted_diff = responsibilities[:, k].reshape(-1, 1) * diff
        covariances[k] = (weighted_diff.T @ diff) / np.sum(responsibilities[:, k])
    return weights, means, covariances

# Compute log-likelihood
def log_likelihood(X, weights, means, covariances):
    num_samples, num_components = X.shape[0], weights.shape[0]
    likelihoods = np.zeros((num_samples, num_components))
    for k in range(num_components):
        likelihoods[:, k] = weights[k] * gaussian_pdf(X, means[k], covariances[k])
    return np.sum(np.log(np.sum(likelihoods, axis=1)))

# Fit GMM with C = 2
def fit_gmm(X, num_components, num_iterations=100):
    weights, means, covariances = initialize_parameters(X, num_components)
    log_likelihoods = []
    for iteration in range(num_iterations):
        responsibilities = expectation_step(X, weights, means, covariances)
        weights, means, covariances = maximization_step(X, responsibilities)
        log_likelihoods.append(log_likelihood(X, weights, means, covariances))
        if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-3:
            break
        
    return weights, means, covariances, log_likelihoods, responsibilities

# Fit GMM with C = 2
num_components = 2
weights, means, covariances, log_likelihoods, responsibilities = fit_gmm(X_pca, num_components)
# k-means
kmeans = KMeans(n_clusters=num_components)

# Fit the KMeans model to your data
kmeans.fit(X_pca)

# Get the cluster labels assigned to each data point
label = kmeans.labels_

# Map the means back to the original space
mapped_means = pca.inverse_transform(means)  # Assuming you have used PCA for dimensionality reduction

# Reformat means into 28-by-28 matrices
reformatted_means = mapped_means.reshape((-1,28, 28))


# Convert output labels to integers
output_labels = np.argmax(responsibilities, axis=1)

# Calculate misclassification rate
misclassification_2=np.mean(output_labels[true_labels == 2] != 0)
misclassification_6 = np.mean(output_labels[true_labels == 6] != 1)
# k-means mis classifiacation
misclassification_2k = np.mean(label[true_labels == 2] != 0)
misclassification_6k = np.mean(label[true_labels == 6] != 1)

# Print the misclassification rate
print("Misclassification Rate of 2: {:.2%}\nMisclassification Rate of 2 for k means: {:.2%}".format(misclassification_2,misclassification_2k))
print("Misclassification Rate of 6: {:.2%}\nMisclassification Rate of 6 for k means: {:.2%}".format(misclassification_6,misclassification_6k))

# Display the means as images

fig, axes = plt.subplots(nrows=1, ncols=num_components, figsize=(10, 4))

# Iterate over each mean and display it in a subplot
for i, mean in enumerate(reformatted_means):
    axes[i].imshow(mean, cmap='gray')
    axes[i].set_title("Component {}".format(i+1))
    axes[i].axis('off')

plt.tight_layout()
plt.show()

#You have two components (C = 2)
cov1 = covariances[0]
cov2 = covariances[1]


# Visualize the intensity images or heat maps of the covariance matrices
plt.imshow(cov1, cmap='gray')
plt.title("Covariance Matrix of 6")
plt.show()
plt.imshow(cov2, cmap='gray')
plt.title("Covariance Matrix of 2")
plt.show()
# Plot log-likelihoods
plt.plot(log_likelihoods)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('EM Algorithm - GMM Log-Likelihood')
plt.show()
