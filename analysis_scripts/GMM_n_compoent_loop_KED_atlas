import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Limit the number of threads for BLAS libraries to avoid segmentation fault
with threadpool_limits(limits=1):
    # 1. Read and clean the data
    file_path = '/lcrc/group/ATLAS/users/jj/data/atlas/val_set_dcm_rates_2023.csv'
    folder_name = os.path.basename(os.path.dirname(file_path))
    data = pd.read_csv(file_path)
    data_cleaned = data.iloc[:, 1:].fillna(0)

    # 2. Randomly select a row of data and standardize it
    random_index = np.random.randint(0, len(data))
    random_row = data_cleaned.iloc[random_index].values.flatten()
    timestamp = data.iloc[random_index, 0]
    print(f"Selected timestamp: {timestamp}")

    # Standardize the data
    scaler = StandardScaler()
    random_row_scaled = scaler.fit_transform(random_row.reshape(-1, 1)).flatten()



    # 3. Plot the standardized histogram
    plt.figure(figsize=(10, 6))
    plt.hist(random_row_scaled, bins=30, edgecolor='black', density=True, alpha=0.5, color='gray',
             label=f'{folder_name} Histogram')

    # 4. Automatically select the initial number of components for GMM using BIC
    bic = []
    models = []
    n_components_range = range(1, 10)

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(random_row_scaled.reshape(-1, 1))
        bic.append(gmm.bic(random_row_scaled.reshape(-1, 1)))
        models.append(gmm)

    # Choose the model with the lowest BIC score
    best_n_components = n_components_range[np.argmin(bic)]
    print(f"Initial optimal number of components (by BIC): {best_n_components}")

    # 5. Iteratively reduce n_components if adjacent means are too close
    threshold = 0.55 * np.std(random_row_scaled)
    while best_n_components > 1:
        best_gmm = GaussianMixture(n_components=best_n_components, covariance_type='full', random_state=42)
        best_gmm.fit(random_row_scaled.reshape(-1, 1))

        # Get the means and sort them
        means = np.sort(best_gmm.means_.flatten())
        weights = best_gmm.weights_

        # Check if any adjacent means are closer than the threshold
        merge_needed = False
        for i in range(1, len(means)):
            if abs(means[i] - means[i - 1]) < threshold:
                merge_needed = True
                break

        # If merge is needed, reduce n_components and refit the GMM
        if merge_needed:
            best_n_components -= 1
            print(f"Reducing n_components to {best_n_components} due to close means.")
        else:
            # No merge needed, break the loop
            break

    print(f"Final optimal number of components: {best_n_components}")

    # 6. Fit the final GMM model
    final_gmm = GaussianMixture(n_components=best_n_components, covariance_type='full', random_state=42)
    final_gmm.fit(random_row_scaled.reshape(-1, 1))

    # Get the final means, covariances, and weights
    final_means = np.sort(final_gmm.means_.flatten())
    final_covariances = np.sqrt(final_gmm.covariances_).flatten()
    final_weights = final_gmm.weights_

    print(f"Final GMM means: {final_means}")
    print(f"Final GMM weights: {final_weights}")
    print(f"Final GMM peak count: {len(final_means)}")

    # 7. Perform density estimation using KDE
    # Adjust bandwidth ratio if needed
    bw_method_ratio = 1  # Start with a default ratio
    epsilon = 1e-5  # Small perturbation to avoid singular matrix

    # Add noise to avoid duplicate points
    random_row_scaled = random_row_scaled + np.random.normal(0, epsilon, random_row_scaled.shape)

    # Compute KDE with adjusted bandwidth
    base_bw = random_row_scaled.std() * (len(random_row_scaled) ** (-1 / 5))
    adjusted_bw = max(1e-3, bw_method_ratio * base_bw)  # Ensure positive bandwidth
    kde = gaussian_kde(random_row_scaled, bw_method=adjusted_bw)

    # Generate KDE PDF
    x = np.linspace(random_row_scaled.min(), random_row_scaled.max(), 1000)
    kde_pdf = kde(x)


    # Automatically detect peaks in the KDE
    data_range = random_row_scaled.max() - random_row_scaled.min()
    prominence = 0.01 * data_range
    kde_peaks, _ = find_peaks(kde_pdf, prominence=prominence)
    kde_peak_count = len(kde_peaks)
    print(f"KDE detected peak count: {kde_peak_count}")

    # 8. Plot the results
    plt.plot(x, kde_pdf, label=f'KDE (bw_ratio: {bw_method_ratio})', color='green')
    log_prob = final_gmm.score_samples(x.reshape(-1, 1))
    gmm_pdf = np.exp(log_prob)
    plt.plot(x, gmm_pdf, label='GMM Fit', color='red')

    # Mark GMM component peaks
    plt.scatter(final_means, kde(final_means), color='blue', marker='x', s=100, label=f'GMM Component Peaks (Count: {len(final_means)})')

    # Mark KDE detected peaks
    plt.scatter(x[kde_peaks], kde_pdf[kde_peaks], color='purple', marker='o', s=80, label=f'KDE Peaks (Count: {kde_peak_count})')

    # Set plot title and save the figure
    plt.xlabel('Standardized Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=13, loc='upper left')
    histogram_filename = f"final_gmm_components_{len(final_means)}_{folder_name}.png"
    plt.savefig(histogram_filename, dpi=300)
    plt.close()

    print(f"Plot saved as: {histogram_filename}")
