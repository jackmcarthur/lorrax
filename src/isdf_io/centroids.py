"""Centroid file loading and preprocessing."""
import numpy as np


def load_centroids(centroids_file: str, fft_grid: tuple) -> tuple[np.ndarray, np.ndarray, int]:
	"""Load centroids and compute grid indices with periodic boundary handling.
	
	Args:
		centroids_file: Path to file with fractional centroid coordinates
		fft_grid: Tuple (nx, ny, nz) of FFT grid dimensions
	
	Returns:
		Tuple of (centroids_frac, centroid_indices, n_rmu) where:
		- centroids_frac: (n_rmu, 3) fractional coordinates
		- centroid_indices: (n_rmu, 3) integer grid indices
		- n_rmu: Number of centroids
	"""
	centroids_frac = np.loadtxt(centroids_file)
	n_rmu = int(centroids_frac.shape[0])
	
	# Convert to integer grid indices and handle periodic boundary
	centroid_indices = np.round(centroids_frac * np.array(fft_grid)).astype(int)
	for i in range(3):
		centroid_indices[centroid_indices[:, i] == fft_grid[i], i] = 0
	
	return centroids_frac, centroid_indices, n_rmu

