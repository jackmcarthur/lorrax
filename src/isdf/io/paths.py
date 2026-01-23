"""Path resolution utilities for input file processing."""
import os


def resolve_input_paths(params: dict, input_dir: str) -> dict:
	"""Resolve relative paths in params against input file directory.
	
	Modifies params in-place and returns it for convenience.
	
	Args:
		params: Parameter dictionary from read_cohsex_input
		input_dir: Directory containing the input file
	
	Returns:
		params with resolved paths
	"""
	def _resolve_path(path: str) -> str:
		return path if os.path.isabs(path) else os.path.join(input_dir, path)
	
	path_keys = ["wfn_file", "centroids_file", "output_file", "kin_ion_file", "eqp_output_file"]
	for key in path_keys:
		if key in params:
			params[key] = _resolve_path(params[key])
	
	return params

