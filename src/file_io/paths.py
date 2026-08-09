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
	
	path_keys = [
		"wfn_file", "centroids_file", "kin_ion_file",
		"sigma_diag_file", "eqp0_file", "eqp1_file",
		# The multipole fit store (``compute_mode = mpa``).  OPTIONAL, and
		# the emptiness guard below is what makes it safe to list here: an
		# unset optional path is the empty string, and joining that onto
		# ``input_dir`` yields the deck's own DIRECTORY, which is a
		# perfectly good path that exists and is not a file — so "not set"
		# would arrive downstream looking set.
		"mpa_fit_file",
	]
	for key in path_keys:
		if key in params and params[key]:
			params[key] = _resolve_path(params[key])
	
	return params



def resolve_input_path(input_dir: str, path: str) -> str:
    """Join ``path`` onto ``input_dir`` unless empty or already absolute.

    (moved from gw/gw_driver_helpers.py 2026-07-09)
    """
    if path and (not os.path.isabs(path)):
        return os.path.join(input_dir, path)
    return path
