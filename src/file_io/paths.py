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
	# ONE join rule, the one :func:`resolve_input_path` below already
	# states: an EMPTY optional path stays empty.  Joining "" onto the deck
	# directory yields the DIRECTORY, which ``os.path.exists`` then reports
	# as present -- exactly how an unset ``static_gauge_hall_file`` (empty
	# by default since 2026-09-01) would have looked like a Hall artifact.
	def _resolve_path(path: str) -> str:
		return resolve_input_path(input_dir, path)

	path_keys = [
		"wfn_file", "centroids_file", "centroids_file_current",
		"kin_ion_file", "pseudo_dir", "bgw_vcoul_file", "bgw_vcoul_sym_wfn",
		"parallel_transport_file", "static_gauge_hall_file",
		"mpa_fit_reuse_file",
		"sigma_diag_file", "eqp0_file", "eqp1_file", "report_file",
	]
	for key in path_keys:
		if key in params:
			params[key] = _resolve_path(params[key])
	
	return params



def resolve_input_path(input_dir: str, path: str) -> str:
    """Join ``path`` onto ``input_dir`` unless empty or already absolute.

    (moved from gw/gw_driver_helpers.py 2026-07-09)
    """
    if path and (not os.path.isabs(path)):
        return os.path.join(input_dir, path)
    return path
