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
		"kin_ion_file", "bgw_vcoul_file", "bgw_vcoul_sym_wfn",
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


#: Cray DVS's fstype in ``/proc/self/mounts``.  DVS forwards a client's I/O to its server
#: nodes, and a collective MPI-IO write onto it runs at stripe width 1 ("DVS stripe width of
#: 36 was requested but DVS set it to 1": job 61163888 wrote 440 GB of tmp/zeta_q.h5 to CFS
#: at about 70 MB/s and timed out).
DVS_FSTYPE = "dvs"
#: The largest collective dataset ``file_io.slab_io`` writes onto DVS (about 15 s there).
DVS_COLLECTIVE_LIMIT = 1 << 30


def _mount_table(table: str = "/proc/self/mounts") -> tuple:
    """``((mount point, fstype), ...)`` of this process; empty where there is no table."""
    try:
        with open(table) as f:
            rows = [line.split() for line in f]
    except OSError:
        return ()
    # The kernel octal-escapes spaces, tabs and backslashes in mount points (\040).
    unescape = lambda s: s.encode("latin-1").decode("unicode_escape")
    return tuple((unescape(r[1]), r[2]) for r in rows if len(r) >= 3)


def _under(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return root == "/" or path == root or path.startswith(root + "/")


def dvs_reason(path, *, mounts=None, site=None) -> str | None:
    """Why ``path`` is on DVS as a compute node sees it, or None.

    Either its mount (the longest mount point holding its real path) is ``dvs``, or the
    site profile's ``[filesystems] compute_dvs_roots`` holds it: a Perlmutter login node
    mounts CFS as GPFS, so a login-node preflight (``lx doctor``) needs the site's word for
    what the compute nodes see.  ``mounts`` and ``site`` replace this process's mount table
    and ``runtime.network_env.site_profile()`` (tests)."""
    real = os.path.realpath(os.fspath(path))
    rows = _mount_table() if mounts is None else tuple(mounts)
    holding = [m for m in rows if _under(real, m[0])]
    if holding:
        point, fstype = max(holding, key=lambda m: len(m[0].rstrip("/")))
        if fstype == DVS_FSTYPE:
            return f"{real} is on the DVS mount {point}"
    if site is None:
        from runtime.network_env import site_profile
        site = site_profile() or {}
    for root in (site.get("filesystems") or {}).get("compute_dvs_roots", ()):
        if _under(real, root):
            return (f"{real} is under {root}, which {site.get('site', 'this site')}'s compute "
                    "nodes mount through DVS (config/<site>/site.toml compute_dvs_roots)")
    return None
