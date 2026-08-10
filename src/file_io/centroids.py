"""Centroid file loading and preprocessing."""
import numpy as np


def read_centroid_source(centroids_file: str) -> str | None:
	"""The ``centroid_source:`` stamp a centroid file carries, or ``None``.

	``centroid.kmeans_cli`` writes one comment line naming the selector that
	produced the table (``kmeans_pivoted_cholesky`` or
	``pivoted_cholesky_full_grid``).  Files written before that stamp existed
	have no line, and ``None`` means exactly that — "this file does not say",
	which is a different fact from "this file was made by k-means" and must
	not be collapsed into it.

	Cheap by construction: the stamp is in the first few comment lines, so
	this stops at the first non-comment row rather than reading the table.
	"""
	try:
		with open(centroids_file, "r") as fh:
			for line in fh:
				if not line.startswith("#"):
					break
				if "centroid_source:" in line:
					return line.split("centroid_source:", 1)[1].strip()
	except OSError:
		return None
	return None


#: Selector name (the ``--centroid-selector`` / ``centroid_selector`` value)
#: → the ``centroid_source:`` stamp its output carries.  ONE table, read by
#: the writer (``centroid.kmeans_cli``) and by the reader below, so the deck
#: assertion cannot drift from the stamp by a typo in either.
CENTROID_SOURCE_STAMPS = {
	"kmeans": "kmeans_pivoted_cholesky",
	"pivoted_full_grid": "pivoted_cholesky_full_grid",
}

#: Deck value → the stamps that satisfy it (a tuple, so a selector may grow
#: a second spelling without the assertion becoming a chain of ``or``).
_SELECTOR_STAMPS = {k: (v,) for k, v in CENTROID_SOURCE_STAMPS.items()}


def assert_centroid_selector(centroids_file: str, declared: str,
                             *, print_fn=print) -> None:
	"""Refuse when the deck's ``centroid_selector`` and the file disagree.

	The deck key cannot GENERATE the centroids — ``centroid.kmeans_cli`` does
	not read the deck, by design — so this is the whole of what it does, and
	it is worth doing: the two selectors produce point sets whose Sigma_x
	differ by tens of meV, the file name carries no hint of which is which,
	and a centroid table is typically named in a deck months after it was
	made.  Empty ``declared`` asserts nothing, which is every existing deck.

	An UNSTAMPED file under a non-empty assertion is a refusal, not a pass:
	"the file does not say" and "the file says what you asked for" are
	different facts and the second is the one being claimed.
	"""
	declared = (declared or "").strip()
	if not declared:
		return
	if declared not in _SELECTOR_STAMPS:
		raise ValueError(
			f"centroid_selector = {declared!r} is not a selector. "
			f"Known: {', '.join(sorted(_SELECTOR_STAMPS))}, or \"\" to "
			f"assert nothing.")
	found = read_centroid_source(centroids_file)
	if found is None:
		raise ValueError(
			f"centroid_selector = {declared!r}, but {centroids_file} carries "
			f"no `centroid_source:` stamp, so it CANNOT be checked.  It was "
			f"written before the stamp existed (or not by "
			f"centroid.kmeans_cli).  Regenerate it with "
			f"`--centroid-selector {declared}`, or drop the deck key — an "
			f"unstamped file is not evidence of anything, and passing it "
			f"here would turn the assertion into decoration.")
	if found not in _SELECTOR_STAMPS[declared]:
		raise ValueError(
			f"centroid_selector = {declared!r}, but {centroids_file} was "
			f"written by {found!r}.  The two selectors put the interpolation "
			f"points in different places and Sigma_x differs by tens of meV "
			f"between them, so this is a wrong-basis run, not a naming "
			f"quibble.  Regenerate the table or correct the deck.")
	print_fn(f"  [centroids] selector asserted and confirmed: {declared} "
	         f"(stamp {found!r})")


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

