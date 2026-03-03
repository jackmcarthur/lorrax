import os
import sys
from pathlib import Path
from dataclasses import is_dataclass, fields
from collections.abc import Sequence

from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.parsers.config import ParserConfig
import xml.etree.ElementTree as ET


def detect_version(upf_path: str | Path) -> str:
	# Decide schema by inspecting the root tag
	import xml.etree.ElementTree as ET
	root = ET.parse(str(upf_path)).getroot()
	tag = root.tag
	local = tag.split('}', 1)[1] if tag.startswith('{') and '}' in tag else tag
	if local.upper() == 'UPF':
		return '2.0.1'
	return '0.99'


def load_upf(upf_path: str | Path):
	upf_path = Path(upf_path)
	version = detect_version(upf_path)
	parser = XmlParser(config=ParserConfig(
		fail_on_unknown_properties=False,
		fail_on_unknown_attributes=False,
	))
	if version == '2.0.1':
		from psp import upf_model_2_0_1 as model
		root_cls = model.Upf
		# Preprocess: rename PP_BETA.n -> PP_BETA and inject index=n for compatibility
		try:
			tree = ET.parse(str(upf_path))
			root = tree.getroot()
			def rename_beta(elem: ET.Element):
				for child in list(elem):
					tag = child.tag
					if isinstance(tag, str) and 'PP_BETA.' in tag:
						try:
							before, after = tag.rsplit('PP_BETA.', 1)
							num = int(after)
							child.tag = before + 'PP_BETA'
							child.set('index', str(num))
						except Exception:
							pass
					rename_beta(child)
			rename_beta(root)
			# Write to temp and parse from there
			from tempfile import NamedTemporaryFile
			with NamedTemporaryFile('wb', delete=False, suffix='.upf') as tmp:
				tree.write(tmp, encoding='utf-8', xml_declaration=True)
				tmp_path = Path(tmp.name)
			try:
				obj = parser.from_path(tmp_path, root_cls)
				# Annotate PP_BETA entries with FR j and l from PP_SPIN_ORB/PP_RELBETA.n if present
				try:
					# Build index->(lll,jjj) mapping from the original tree
					relmap: dict[int, tuple[int, float]] = {}
					for spin_orb in root.iter():
						if isinstance(spin_orb.tag, str) and spin_orb.tag.endswith('PP_SPIN_ORB'):
							for child in list(spin_orb):
								tag = child.tag
								if isinstance(tag, str) and 'PP_RELBETA.' in tag:
									idx = child.get('index')
									lll = child.get('lll')
									jjj = child.get('jjj')
									if idx is not None and jjj is not None:
										try:
											relmap[int(idx)] = (int(lll) if lll is not None else None, float(jjj))
										except Exception:
											pass
					# Attach to dataclass beta entries
					if hasattr(obj, 'pp_nonlocal') and hasattr(obj.pp_nonlocal, 'pp_beta'):
						for beta in obj.pp_nonlocal.pp_beta:
							try:
								bidx = int(getattr(beta, 'index', 0))
								if bidx in relmap:
									rl, rj = relmap[bidx]
									setattr(beta, 'jjj', rj)
									setattr(beta, 'lll', rl)
							except Exception:
								pass
				except Exception:
					pass
				return obj
			finally:
				try:
					tmp_path.unlink(missing_ok=True)
				except Exception:
					pass
		except Exception:
			# Fallback to direct parse if preprocessing fails
			return parser.from_path(upf_path, root_cls)
	else:
		from psp import upf_model as model
		# pick the root (pseudo) class
		root_cls = None
		for name, obj in vars(model).items():
			if is_dataclass(obj) and getattr(getattr(obj, 'Meta', None), 'name', '').lower() in {'pseudo', 'upf'}:
				root_cls = obj
				break
		if root_cls is None:
			raise RuntimeError('Could not find root class for 0.99 model')
	return parser.from_path(upf_path, root_cls)


def _brief_value(value):
	if value is None:
		return 'None'
	if isinstance(value, (int, float, bool)):
		return repr(value)
	if isinstance(value, str):
		return repr(value if len(value) <= 80 else value[:77] + '...')
	return f'<{type(value).__name__}>'


def _print_dataclass_tree(obj, name='root', indent='', depth=0, max_depth=5, list_limit=5):
	cls_name = type(obj).__name__
	print(f"{indent}{name}: {cls_name}")
	if depth >= max_depth:
		print(f"{indent}  ... (max depth {max_depth} reached)")
		return
	for f in fields(obj):
		value = getattr(obj, f.name)
		child_name = f.name
		if is_dataclass(value):
			_print_dataclass_tree(value, name=child_name, indent=indent + '  ', depth=depth + 1, max_depth=max_depth, list_limit=list_limit)
		elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
			length = len(value)
			print(f"{indent}  {child_name}: list(len={length})")
			show = min(length, list_limit)
			for i in range(show):
				item = value[i]
				if is_dataclass(item):
					_print_dataclass_tree(item, name=f'[{i}]', indent=indent + '    ', depth=depth + 1, max_depth=max_depth, list_limit=list_limit)
				else:
					print(f"{indent}    [{i}]: {_brief_value(item)}")
			if length > show:
				print(f"{indent}    ... (+{length - show} more)")
		else:
			print(f"{indent}  {child_name}: {_brief_value(value)}")


def print_dataclass_tree(obj, max_depth=5, list_limit=5):
	_print_dataclass_tree(obj, name='UPF', indent='', depth=0, max_depth=max_depth, list_limit=list_limit)


def main(argv=None):
	if argv is None:
		argv = sys.argv[1:]
	# Optional flag: --numpy to convert numeric text to numpy arrays
	use_numpy = True
	if argv and argv[0] == '--numpy':
		use_numpy = True
		argv = argv[1:]
	#if len(argv) != 1:
	#	print('Usage: uv run load_upf.py [--numpy] PSEUDOPOTENTIAL.upf')
	#	return 2
	pseudo_arg = "tests/cohsex_prod/Mo_ONCV_PBE_FR-1.0.upf" #argv[0]
	pseudo_path = os.path.abspath(pseudo_arg)
	if not os.path.isfile(pseudo_path):
		print(f'Error: file not found: {pseudo_arg}')
		return 1
	obj = load_upf(pseudo_path)
	if use_numpy:
		from psp.normalize import normalize_dataclass
		obj = normalize_dataclass(obj)
	print(f"Loaded: {os.path.basename(pseudo_path)} -> {type(obj).__name__}")
	# convert D_ij to square array
	n_proj = obj.pp_header.number_of_proj
	obj.pp_nonlocal.pp_dij.value = obj.pp_nonlocal.pp_dij.value.reshape(n_proj, n_proj)
	print_dataclass_tree(obj)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
