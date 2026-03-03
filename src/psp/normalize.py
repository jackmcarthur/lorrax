from __future__ import annotations

from dataclasses import is_dataclass, fields, replace
from collections.abc import Sequence
from typing import Any
import numpy as np


def _convert_value_field_if_numeric(value: Any, field_name: str) -> Any:
	# Only convert dataclass textual content stored under a field named "value".
	if field_name == "value" and isinstance(value, str):
		text = value.strip()
		if text:
			try:
				arr = np.fromstring(text, sep=' ')
				if arr.size > 0:
					return arr
			except Exception:
				pass
	return value


def normalize_dataclass(obj: Any) -> Any:
	"""Return a new dataclass where string fields named "value" are np.ndarray.

	Recurses through nested dataclasses and sequences, but only converts
	fields literally named "value" when they contain whitespace-delimited numbers.
	Also preserves dynamically attached attributes (e.g., 'jjj') that are not
	declared as dataclass fields.
	"""
	if not is_dataclass(obj):
		return obj
	field_names = {f.name for f in fields(obj)}
	updates: dict[str, Any] = {}
	for f in fields(obj):
		value = getattr(obj, f.name)
		if is_dataclass(value):
			updates[f.name] = normalize_dataclass(value)
		elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
			updates[f.name] = type(value)(
				normalize_dataclass(v) if is_dataclass(v) else v for v in value
			)
		else:
			updates[f.name] = _convert_value_field_if_numeric(value, f.name)
	new_obj = replace(obj, **updates)
	# Preserve any extra attributes that aren't declared dataclass fields
	try:
		for name, val in vars(obj).items():
			if name not in field_names:
				try:
					setattr(new_obj, name, val)
				except Exception:
					pass
	except TypeError:
		# Some dataclasses may use slots and lack __dict__
		pass
	return new_obj
