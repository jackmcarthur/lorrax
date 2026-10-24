"""Accepted module image assets stay self-contained and refuse tampering."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_spec = importlib.util.spec_from_file_location(
    'module_cubin_store', Path(__file__).resolve().parents[1]
    / 'config/perlmutter/module/cubin_store.py')
store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(store)


class StoreAssets(unittest.TestCase):
    def test_accepted_copy_and_refusals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, source = root/'cache', root/'source'
            cache.mkdir(); source.mkdir()
            (source/'SOURCE_COMMIT').write_text('revision\n')
            payload = root/'pruneable-image'
            payload.write_bytes(b'asset-copy fixture; native-key validation belongs to rtc_build')
            (cache/'image.cubin').symlink_to(payload)
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps(dict(source_revision='revision',
                bundle_id='bundle', cache=str(cache), device_kind='test')))
            self.assertEqual(store.capture(receipt, source), 1)
            captured = source/'cubin_store'
            self.assertFalse((captured/'image.cubin').is_symlink())
            payload.unlink()
            self.assertEqual(store.verify(captured, 'revision', 'bundle')['images'][0]['bytes'],
                             (captured/'image.cubin').stat().st_size)
            destination = root/'published'
            self.assertEqual(store.copy(captured, destination, 'revision', 'bundle'), 1)
            self.assertEqual((captured/'image.cubin').read_bytes(),
                             (destination/'image.cubin').read_bytes())
            for revision, bundle in [('other', 'bundle'), ('revision', 'other')]:
                with self.assertRaisesRegex(ValueError, 'accepted source/bundle'):
                    store.verify(captured, revision, bundle)
            (captured/'image.cubin').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'image differs'):
                store.copy(captured, root/'refused', 'revision', 'bundle')
            self.assertFalse((root/'refused').exists())
            # No images is not a successful CUDA acceptance store.
            (cache/'image.cubin').unlink()
            with self.assertRaisesRegex(ValueError, 'inventory'):
                store.capture(receipt, source)


if __name__ == '__main__':
    unittest.main()
