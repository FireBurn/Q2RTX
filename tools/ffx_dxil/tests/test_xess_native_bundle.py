# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import xess_native_bundle as bundle


class NativeBundleTests(unittest.TestCase):
    def test_relocation_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source';source.mkdir()
            data=b'fixture';(source/'asset.bin').write_bytes(data)
            manifest={'schema':'xess.native-bundle','version':1,
                'profile':'753x424-1280x720-unit-exposure','prepare':'asset.bin',
                'convolutions':{'file':'asset.bin'},'resolve':{'shader':'asset.bin'},
                'assets':{'asset.bin':hashlib.sha256(data).hexdigest()}}
            (source/'bundle.json').write_text(json.dumps(manifest))
            relocated=root/'relocated';source.rename(relocated)
            result=bundle.load(relocated)
            self.assertEqual(result['prepare'],str(relocated/'asset.bin'))
            (relocated/'asset.bin').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'): bundle.load(relocated)

    def test_external_path_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest={'schema':'xess.native-bundle','version':1,
                'profile':'753x424-1280x720-unit-exposure',
                'assets':{'../outside':'irrelevant'}}
            (root/'bundle.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'asset path'): bundle.load(root)


if __name__=='__main__': unittest.main()
