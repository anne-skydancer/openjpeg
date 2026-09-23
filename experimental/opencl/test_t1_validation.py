# SPDX-License-Identifier: BSD-2-Clause
"""Reject unsupported/invalid capture descriptors before GPU submission."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_t1 import load_blocks


class DescriptorValidation(unittest.TestCase):
    def test_rejects_invalid_descriptors(self):
        valid = dict(version=1, width=1, height=1, orientation=0, numbps=1,
                     style=0, roi=0, corrupted=0, coefficients=[3],
                     segments=[[1, 1]], bytes="00")
        mutations = [dict(version=2), dict(roi=1), dict(corrupted=1),
                     dict(style=16), dict(style=64), dict(style=-1),
                     dict(width=0), dict(height=-1), dict(width=1025),
                     dict(orientation=4), dict(numbps=31), dict(coefficients=[]),
                     dict(segments=[[2, 1]]), dict(segments=[[-1, 1]]),
                     dict(segments=[[1, 2]]), dict(segments=[[1, 0]]),
                     dict(style=1, numbps=8, segments=[[1, 22]]),
                     dict(bytes="zz")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocks.jsonl"
            path.write_text(json.dumps(valid))
            self.assertEqual(len(load_blocks(path)), 1)
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    case = copy.deepcopy(valid)
                    case.update(mutation)
                    path.write_text(json.dumps(case))
                    with self.assertRaises(ValueError):
                        load_blocks(path)
            path.write_text("")
            with self.assertRaises(ValueError):
                load_blocks(path)


if __name__ == "__main__":
    unittest.main()
