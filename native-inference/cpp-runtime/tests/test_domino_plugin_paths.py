"""Check that DoMINO integration loads the exact plugin paths supplied by CMake."""

import ctypes
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import domino_exact_integration


class StopBeforeCudaBuild(Exception):
    pass


class DominoPluginPathTests(unittest.TestCase):
    def test_loads_supplied_plugins_without_assuming_linux_names(self):
        for filenames in (
            (
                "pnmir_tensorrt_exact_scalar_div_plugin.dll",
                "pnmir_tensorrt_exact_inverse_distance_blend_plugin.dll",
            ),
            (
                "libpnmir_tensorrt_exact_scalar_div_plugin.so",
                "libpnmir_tensorrt_exact_inverse_distance_blend_plugin.so",
            ),
        ):
            with (
                self.subTest(filenames=filenames),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp) / "SDK with spaces"
                root.mkdir()
                paths = [root / filename for filename in filenames]
                for path in paths:
                    path.touch()
                args = SimpleNamespace(
                    # Older integration code ignores explicit paths and guesses .so names.
                    plugin_dir=root,
                    scalar_div_plugin=paths[0],
                    inverse_distance_blend_plugin=paths[1],
                )
                logger = mock.Mock()
                trt = SimpleNamespace(
                    get_plugin_registry=mock.Mock(return_value=mock.Mock()),
                    Logger=logger,
                    Builder=mock.Mock(side_effect=StopBeforeCudaBuild),
                )
                with (
                    mock.patch.dict(sys.modules, {"tensorrt": trt}),
                    mock.patch.object(ctypes, "CDLL") as load,
                    self.assertRaises(StopBeforeCudaBuild),
                ):
                    domino_exact_integration.check_tensorrt(args, None)
                self.assertEqual(
                    load.call_args_list,
                    [
                        mock.call(str(path.resolve()), mode=ctypes.RTLD_GLOBAL)
                        for path in paths
                    ],
                )


if __name__ == "__main__":
    unittest.main()
