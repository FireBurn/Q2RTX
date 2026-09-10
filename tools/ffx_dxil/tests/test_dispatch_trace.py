import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("dispatch_trace",
    Path(__file__).resolve().parents[1] / "dispatch_trace.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def line(function, message):
    return f"0024:trace:d3d12_command_list_{function}: {message}"


class TraceTests(unittest.TestCase):
    def test_binding_and_snapshot(self):
        rows = [line("SetPipelineState", "iface 0001, pipeline_state 0002."),
                line("SetPipelineState", "Binding compute module with hash: 0123456789abcdef."),
                line("SetComputeRootConstantBufferView", "iface 0001, root_parameter_index 2, address 0x1000."),
                line("Dispatch", "iface 0001, x 3, y 90, z 1."),
                line("SetComputeRootConstantBufferView", "iface 0001, root_parameter_index 2, address 0x1100."),
                line("Dispatch", "iface 0001, x 3, y 90, z 1.")]
        result = tool.index_trace(rows)
        self.assertFalse(result["replay_ready"])
        self.assertEqual(result["dispatches"][0]["cbv"], {"2": "0x1000"})
        self.assertEqual(result["dispatches"][1]["cbv"], {"2": "0x1100"})

    def test_unresolved_and_indirect_fail(self):
        for function, message in [("Dispatch", "iface 0001, x 1, y 1, z 1."),
                                  ("ExecuteIndirect", "iface 0001.")]:
            with self.assertRaises(ValueError):
                tool.index_trace([line(function, message)])

    def test_empty_fails(self):
        with self.assertRaises(ValueError):
            tool.index_trace([])

    def test_gpu_table_maps_to_cpu_heap(self):
        rows = ["REFERENCE_HEAP iface=0003 cpu=0x1001 gpu=0x300000000 count=8 stride=64 type=0",
                line("SetPipelineState", "iface 0001, pipeline_state 0002."),
                line("SetPipelineState", "Binding compute module with hash: 0123456789abcdef."),
                line("SetComputeRootDescriptorTable_embedded_64_16",
                     "iface 0001, root_parameter_index 0, base_descriptor 0x300000080."),
                line("Dispatch", "iface 0001, x 1, y 1, z 1.")]
        result = tool.index_trace(rows)["dispatches"][0]["table_bases"]["0"]
        self.assertEqual(result["cpu"], "0x1081")
        self.assertEqual(result["index"], 2)
        rows[-2] = rows[-2].replace("0x300000080", "0x300000081")
        with self.assertRaises(ValueError):
            tool.index_trace(rows)

    def test_range_copy_retains_dispatch_snapshot(self):
        rows = ["REFERENCE_HEAP iface=0003 cpu=0x1000 gpu=0x300000000 count=8 stride=64 type=0",
                "REFERENCE_RANGE root=0004 parameter=0 range=0 type=0 count=1 register=7 space=2 offset=0",
                "REFERENCE_VIEW kind=SRV descriptor=0x2000 size=4 offset=0 word=0000002a",
                "0024:trace:d3d12_device_CopyDescriptorsSimple_embedded: iface 0010, descriptor_count 1, dst_descriptor_range_offset 0x1000, src_descriptor_range_offset 0x2000, descriptor_heap_type 0.",
                line("SetPipelineState", "iface 0001, pipeline_state 0002."),
                line("SetPipelineState", "Binding compute module with hash: 0123456789abcdef."),
                line("SetComputeRootSignature", "iface 0001, root_signature 0004."),
                line("SetComputeRootDescriptorTable_embedded_64_16", "iface 0001, root_parameter_index 0, base_descriptor 0x300000000."),
                line("Dispatch", "iface 0001, x 1, y 1, z 1."),
                "REFERENCE_VIEW kind=SRV descriptor=0x1000 size=4 offset=0 word=000000ff"]
        entry = tool.index_trace(rows)["dispatches"][0]["resource_views"]["0"][0]
        self.assertEqual((entry["register"], entry["space"]), (7, 2))
        self.assertEqual(entry["view"]["words"], {"0": "0000002a"})
