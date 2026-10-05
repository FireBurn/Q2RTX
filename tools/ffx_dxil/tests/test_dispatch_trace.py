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
    def test_root_constant_partial_writes_and_unknown_bytes(self):
        rows = [line("SetPipelineState", "iface 0001, pipeline_state 0002."),
                line("SetPipelineState", "Binding compute module with hash: 0123456789abcdef."),
                line("SetComputeRootSignature", "iface 0001, root_signature 0004."),
                line("SetComputeRoot32BitConstants", "iface 0001, root_parameter_index 2, constant_count 2, data 0010, dst_offset 0."),
                "REFERENCE_ROOT_CONSTANT list=0001 parameter=2 offset=0 word=3f800000",
                "REFERENCE_ROOT_CONSTANT list=0001 parameter=2 offset=1 word=00000007",
                line("Dispatch", "iface 0001, x 1, y 1, z 1."),
                line("SetComputeRoot32BitConstants", "iface 0001, root_parameter_index 2, constant_count 1, data 0010, dst_offset 1."),
                line("Dispatch", "iface 0001, x 1, y 1, z 1."),
                line("SetComputeRootSignature", "iface 0001, root_signature 0005."),
                line("Dispatch", "iface 0001, x 1, y 1, z 1.")]
        ds = tool.index_trace(rows)["dispatches"]
        self.assertEqual(ds[0]["constants"], {"2": {"0": "3f800000", "1": "00000007"}})
        self.assertEqual(ds[1]["constants"], {"2": {"0": "3f800000", "1": None}})
        self.assertEqual(ds[2]["constants"], {})

    def test_cbv_snapshot_and_uncaptured_replacement(self):
        rows = ["REFERENCE_HEAP iface=0003 cpu=0x1000 gpu=0x300000000 count=8 stride=64 type=0",
                "REFERENCE_RANGE root=0004 parameter=0 range=0 type=2 count=1 register=0 space=0 offset=0",
                "REFERENCE_CBV descriptor=0x1000 address=0xffff800000001000 size=256",
                line("SetPipelineState", "iface 0001, pipeline_state 0002."),
                line("SetPipelineState", "Binding compute module with hash: 0123456789abcdef."),
                line("SetComputeRootSignature", "iface 0001, root_signature 0004."),
                line("SetComputeRootDescriptorTable_embedded_64_16", "iface 0001, root_parameter_index 0, base_descriptor 0x300000000."),
                line("Dispatch", "iface 0001, x 1, y 1, z 1."),
                "0024:trace:d3d12_device_CreateConstantBufferView_embedded: iface 0010, desc 0000, descriptor 0x1000.",
                line("Dispatch", "iface 0001, x 1, y 1, z 1.")]
        result = tool.index_trace(rows)["dispatches"]
        first = result[0]["resource_views"]["0"][0]["view"]
        self.assertEqual(first["address"], "0xffff800000001000")
        self.assertEqual(first["size"], 256)
        self.assertIsNone(result[1]["resource_views"]["0"][0]["view"])

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
                "0024:trace:d3d12_device_CreateShaderResourceView_embedded: iface 0010, resource 00abcd, desc 0030, descriptor 0x2000.",
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
        self.assertEqual(entry["view"]["resource"], "00abcd")
