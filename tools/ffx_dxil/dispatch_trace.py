#!/usr/bin/env python3
"""Index direct compute calls in a vkd3d trace; not a replay-ready graph."""
import argparse
import copy
import json
import re


def index_trace(lines):
    lists = {}
    binding_thread = {}
    dispatches = []
    heaps = {}
    ranges = {}
    views = {}
    view_resources = {}
    for number, line in enumerate(lines, 1):
        created = re.search(r"Create(?:ShaderResourceView|UnorderedAccessView)_\w+: .*?resource ([0-9a-f]+),.*?descriptor (0x[0-9a-f]+)", line)
        if created:
            handle = int(created[2], 16)
            view_resources[handle] = created[1]
            views.pop(handle, None)  # A default/uncaptured replacement must not retain old bytes.
            continue
        declared = re.search(r"REFERENCE_RANGE root=(\w+) parameter=(\d+) range=(\d+) type=(\d+) count=(\d+) register=(\d+) space=(\d+) offset=(\d+)", line)
        if declared:
            root, parameter, index, kind, count, register, space, offset = declared.groups()
            if int(count) > 4096:
                raise ValueError(f"line {number}: unbounded/oversized descriptor range")
            ranges.setdefault(root, {}).setdefault(parameter, {})[index] = dict(
                type=int(kind), count=int(count), register=int(register), space=int(space), offset=int(offset))
            continue
        view = re.search(r"REFERENCE_VIEW kind=(\w+) descriptor=(\w+) size=(\d+) offset=(\d+) word=(\w+)", line)
        if view:
            kind, handle, size, offset, word = view.groups()
            handle, size, offset = int(handle, 0), int(size), int(offset)
            if offset == 0:
                views[handle] = dict(kind=kind, size=size, words={},
                    resource=view_resources.get(handle), source_descriptor=hex(handle))
            if handle not in views or views[handle]["size"] != size or offset + 4 > size:
                raise ValueError(f"line {number}: malformed view capture")
            views[handle]["words"][str(offset)] = word
            continue
        copied = re.search(r"CopyDescriptorsSimple_\w+: .*descriptor_count (\d+), dst_descriptor_range_offset (\w+), src_descriptor_range_offset (\w+),", line)
        if copied:
            count, destination, source = int(copied[1]), int(copied[2], 0), int(copied[3], 0)
            def stride_at(address):
                candidates = [h["stride"] for h in heaps.values()
                    if h["cpu"] <= address < h["cpu"] + h["count"] * h["stride"]]
                if len(candidates) != 1:
                    raise ValueError(f"line {number}: unresolved CPU descriptor heap")
                return candidates[0]
            ds, ss = (stride_at(destination), stride_at(source)) if count > 1 else (0, 0)
            snapshots = [copy.deepcopy(views.get(source + i * ss)) for i in range(count)]
            for i, snapshot in enumerate(snapshots):
                views[destination + i * ds] = snapshot
            continue
        heap = re.search(r"REFERENCE_HEAP iface=(\w+) cpu=(\w+) gpu=(\w+) count=(\d+) stride=(\d+) type=(\d+)", line)
        if heap:
            identity, cpu, gpu, count, stride, kind = heap.groups()
            heaps[identity] = dict(cpu=int(cpu, 0), gpu=int(gpu, 0),
                count=int(count), stride=int(stride), type=int(kind))
            if not int(stride):
                raise ValueError(f"line {number}: zero descriptor stride")
            continue
        match = re.search(r"([0-9a-f]+):trace:d3d12_command_list_(\w+): (.*)", line)
        if not match:
            continue
        thread, function, message = match.groups()
        iface = re.search(r"iface ([0-9a-f]+)", message)
        if function == "SetPipelineState" and iface:
            binding_thread[thread] = iface[1]
        shader = re.search(r"Binding compute module with hash: ([0-9a-f]{16})", message)
        if shader:
            if thread not in binding_thread:
                raise ValueError(f"line {number}: shader without command list")
            lists.setdefault(binding_thread[thread], {})["shader"] = shader[1]
            continue
        if not iface:
            continue
        key = iface[1]
        state = lists.setdefault(key, {})
        if function == "Reset":
            state.clear()
        elif function == "SetPipelineState":
            state.pop("shader", None)
        elif function == "SetComputeRootSignature":
            signature = re.search(r"root_signature ([0-9a-f]+)", message)
            if not signature:
                raise ValueError(f"line {number}: malformed root signature")
            if state.get("root_signature") != signature[1]:
                state["root_signature"] = signature[1]
                state["cbv"] = {}
                state["tables"] = {}
        elif function == "SetComputeRootConstantBufferView":
            value = re.search(r"root_parameter_index (\d+), address (0x[0-9a-f]+)", message)
            if not value:
                raise ValueError(f"line {number}: malformed CBV")
            state.setdefault("cbv", {})[value[1]] = value[2]
        elif function.startswith("SetComputeRootDescriptorTable"):
            value = re.search(r"root_parameter_index (\d+), base_descriptor (0x[0-9a-f]+)", message)
            if not value:
                raise ValueError(f"line {number}: malformed descriptor table")
            state.setdefault("tables", {})[value[1]] = value[2]
        elif function == "Dispatch":
            groups = re.search(r"x (\d+), y (\d+), z (\d+)", message)
            if not groups or not state.get("shader"):
                raise ValueError(f"line {number}: dispatch has unresolved shader/dimensions")
            table_bases = {}
            resource_views = {}
            for parameter, address in state.get("tables", {}).items():
                address = int(address, 16)
                candidates = [(identity, h) for identity, h in heaps.items()
                    if h["gpu"] and h["gpu"] <= address < h["gpu"] + h["count"] * h["stride"]]
                if len(candidates) > 1:
                    raise ValueError(f"line {number}: ambiguous GPU descriptor heap")
                if candidates:
                    identity, h = candidates[0]
                    offset = address - h["gpu"]
                    if offset % h["stride"]:
                        raise ValueError(f"line {number}: unaligned descriptor table")
                    table_bases[parameter] = dict(heap=identity,
                        cpu=hex(h["cpu"] + offset), index=offset // h["stride"],
                        stride=h["stride"])
                    definitions = ranges.get(state.get("root_signature"), {}).get(parameter, {})
                    entries = []
                    for definition in definitions.values():
                        for element in range(definition["count"]):
                            cpu = h["cpu"] + offset + (definition["offset"] + element) * h["stride"]
                            if cpu >= h["cpu"] + h["count"] * h["stride"]:
                                raise ValueError(f"line {number}: descriptor range exceeds heap")
                            entries.append(dict(register=definition["register"] + element,
                                space=definition["space"], type=definition["type"], cpu=hex(cpu),
                                view=copy.deepcopy(views.get(cpu))))
                    resource_views[parameter] = entries
            dispatches.append(dict(line=number, command_list=key, table_bases=table_bases,
                resource_views=resource_views,
                groups=[int(v) for v in groups.groups()], **copy.deepcopy(state)))
        elif function in ("ExecuteIndirect", "ExecuteBundle"):
            raise ValueError(f"line {number}: {function} is not supported by this indexer")
    if not dispatches:
        raise ValueError("no resolved direct dispatches in trace")
    return {"schema": 1, "replay_ready": False,
        "scope": "recorded direct compute calls, not proof of GPU execution",
        "missing": ["descriptor view contents", "resource lifecycle and barriers",
                    "constant sizes and dispatch-time bytes", "model layout"],
        "dispatches": dispatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("output", help="new local JSON file (never overwritten)")
    args = parser.parse_args()
    with open(args.trace, encoding="utf-8", errors="strict") as source:
        result = index_trace(source)
    with open(args.output, "x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
