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
    for number, line in enumerate(lines, 1):
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
            dispatches.append(dict(line=number, command_list=key,
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
