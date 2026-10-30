#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the recovered native checkerboard graph with history and compare all frames.

This validation harness uses host readback between stages. It is not the
reusable GPU-resident backend or a general game-input/quality benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import xess_conv_replay as convolutions
import xess_resolve_replay as resolve
import xess_native_bundle


def compare(reference, native, width=1280, height=720, reference_rgb_scale=1):
    expected, actual = reference.read_bytes(), native.read_bytes()
    if width<=0 or height<=0 or len(expected) != width*height*8 or len(actual) != len(expected):
        raise ValueError("Comparison image shape mismatch")
    source_hash=hashlib.sha256(expected).hexdigest()
    if not np.isfinite(reference_rgb_scale) or reference_rgb_scale<=0:
        raise ValueError("Invalid reference colour scale")
    if reference_rgb_scale!=1:
        scaled=np.frombuffer(expected,dtype='<f2').reshape(-1,4).copy()
        scaled[:,:3]=scaled[:,:3].astype(np.float32)*np.float32(reference_rgb_scale)
        expected=scaled.tobytes()
    a = np.frombuffer(actual, dtype="<f2").reshape(-1,4).astype(np.float64)
    b = np.frombuffer(expected, dtype="<f2").reshape(-1,4).astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite comparison output")
    difference = np.abs(a[:,:3] - b[:,:3])
    return {"bitwise_equal_rgba": actual == expected,
        "rgb_components": int(difference.size), "changed_rgb_components": int(np.count_nonzero(difference)),
        "maximum_rgb_error": float(difference.max()), "mean_rgb_error": float(difference.mean()),
        "changed_alpha_components": int(np.count_nonzero(a[:,3] != b[:,3])),
        "source_reference_sha256":source_hash,"reference_rgb_scale":reference_rgb_scale,
        "reference_sha256": hashlib.sha256(expected).hexdigest(), "native_sha256": hashlib.sha256(actual).hexdigest()}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = xess_native_bundle.load(args.bundle) if args.bundle else None
    if bundle:
        convolutions.require(args.frame_parameters is not None and args.reference_frames is not None,
            "Bundle replay requires frame parameters and explicit reference frames")
        (args.output / "convolutions").mkdir()
        (args.output / "convolutions/manifest.json").write_text(json.dumps(bundle["convolutions"]))
        dispatches = []
    else:
        dispatches = json.loads((args.trace / "dispatch-index.json").read_text())["dispatches"]
        convolutions.require(len(dispatches) == 56, "Expected the four-frame probe")
        convolutions.build(args.trace, args.shaders, args.output / "convolutions")
    frames = json.loads(args.frame_parameters.read_text()) if args.frame_parameters else None
    if frames is not None:
        convolutions.require(isinstance(frames,list) and len(frames)==4, "Expected four frame parameter objects")
    env = os.environ.copy()
    env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    report = {"scope": "native fixed checkerboard reset/history/history/reset replay",
        "reference_frames": str((args.reference_frames or args.trace / "frames").resolve()),
        "maximum_allowed_rgb_error": args.max_absolute_error, "frames": []}

    def execute(binary, arguments, log):
        with log.open("w") as output:
            subprocess.run([str((args.binaries / binary).resolve()), *map(str, arguments)],
                stdout=output, stderr=subprocess.STDOUT, env=env, check=True, timeout=90)
        contents = log.read_text()
        if any(token in contents for token in ("VUID-", "Validation Error", "Validation Warning")):
            raise ValueError("Vulkan validation reported a problem: " + str(log))

    for frame in range(4):
        for stage in range(1,13) if not bundle else []:
            original, current = dispatches[stage], dispatches[frame*14+stage]
            convolutions.require(all(original[k] == current[k] for k in ("shader", "constants", "groups")),
                "Convolution profile changed between frames")
        directory = args.output / f"frame-{frame}"
        directory.mkdir()
        input_hashes = {}
        if args.inputs:
            env["XESS_REPLAY_INPUTS"] = str((args.inputs / f"frame-{frame}").resolve())
        if env.get("XESS_REPLAY_INPUTS"):
            for name in ("color.rgba16f", "motion.rg16f", "depth.r32f"):
                path = Path(env["XESS_REPLAY_INPUTS"]) / name
                input_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        push_file = directory / "prepare-push.bin"
        if not bundle:
            prepare = dispatches[frame*14]
            convolutions.require(prepare["shader"] == "40553fb1ea6dd6f3", "Unexpected prepare shader")
            constants = prepare["constants"]["2"]
            convolutions.require(set(constants) == {str(i) for i in range(28)}, "Incomplete prepare constants")
            push = [int(constants[str(i)],16) for i in range(28)] + [0,8]
            convolutions.require(push[0:2] == [0x443c4000,0x43d40000], "Unexpected synthetic input size")
            push_file.write_bytes(struct.pack("<30I", *push))
        if frames is not None:
            current = frames[frame]
            previous = frames[frame-1]["jitter"] if frame else [0,0]
            convolutions.require(len(current["jitter"])==2 and type(current["reset"]) is bool,
                "Expected jitter pair and boolean reset")
            execute("xess_frame_constants", [*current["jitter"], *previous,
                int(current["reset"]), push_file, directory / "resolve-push.bin"], directory / "constants.log")
            push = list(struct.unpack("<30I",push_file.read_bytes()))
        prepare_args = [bundle["prepare"] if bundle else args.shaders / "40553fb1ea6dd6f3.spv", push_file,
            directory / "prepare.rgba32ui", directory / "prepare.rgba16f"]
        if frame:
            previous = args.output / f"frame-{frame-1}" / "resolved"
            prepare_args.extend([previous / "history.rgba16f", previous / "mask.r8"])
        execute("native_xess_prepare", prepare_args, directory / "prepare.log")
        execute("native_xess_convolutions", [args.output / "convolutions/manifest.json",
            directory / "prepare.rgba32ui", directory / "convolved"] +
            (["--resident-repeat"] if args.resident_convolutions else []), directory / "convolve.log")
        if bundle:
            (directory / "resolve-data").mkdir()
            (directory / "resolve-data/manifest.json").write_text(json.dumps(bundle["resolve"]))
        else:
            resolve.build(args.trace, args.shaders, directory / "resolve-data", args.padding_byte, frame)
        if frames is not None:
            path = directory / "resolve-data/manifest.json"
            manifest = json.loads(path.read_text())
            generated = (directory / "resolve-push.bin").read_bytes()
            manifest["push"] = list(struct.unpack("<23I",generated))
            manifest["frame_parameters"] = frames[frame]
            path.write_text(json.dumps(manifest,indent=2)+"\n")
            (directory / "resolve-data/push.bin").write_bytes(generated)
        execute("native_xess_resolve", [directory / "resolve-data/manifest.json",
            directory / "convolved/pass-12.rgba32ui", directory / "prepare.rgba16f",
            directory / "resolved"], directory / "resolve.log")
        result = compare((args.reference_frames or args.trace / "frames") / f"frame-{frame}.rgba16f", directory / "resolved/output.rgba16f")
        result["frame"] = frame
        result["input_sha256"] = input_hashes
        result["reset"] = bool(push[25])
        report["frames"].append(result)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"frame {frame}: bitwise_equal={result['bitwise_equal_rgba']} max_error={result['maximum_rgb_error']}", flush=True)
    first = (args.output / "frame-0/resolved/output.rgba16f").read_bytes()
    last = (args.output / "frame-3/resolved/output.rgba16f").read_bytes()
    report["native_reset_repeat_bitwise_equal"] = first == last
    report["passed"] = first == last and all((args.max_absolute_error != 0 or r["bitwise_equal_rgba"]) and
        r["maximum_rgb_error"] <= args.max_absolute_error and
        r["changed_alpha_components"] == 0 for r in report["frames"])
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("shaders", type=Path)
    parser.add_argument("binaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle", type=Path,
        help="Replay a relocatable native bundle; trace and shaders positional paths are unused")
    parser.add_argument("--resident-convolutions", action="store_true",
        help="Use the shared graph recorder and verify two submissions reuse the same convolution resources")
    parser.add_argument("--reference-frames", type=Path,
        help="Reference images captured on the replay GPU; defaults to the trace's frames")
    parser.add_argument("--inputs", type=Path,
        help="Per-frame captured inputs: frame-N/{color.rgba16f,motion.rg16f,depth.r32f}")
    parser.add_argument("--frame-parameters", type=Path,
        help="Four JSON objects with jitter:[x,y] and reset:boolean; generate frame constants using the shared header")
    parser.add_argument("--max-absolute-error", type=float, default=0)
    parser.add_argument("--padding-byte", type=lambda s: int(s,0), default=0)
    args = parser.parse_args()
    if not np.isfinite(args.max_absolute_error) or args.max_absolute_error < 0:
        parser.error("Maximum error must be finite and nonnegative")
    raise SystemExit(run(args))
