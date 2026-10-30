#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the persistent native graph using a local bundle and reference frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from xess_native_bundle import load
from replay_xess_sequence import compare


def run(args):
    bundle=load(args.bundle) if not args.packed_model else None
    frames=json.loads(args.frames.read_text())
    if not isinstance(frames,list) or len(frames)!=4:
        raise ValueError("Expected four frames")
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=args.packed_model.resolve() if args.packed_model else args.output/'resolved-bundle.json'
    if bundle is not None:
        manifest.write_text(json.dumps(bundle,indent=2)+'\n')
    env=os.environ.copy()
    env['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
    env.pop('VK_LAYER_ENABLES',None)
    env['VK_LAYER_VALIDATE_SYNC']='1'
    env.pop('XESS_REPLAY_ENGINE_INPUTS',None)
    if args.engine_inputs:env['XESS_REPLAY_ENGINE_INPUTS']='1'
    with (args.output/'native.log').open('w') as log:
        subprocess.run([str(args.binary.resolve()),str(manifest),str(args.frames),
            str(args.inputs),str(args.output/'frames'),*map(str,args.extent)],env=env,stdout=log,
            stderr=subprocess.STDOUT,check=True,timeout=120)
    log=(args.output/'native.log').read_text()
    if any(token in log for token in ['VUID-','Validation Error','Validation Warning','SYNC-HAZARD']):
        raise ValueError('Validation diagnostics in native.log')
    results=[]
    for i in range(4):
        result=compare(args.reference/f'frame-{i}.rgba16f',args.output/'frames'/f'frame-{i}.rgba16f',
            *args.extent[2:],reference_rgb_scale=128 if args.engine_inputs else 1)
        result['frame']=i
        result['parameters']=frames[i]
        result['input_sha256']={name:hashlib.sha256((args.inputs/f'frame-{i}'/name).read_bytes()).hexdigest()
            for name in ['color.rgba16f','motion.rgba16f' if args.engine_inputs else 'motion.rg16f','depth.r32f']}
        results.append(result)
        print(f"frame {i}: bitwise_equal={result['bitwise_equal_rgba']}")
    report={'scope':'persistent fixed-profile 14-pass Vulkan graph',
        'extent':args.extent,
        'engine_input_units':args.engine_inputs,
        'packed_model_sha256':hashlib.sha256(args.packed_model.read_bytes()).hexdigest() if args.packed_model else None,
        'reference':str(args.reference.resolve()),'frames':results,
        'passed':all(r['bitwise_equal_rgba'] for r in results)}
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','binary','frames','inputs','reference','output']:
        parser.add_argument(name,type=Path)
    parser.add_argument('--extent',type=int,nargs=4,default=[753,424,1280,720],
        metavar=('INPUT_WIDTH','INPUT_HEIGHT','OUTPUT_WIDTH','OUTPUT_HEIGHT'))
    parser.add_argument('--packed-model',type=Path,
        help='Load one XESSVK1 file; the bundle positional argument is unused')
    parser.add_argument('--engine-inputs',action='store_true',
        help='Use GPU conversion for colour x128 / RGBA UV motion; compare output in engine units')
    raise SystemExit(run(parser.parse_args()))
