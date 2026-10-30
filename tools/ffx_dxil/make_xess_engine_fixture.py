#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Make exactly reversible Q2RTX-unit inputs from four canonical probe frames."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def build(reference,width,height,output):
    if not 0<width<=4096 or not 0<height<=4096:
        raise ValueError('Invalid input dimensions')
    output.mkdir(parents=True,exist_ok=False)
    y,x=np.indices((height,width))
    scales=np.array([width,height],dtype=np.float32)
    for frame in range(4):
        source=reference/f'frame-{frame}';destination=output/f'frame-{frame}'
        color=np.fromfile(source/'color.rgba16f',dtype='<f2').reshape(height,width,4)
        motion=np.fromfile(source/'motion.rg16f',dtype='<f2').reshape(height,width,2)
        engine_color=color.copy()
        engine_color[:,:,:3]=color[:,:,:3].astype(np.float32)*128
        engine_color[:,:,3]=np.where((x+y)&1,.125,.625)
        engine_motion=np.empty((height,width,4),dtype='<f2')
        engine_motion[:,:,:2]=motion.astype(np.float32)/scales
        engine_motion[:,:,2]=7;engine_motion[:,:,3]=-13
        if not np.isfinite(engine_color).all() or not np.array_equal(
            (engine_color[:,:,:3].astype(np.float32)/128).astype('<f2'),color[:,:,:3]):
            raise ValueError('Colour cannot round-trip through engine units exactly')
        if not np.array_equal((engine_motion[:,:,:2].astype(np.float32)*scales).astype('<f2'),motion):
            raise ValueError('Motion cannot round-trip through RGBA16F UV units exactly')
        if (source/'depth.r32f').stat().st_size!=width*height*4:
            raise ValueError('Depth shape mismatch')
        destination.mkdir()
        engine_color.tofile(destination/'color.rgba16f')
        engine_motion.tofile(destination/'motion.rgba16f')
        shutil.copyfile(source/'depth.r32f',destination/'depth.r32f')
    (output/'fixture.json').write_text(json.dumps({'source':str(reference.resolve()),'input':[width,height],
        'color_scale':1/128,'output_scale':128,'motion_scale':[width,height],
        'metadata_channels_discarded':True,'lossless_half_roundtrip':True},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('width',type=int)
    parser.add_argument('height',type=int)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    build(args.reference,args.width,args.height,args.output)
