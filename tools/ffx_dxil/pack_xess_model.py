#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pack a verified local native bundle into one runtime XESSVK1 model file.

The output contains recovered payloads and remains a private build artifact.
"""
import argparse
from pathlib import Path
import struct
import zlib
from xess_native_bundle import load


def encode(payloads):
    if len(payloads)!=53:
        raise ValueError("Expected fourteen shaders and thirty-nine model buffers")
    output=bytearray(16+53*24)
    struct.pack_into('<8sII',output,0,b'XESSVK1\0',1,53)
    for i,data in enumerate(payloads):
        if not 0<len(data)<=4*1024*1024:
            raise ValueError("Invalid model entry size")
        output.extend(bytes((-len(output))%16))
        offset=len(output)
        output.extend(data)
        struct.pack_into('<QQII',output,16+i*24,offset,len(data),zlib.crc32(data),0)
    if len(output)>64*1024*1024:
        raise ValueError("Model container exceeds runtime bound")
    return bytes(output)


def pack(bundle,output):
    bundle=load(bundle)
    conv=bundle['convolutions'];resolve=bundle['resolve']
    if len(conv['passes'])!=12 or len(resolve['buffers'])!=3:
        raise ValueError("Unexpected graph shape")
    payloads=[Path(bundle['prepare']).read_bytes(),Path(resolve['shader']).read_bytes()]
    for stage in conv['passes']:
        if len(stage['buffers'])!=3:
            raise ValueError("Expected weight/scale/bias buffers")
        payloads.append(Path(stage['shader']).read_bytes())
        payloads.extend(Path(conv['buffers'][key]['file']).read_bytes() for key in stage['buffers'])
    payloads.extend(Path(buffer['file']).read_bytes() for buffer in resolve['buffers'])
    data=encode(payloads)
    with output.open('xb') as destination:
        destination.write(data)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    pack(args.bundle,args.output)
