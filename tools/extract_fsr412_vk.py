#!/usr/bin/env python3
"""
Extract and normalize Vulkan SPIR-V shaders and initializer weight blobs from
AMD's official WHQL-signed FSR 4.1.2 Vulkan driver provider DLL
(amdvkffx64_4.1.2_2.4.0.4370.dll, SHA-256 80e761f7c8c0ec8ead7b321df01911baaf917defae51388d29eaecb1d6bc58bd).

Supported FSR 4.1.2 model variants:
  - i8_2    (Fsr4Int8navi2UpscalerModel, "FSR4-i8-2"):
      Official RDNA2 / Navi2 INT8 27-pass model (17 tensors, 131,072-byte weights).
      Requires only DotProductInput4x8BitPacked + Float16 (no OpCapability Int8
      and no ComputeDerivativeGroupLinearKHR).
  - lite_i8 (Fsr4LiteInt8UpscalerModel, "FSR4L-i8"):
      Official FSR 4.1.2 Lite INT8 10-pass model (9 tensors, 27,392-byte weights).
      Does not require OpCapability Int8.
  - i8      (Fsr4Int8UpscalerModel, "FSR4-i8"):
      Official FSR 4.1.2 RDNA3 INT8 27-pass model (17 tensors, 131,072-byte weights).

Normalization steps applied to extracted SPIR-V modules:
  1. Decompress FXZF (8-byte header b"FXZF" + uint32 uncompressed_size + zlib stream).
  2. NormalizeHalfImageIO: promote OpTypeImage %half to %float with OpFConvert around
     image reads/samples and writes so every module passes spirv-val --target-env vulkan1.3
     (VUID-StandaloneSpirv-OpTypeImage-04656).
  3. Normalize OpEntryPoint name to "main" and remap descriptor bindings onto the
     compact ffx-vulkan FSR4 descriptor layout:
       - Full passes (pre, post, rcas, spd_auto_exposure):
           bindings 0..20  : SAMPLED_IMAGE (r_auto_exposure_texture 17 -> 6)
           bindings 21..32 : STORAGE_IMAGE (128+u -> 21+u; rw_rcas_output 11 -> 32)
           binding 32      : ScratchBuffer STORAGE_BUFFER (141 -> 32 in pre/post)
           binding 33      : InitializerBuffer STORAGE_BUFFER (21 -> 33 in pre)
           binding 35      : g_history_sampler SAMPLER (320 -> 35)
           binding 43      : Main UBO (MLSR_Optimized_Constants 257 / SPD 256 / RCAS 0 -> 43)
       - Model & border-padding passes (passes 1..25 for i8_2/i8, 1..8 for lite_i8):
           binding 1       : InitializerBuffer STORAGE_BUFFER (21 -> 1)
           binding 2       : ScratchBuffer STORAGE_BUFFER (141 -> 2)
           binding 4       : CsTensorSizes UNIFORM_BUFFER (256 -> 4)
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import struct
import subprocess
import sys
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

EXPECTED_DLL_SIZE = 72497568
EXPECTED_DLL_SHA256 = "80e761f7c8c0ec8ead7b321df01911baaf917defae51388d29eaecb1d6bc58bd"

PRESET_NAMES: Tuple[str, ...] = (
    "native",
    "quality",
    "balanced",
    "performance",
    "drs",
    "ultraperf",
)
TIER_NAMES: Tuple[str, ...] = ("1080", "2160", "4320")

# Exact file offsets and sizes in amdvkffx64_4.1.2_2.4.0.4370.dll.
# Presets 0..4 (native, quality, balanced, performance, drs) share the unified
# DRS-trained 4.1.2 weights and shader blobs ("default"); preset 5 (ultraperf)
# has its own weights and shader blobs ("ultraperf").
FSR412_CATALOG: Dict[str, dict] = {
    "i8_2": {
        "pass_count": 27,
        "initializer_size": 131072,
        "initializers": {
            "default": (68722144, 131072),
            "ultraperf": (68328912, 131072),
        },
        "shaders": {
            "default_1080": [
                (59571712, 48435), (66446464, 1189), (57973040, 18159), (59060112, 1201),
                (58343856, 18160), (61070752, 1196), (62450336, 5244),  (56331392, 1326),
                (62917200, 32129), (68063968, 1328), (62455680, 31995), (62650688, 1326),
                (64326416, 8718),  (62425168, 1466), (62419360, 5795),  (55535168, 1466),
                (65025616, 5782),  (60483328, 1466), (60510320, 7824),  (56491216, 1327),
                (57682080, 31530), (60117824, 1327), (55774944, 63011), (59111296, 1206),
                (60680256, 17989), (58362384, 1196), (55862256, 23554),
            ],
            "default_2160": [
                (64515296, 48434), (63839408, 1190), (63615184, 18160), (60652784, 1202),
                (64131584, 18161), (56071104, 1198), (56880464, 5245),  (60312992, 1327),
                (59276880, 32128), (59058704, 1326), (57031088, 31993), (59866448, 1327),
                (61494768, 8722),  (60654176, 1467), (63973696, 5795),  (62262384, 1467),
                (55971856, 5785),  (55773424, 1467), (56433568, 7827),  (60484928, 1325),
                (63485008, 31528), (67180384, 1328), (63840752, 63014), (66303344, 1210),
                (61233680, 17991), (64027872, 1198), (57388640, 23554),
            ],
            "default_4320": [
                (61122112, 48434), (59155152, 1193), (57168848, 18160), (57462896, 1205),
                (59526304, 18161), (60338208, 1200), (63684496, 5244),  (68065328, 1330),
                (64294272, 32131), (60062336, 1332), (64383888, 31996), (61732448, 1331),
                (67148000, 8723),  (56755072, 1473), (58242656, 5799),  (62870336, 1472),
                (59309056, 5787),  (66951664, 1473), (60565456, 7833),  (61095968, 1332),
                (67623904, 31531), (66179968, 1331), (63071120, 63012), (57257744, 1214),
                (57811376, 17991), (59694000, 1200), (62822176, 23554),
            ],
            "ultraperf_1080": [
                (58756912, 48435), (63633376, 1189), (65917264, 18214), (58889680, 1201),
                (56166992, 18186), (66255216, 1196), (61548704, 5545),  (63242000, 1326),
                (58675216, 32066), (65672480, 1328), (61251840, 31970), (66685152, 1326),
                (61677312, 8718),  (60895696, 1466), (59520400, 5795),  (63979504, 1466),
                (56427616, 5782),  (56279856, 1466), (67518640, 7824),  (66350608, 1327),
                (63807648, 31745), (63948768, 1327), (65467536, 63013), (66128512, 1206),
                (65703936, 18038), (65816224, 1196), (60605488, 23595),
            ],
            "ultraperf_2160": [
                (57258992, 48434), (67419600, 1190), (55953488, 18215), (56705824, 1202),
                (59362272, 18187), (61651504, 1198), (62341712, 5547),  (63613824, 1327),
                (63163872, 32066), (61399360, 1326), (60573360, 31970), (58838464, 1327),
                (58805360, 8722),  (63565104, 1467), (64463152, 5795),  (57331120, 1467),
                (65673840, 5785),  (56753552, 1467), (58363680, 7827),  (56329984, 1325),
                (64635376, 31744), (61069376, 1328), (61170576, 63015), (60897184, 1210),
                (55935120, 18038), (64833328, 1198), (67156736, 23595),
            ],
            "ultraperf_4320": [
                (66686736, 48434), (60698304, 1193), (59136912, 18216), (59569072, 1205),
                (62537232, 18187), (59570336, 1200), (63134144, 5545),  (65080288, 1330),
                (60699584, 32069), (59964864, 1332), (58160816, 31972), (62338768, 1331),
                (57159808, 8723),  (59966448, 1473), (59860560, 5799),  (60996576, 1472),
                (57382720, 5787),  (57413872, 1473), (60063744, 7833),  (62340128, 1332),
                (64099488, 31747), (61804896, 1331), (58419904, 63012), (57412208, 1214),
                (59502208, 18040), (62973984, 1200), (62264336, 23595),
            ],
        },
    },
    "lite_i8": {
        "pass_count": 10,
        "initializer_size": 27392,
        "initializers": {
            "default": (55370736, 27392),
            "ultraperf": (55480352, 27392),
        },
        "shaders": {
            "default_1080": [
                (50199072, 14580), (54927104, 1190), (52333232, 26867), (52173328, 1202),
                (53910288, 1328),  (53911680, 31885), (52413904, 1315), (54779984, 9702),
                (53112416, 1210),  (55183200, 24604),
            ],
            "default_2160": [
                (54096288, 14581), (50137376, 1192), (54942528, 26869), (53217376, 1204),
                (52101888, 1329),  (55338352, 31883), (54069856, 1314), (51816896, 9706),
                (50635920, 1211),  (53798368, 24604),
            ],
            "default_4320": [
                (50440768, 14581), (54693792, 1195), (53316736, 26870), (50182816, 1207),
                (51815520, 1335),  (52986496, 31886), (51702544, 1317), (50228992, 9708),
                (52103264, 1217),  (52907568, 24605),
            ],
            "ultraperf_1080": [
                (50184368, 14580), (51088352, 1190), (54727952, 26877), (55312704, 1202),
                (50555968, 1328),  (50509296, 31851), (54724688, 1315), (54917312, 9717),
                (52457504, 1210),  (52660960, 24723),
            ],
            "ultraperf_2160": [
                (51118928, 14581), (51574816, 1192), (51061312, 26882), (53771024, 1204),
                (52525536, 1329),  (53863216, 31849), (50927264, 1314), (54515792, 9723),
                (55142432, 1211),  (54382736, 24725),
            ],
            "ultraperf_4320": [
                (51089600, 14581), (50557520, 1195), (54969536, 26885), (51172400, 1207),
                (51732832, 1335),  (52755456, 31852), (50953024, 1317), (53462720, 9724),
                (54726400, 1217),  (50470544, 24723),
            ],
        },
    },
    "i8": {
        "pass_count": 27,
        "initializer_size": 131072,
        "initializers": {
            "default": (49356496, 131072),
            "ultraperf": (49225392, 131072),
        },
        "shaders": {
            "default_1080": [
                (44168640, 14163), (45157040, 1190), (47907024, 16044), (46039760, 1203),
                (48333616, 16047), (47252752, 1198), (46944416, 4386),  (47435920, 1327),
                (43332896, 28405), (48127712, 1329), (48004480, 28237), (43848784, 1327),
                (46276944, 7012),  (45943440, 1467), (45171712, 5379),  (43958304, 1466),
                (47954176, 5377),  (45373408, 1466), (47086896, 7224),  (47030320, 1328),
                (47541440, 27782), (43537584, 1327), (45555408, 7065),  (46275680, 1208),
                (45647696, 15877), (45482960, 1198), (46509072, 21282),
            ],
            "default_2160": [
                (47327008, 14163), (46340800, 1191), (45927360, 16044), (43818736, 1203),
                (45374880, 16049), (48917120, 1200), (47746544, 4395),  (44580976, 1327),
                (46822464, 28404), (47952672, 1327), (46430064, 28236), (48002480, 1327),
                (43996448, 7018),  (44116352, 1467), (44197152, 5379),  (46948944, 1467),
                (45511792, 5375),  (44520544, 1467), (47689408, 7228),  (47031664, 1326),
                (43700368, 27782), (48679920, 1328), (46623520, 7068),  (47085648, 1212),
                (48038160, 15881), (47289888, 1200), (48809440, 21282),
            ],
            "default_4320": [
                (48591376, 14164), (48310384, 1194), (47583328, 16045), (46075392, 1207),
                (46964512, 16048), (45325056, 1202), (43533104, 4394),  (47764928, 1331),
                (44003616, 28408), (45039664, 1333), (45946368, 28239), (44864224, 1331),
                (44357568, 7020),  (47524896, 1473), (46530608, 5379),  (48620224, 1472),
                (48032768, 5379),  (46942784, 1473), (44719872, 7230),  (48054224, 1333),
                (43405248, 27784), (47629728, 1332), (43914896, 7067),  (47923152, 1216),
                (47891056, 15878), (46414000, 1202), (48349680, 21282),
            ],
            "ultraperf_1080": [
                (46458304, 14163), (44519328, 1190), (48636096, 16100), (44283536, 1203),
                (44466512, 16079), (43531840, 1198), (43461328, 4705),  (43641184, 1327),
                (45440576, 28339), (43638256, 1329), (46712848, 28228), (46993888, 1327),
                (44202640, 7012),  (44364592, 1467), (43455760, 5379),  (46258064, 1466),
                (48397184, 5377),  (45745648, 1466), (47883808, 7224),  (45945040, 1328),
                (46298816, 28004), (44675648, 1327), (47078560, 7073),  (47487712, 1208),
                (48509920, 15923), (46259648, 1198), (48092656, 21334),
            ],
            "ultraperf_2160": [
                (44975808, 14163), (47077344, 1191), (45795744, 16102), (45926112, 1203),
                (44341456, 16078), (47631104, 1200), (45368688, 4706),  (46219632, 1327),
                (44552464, 28339), (47628224, 1327), (46118832, 28228), (44865680, 1327),
                (46995632, 7018),  (45141136, 1467), (45484256, 5379),  (47236832, 1467),
                (45562640, 5375),  (46850880, 1467), (46691728, 7228),  (46660752, 1326),
                (48717840, 28003), (45517200, 1328), (46986784, 7075),  (48091376, 1212),
                (46644656, 15925), (43995184, 1200), (47291088, 21333),
            ],
            "ultraperf_4320": [
                (48143312, 14164), (47882496, 1194), (44522128, 16101), (45707968, 1207),
                (46755328, 16077), (45532544, 1202), (46980640, 4706),  (44223360, 1331),
                (45296704, 28343), (47128064, 1333), (45055344, 28230), (44246944, 1331),
                (46054832, 7020),  (44953840, 1473), (43361600, 5379),  (43639696, 1472),
                (46937264, 5379),  (44674160, 1473), (45504304, 7230),  (46985408, 1333),
                (45825824, 28007), (43389440, 1332), (44968576, 7075),  (46580080, 1216),
                (43488080, 15923), (46820768, 1202), (47268528, 21334),
            ],
        },
    },
    "rcas": (43288816, 5109),
    "spd_auto_exposure": (43309584, 8574),
}

SPV_MAGIC = 0x07230203
SPV_OP_NAME = 5
SPV_OP_ENTRY_POINT = 15
SPV_OP_TYPE_FLOAT = 22
SPV_OP_TYPE_VECTOR = 23
SPV_OP_TYPE_IMAGE = 25
SPV_OP_TYPE_SAMPLED_IMAGE = 27
SPV_OP_TYPE_POINTER = 32
SPV_OP_TYPE_FUNCTION = 33
SPV_OP_CONSTANT = 43
SPV_OP_FUNCTION = 54
SPV_OP_DECORATE = 71
SPV_OP_COMPOSITE_CONSTRUCT = 80
SPV_OP_COMPOSITE_EXTRACT = 81
SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD = 87
SPV_OP_IMAGE_READ = 98
SPV_OP_IMAGE_WRITE = 99
SPV_OP_FCONVERT = 115
SPV_DECORATION_BINDING = 33

NO_RESULT_TYPE = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    48, 55, 56, 60, 62, 63, 71, 72, 73, 74, 75, 99, 224, 225, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255,
}


def decompress_fxzf(blob: bytes) -> bytes:
    if len(blob) >= 8 and blob[:4] == b"FXZF":
        expected_size = struct.unpack_from("<I", blob, 4)[0]
        raw = zlib.decompress(blob[8:])
        if len(raw) != expected_size:
            raise ValueError(f"FXZF size mismatch: got {len(raw)}, expected {expected_size}")
        return raw
    return blob


def normalize_half_image_io(spv_bytes: bytes) -> bytes:
    if len(spv_bytes) < 20 or len(spv_bytes) % 4 != 0:
        raise ValueError("Invalid SPIR-V byte length")
    words = list(struct.unpack(f"<{len(spv_bytes) // 4}I", spv_bytes))
    if words[0] != SPV_MAGIC:
        raise ValueError("Invalid SPIR-V magic")

    insts: List[List[int]] = []
    off = 5
    while off < len(words):
        cnt = words[off] >> 16
        if cnt == 0 or off + cnt > len(words):
            raise ValueError("Truncated SPIR-V instruction")
        insts.append(words[off : off + cnt])
        off += cnt

    half_types = set()
    float_type = 0
    for ins in insts:
        if (ins[0] & 0xFFFF) == SPV_OP_TYPE_FLOAT and len(ins) == 3:
            if ins[2] == 16:
                half_types.add(ins[1])
            elif ins[2] == 32:
                float_type = ins[1]

    has_half_img = any(
        (ins[0] & 0xFFFF) == SPV_OP_TYPE_IMAGE and len(ins) >= 9 and ins[2] in half_types
        for ins in insts
    )
    if not has_half_img or float_type == 0:
        return spv_bytes

    next_id = words[3]
    decls: List[List[int]] = []
    promoted_types = {ht: float_type for ht in half_types}
    float_vec_types: Dict[int, int] = {}
    for ins in insts:
        if (ins[0] & 0xFFFF) == SPV_OP_TYPE_VECTOR and len(ins) == 4 and ins[2] == float_type:
            float_vec_types.setdefault(ins[3], ins[1])
    for ins in insts:
        if (ins[0] & 0xFFFF) == SPV_OP_TYPE_VECTOR and len(ins) == 4 and ins[2] in half_types:
            if ins[3] in float_vec_types:
                promoted = float_vec_types[ins[3]]
            else:
                promoted = next_id
                next_id += 1
                decls.append([(4 << 16) | SPV_OP_TYPE_VECTOR, promoted, float_type, ins[3]])
            promoted_types[ins[1]] = promoted

    half_value_types: Dict[int, int] = {}
    for ins in insts:
        op = ins[0] & 0xFFFF
        if op not in NO_RESULT_TYPE and len(ins) >= 3 and ins[1] in promoted_types:
            half_value_types[ins[2]] = ins[1]

    aliases: Dict[int, int] = {}
    canonical: List[List[int]] = []

    def resolve(tid: int) -> int:
        return aliases.get(tid, tid)

    for ins in insts:
        op = ins[0] & 0xFFFF
        if op == SPV_OP_TYPE_IMAGE and len(ins) >= 9:
            if ins[2] in half_types:
                ins[2] = float_type
        elif op == SPV_OP_TYPE_SAMPLED_IMAGE and len(ins) == 3:
            ins[2] = resolve(ins[2])
        elif op == SPV_OP_TYPE_POINTER and len(ins) == 4:
            ins[3] = resolve(ins[3])
        else:
            continue
        found = None
        for cand in canonical:
            if cand[0] == ins[0] and cand[2:] == ins[2:]:
                found = cand[1]
                break
        if found is not None:
            aliases[ins[1]] = found
        else:
            canonical.append(ins)

    for ins in insts:
        op = ins[0] & 0xFFFF
        if op not in NO_RESULT_TYPE and len(ins) >= 3:
            ins[1] = resolve(ins[1])
        if op in (SPV_OP_NAME, SPV_OP_DECORATE) and len(ins) >= 3:
            ins[1] = resolve(ins[1])
        if op == SPV_OP_TYPE_FUNCTION:
            for i in range(2, len(ins)):
                ins[i] = resolve(ins[i])

    out = list(words[:5])
    decls_added = False
    for ins in insts:
        op = ins[0] & 0xFFFF
        if op in (SPV_OP_TYPE_IMAGE, SPV_OP_TYPE_SAMPLED_IMAGE, SPV_OP_TYPE_POINTER) and len(ins) >= 2 and ins[1] in aliases:
            continue
        if op == SPV_OP_FUNCTION and not decls_added:
            for d in decls:
                out.extend(d)
            decls_added = True
        if SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD <= op <= SPV_OP_IMAGE_READ and len(ins) >= 4 and ins[1] in promoted_types:
            tmp = next_id
            next_id += 1
            new_ins = list(ins)
            new_ins[1] = promoted_types[ins[1]]
            new_ins[2] = tmp
            out.extend(new_ins)
            out.extend([(4 << 16) | SPV_OP_FCONVERT, ins[1], ins[2], tmp])
            continue
        if op == SPV_OP_IMAGE_WRITE and len(ins) >= 4 and ins[3] in half_value_types:
            tmp = next_id
            next_id += 1
            ht = half_value_types[ins[3]]
            out.extend([(4 << 16) | SPV_OP_FCONVERT, promoted_types[ht], tmp, ins[3]])
            new_ins = list(ins)
            new_ins[3] = tmp
            out.extend(new_ins)
            continue
        out.extend(ins)
    out[3] = next_id
    return struct.pack(f"<{len(out)}I", *out)


def promote_image_write_texels_to_vec4(spv_bytes: bytes) -> bytes:
    if len(spv_bytes) < 20 or len(spv_bytes) % 4 != 0:
        return spv_bytes
    words = list(struct.unpack(f"<{len(spv_bytes) // 4}I", spv_bytes))
    if words[0] != SPV_MAGIC:
        return spv_bytes

    insts: List[List[int]] = []
    off = 5
    while off < len(words):
        cnt = words[off] >> 16
        if cnt == 0 or off + cnt > len(words):
            return spv_bytes
        insts.append(words[off : off + cnt])
        off += cnt

    has_image_write = any((ins[0] & 0xFFFF) == SPV_OP_IMAGE_WRITE for ins in insts)
    if not has_image_write:
        return spv_bytes

    float_type = next((ins[1] for ins in insts if (ins[0] & 0xFFFF) == SPV_OP_TYPE_FLOAT and ins[2] == 32), None)
    if float_type is None:
        return spv_bytes

    v3float_type = next((ins[1] for ins in insts if (ins[0] & 0xFFFF) == SPV_OP_TYPE_VECTOR and ins[2] == float_type and ins[3] == 3), None)
    if v3float_type is None:
        return spv_bytes

    next_id = words[3]
    decls: List[List[int]] = []

    v4float_type = next((ins[1] for ins in insts if (ins[0] & 0xFFFF) == SPV_OP_TYPE_VECTOR and ins[2] == float_type and ins[3] == 4), None)
    if v4float_type is None:
        v4float_type = next_id
        next_id += 1
        decls.append([(4 << 16) | SPV_OP_TYPE_VECTOR, v4float_type, float_type, 4])

    float_1_val = struct.unpack("<I", struct.pack("<f", 1.0))[0]
    const_1 = next((ins[2] for ins in insts if (ins[0] & 0xFFFF) == SPV_OP_CONSTANT and ins[1] == float_type and ins[3] == float_1_val), None)
    if const_1 is None:
        const_1 = next_id
        next_id += 1
        decls.append([(4 << 16) | SPV_OP_CONSTANT, float_type, const_1, float_1_val])

    val_types: Dict[int, int] = {}
    for ins in insts:
        op = ins[0] & 0xFFFF
        if op not in NO_RESULT_TYPE and len(ins) >= 3:
            val_types[ins[2]] = ins[1]

    out = list(words[:5])
    decls_added = False
    for ins in insts:
        op = ins[0] & 0xFFFF
        if op == SPV_OP_FUNCTION and not decls_added:
            for d in decls:
                out.extend(d)
            decls_added = True

        if op == SPV_OP_IMAGE_WRITE and len(ins) >= 4:
            texel_id = ins[3]
            if val_types.get(texel_id) == v3float_type:
                tmp_x = next_id
                tmp_y = next_id + 1
                tmp_z = next_id + 2
                tmp_v4 = next_id + 3
                next_id += 4
                out.extend([(5 << 16) | SPV_OP_COMPOSITE_EXTRACT, float_type, tmp_x, texel_id, 0])
                out.extend([(5 << 16) | SPV_OP_COMPOSITE_EXTRACT, float_type, tmp_y, texel_id, 1])
                out.extend([(5 << 16) | SPV_OP_COMPOSITE_EXTRACT, float_type, tmp_z, texel_id, 2])
                out.extend([(7 << 16) | SPV_OP_COMPOSITE_CONSTRUCT, v4float_type, tmp_v4, tmp_x, tmp_y, tmp_z, const_1])
                new_ins = list(ins)
                new_ins[3] = tmp_v4
                out.extend(new_ins)
                continue

        out.extend(ins)

    out[3] = next_id
    return struct.pack(f"<{len(out)}I", *out)


def normalize_entry_point_and_bindings(spv_bytes: bytes, pass_kind: str) -> bytes:
    """
    pass_kind:
      - 'pre'   : FSR 4.1.2 prepass (pass 0)
      - 'model' : FSR 4.1.2 inner model or post-padding pass (passes 1..N-2)
      - 'post'  : FSR 4.1.2 postpass (pass N-1)
      - 'rcas'  : FSR 4.1.2 RCAS pass
      - 'spd'   : FSR 4.1.2 SPD AutoExposure pass
    """
    words = list(struct.unpack(f"<{len(spv_bytes) // 4}I", spv_bytes))
    out = list(words[:5])
    off = 5
    while off < len(words):
        cnt = words[off] >> 16
        op = words[off] & 0xFFFF
        ins = list(words[off : off + cnt])
        off += cnt

        if op == SPV_OP_ENTRY_POINT and len(ins) >= 4:
            # OpEntryPoint: [word0, exec_model, entry_id, name_words..., interface_ids...]
            # Find the end of the NUL-terminated literal string starting at ins[3]
            str_end = 3
            while str_end < len(ins):
                w = ins[str_end]
                str_end += 1
                if (w & 0xFF) == 0 or (w & 0xFF00) == 0 or (w & 0xFF0000) == 0 or (w & 0xFF000000) == 0:
                    break
            interface_ids = ins[str_end:]
            # Encode "main\0\0\0\0" (2 words: 0x6e69616d, 0x00000000)
            main_words = [0x6E69616D, 0x00000000]
            new_ins = [0, ins[1], ins[2]] + main_words + interface_ids
            new_ins[0] = (len(new_ins) << 16) | SPV_OP_ENTRY_POINT
            out.extend(new_ins)
            continue

        if op == SPV_OP_DECORATE and len(ins) == 4 and ins[2] == SPV_DECORATION_BINDING:
            b = ins[3]
            if pass_kind == "model":
                if b == 21:
                    b = 1  # InitializerBuffer -> srvBuffers[1]
                elif b == 141:
                    b = 2  # ScratchBuffer -> uavBuffers[0]
                elif b == 256:
                    b = 4  # CsTensorSizes -> cbs[0]
            elif pass_kind in ("pre", "post"):
                if b == 17:
                    b = 6  # r_auto_exposure_texture -> t6
                elif b == 21:
                    b = 33  # InitializerBuffer -> SLOT_SRV_INITIALIZER (33)
                elif b == 141:
                    b = 32  # ScratchBuffer -> SLOT_SCRATCH (32)
                elif 128 <= b <= 140:
                    b = (b - 128) + 21  # u0..u12 -> 21..33
                elif b in (256, 257):
                    b = 43  # MLSR_Optimized_Constants -> SLOT_CBUF_MAIN (43)
                elif b == 320:
                    b = 35  # g_history_sampler -> SLOT_SAMPLER (35)
            elif pass_kind == "spd":
                if 128 <= b <= 140:
                    b = (b - 128) + 21
                elif b == 256:
                    b = 43
            elif pass_kind == "rcas":
                if b == 17:
                    b = 6  # r_auto_exposure_texture -> t6
                elif b == 11:
                    b = 32  # rw_rcas_output -> u11 (21 + 11 = 32)
                elif b == 0:
                    b = 43  # cbRCAS -> SLOT_CBUF_MAIN (43)
            ins[3] = b

        out.extend(ins)
    return struct.pack(f"<{len(out)}I", *out)


def extract_fsr412_bundle(
    dll_path: pathlib.Path,
    out_dir: pathlib.Path,
    variants: Sequence[str] = ("i8_2", "lite_i8", "i8"),
    validate: bool = True,
) -> Dict[str, int]:
    dll_bytes = dll_path.read_bytes()
    if len(dll_bytes) != EXPECTED_DLL_SIZE:
        raise ValueError(f"Unexpected DLL size {len(dll_bytes)} (expected {EXPECTED_DLL_SIZE})")
    digest = hashlib.sha256(dll_bytes).hexdigest()
    if digest != EXPECTED_DLL_SHA256:
        raise ValueError(f"Unexpected DLL SHA-256 {digest} (expected {EXPECTED_DLL_SHA256})")

    out_dir.mkdir(parents=True, exist_ok=True)
    spirv_val = shutil.which("spirv-val") if validate else None

    stats = {"shaders": 0, "initializers": 0, "validated": 0}
    # Cache normalized SPIR-V by (off, size, pass_kind) to avoid redundant work
    spv_cache: Dict[Tuple[int, int, str], bytes] = {}

    def get_normalized_spv(off: int, size: int, pass_kind: str) -> bytes:
        key = (off, size, pass_kind)
        if key not in spv_cache:
            raw = decompress_fxzf(dll_bytes[off : off + size])
            norm = normalize_half_image_io(raw)
            norm = promote_image_write_texels_to_vec4(norm)
            norm = normalize_entry_point_and_bindings(norm, pass_kind)
            if spirv_val:
                proc = subprocess.run(
                    [spirv_val, "--target-env", "vulkan1.3", "-"],
                    input=norm,
                    capture_output=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"spirv-val failed for offset {off} ({pass_kind}): {proc.stderr.decode()}"
                    )
                stats["validated"] += 1
            spv_cache[key] = norm
        return spv_cache[key]

    for variant in variants:
        vinfo = FSR412_CATALOG[variant]
        n_passes = vinfo["pass_count"]
        for preset in PRESET_NAMES:
            src_key = "ultraperf" if preset == "ultraperf" else "default"
            ioff, isize = vinfo["initializers"][src_key]
            init_path = out_dir / f"fsr4_model_412_{variant}_{preset}_initializers.bin"
            init_path.write_bytes(dll_bytes[ioff : ioff + isize])
            stats["initializers"] += 1

            for tier in TIER_NAMES:
                passes = vinfo["shaders"][f"{src_key}_{tier}"]
                assert len(passes) == n_passes
                for pid, (poff, psize) in enumerate(passes):
                    if pid == 0:
                        pkind = "pre"
                        fname = f"fsr4_model_412_{variant}_{preset}_{tier}_pre.spv"
                    elif pid == n_passes - 1:
                        pkind = "post"
                        fname = f"fsr4_model_412_{variant}_{preset}_{tier}_post.spv"
                    else:
                        pkind = "model"
                        fname = f"fsr4_model_412_{variant}_{preset}_{tier}_pass{pid}.spv"
                    spv_data = get_normalized_spv(poff, psize, pkind)
                    (out_dir / fname).write_bytes(spv_data)
                    stats["shaders"] += 1

    # Also write rcas_412.spv and spd_auto_exposure_412.spv (plus rcas.spv and spd_auto_exposure.spv if missing)
    rcas_off, rcas_size = FSR412_CATALOG["rcas"]
    spd_off, spd_size = FSR412_CATALOG["spd_auto_exposure"]
    rcas_spv = get_normalized_spv(rcas_off, rcas_size, "rcas")
    spd_spv = get_normalized_spv(spd_off, spd_size, "spd")
    (out_dir / "rcas_412.spv").write_bytes(rcas_spv)
    (out_dir / "spd_auto_exposure_412.spv").write_bytes(spd_spv)
    if not (out_dir / "rcas.spv").exists():
        (out_dir / "rcas.spv").write_bytes(rcas_spv)
    if not (out_dir / "spd_auto_exposure.spv").exists():
        (out_dir / "spd_auto_exposure.spv").write_bytes(spd_spv)
    stats["shaders"] += 2

    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dll",
        type=pathlib.Path,
        default=pathlib.Path("build/vendor-sdks/fsr-4.1.2/amdvkffx64_4.1.2_2.4.0.4370.dll"),
        help="Path to amdvkffx64_4.1.2_2.4.0.4370.dll",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("baseq2/fsr4_shaders"),
        help="Output directory for extracted SPIR-V and initializer assets",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("i8_2", "lite_i8", "i8"),
        default=["i8_2", "lite_i8", "i8"],
        help="FSR 4.1.2 model variants to extract",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip spirv-val --target-env vulkan1.3 validation",
    )
    args = parser.parse_args(argv)
    stats = extract_fsr412_bundle(
        args.dll,
        args.out,
        variants=args.variants,
        validate=not args.no_validate,
    )
    print(
        f"Extracted {stats['shaders']} SPIR-V files and {stats['initializers']} "
        f"initializer blobs into {args.out} ({stats['validated']} unique SPIR-V "
        f"modules validated with spirv-val --target-env vulkan1.3)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
