# SPDX-License-Identifier: BSD-2-Clause
"""Compare OpenCL Tier-1 coefficients against captured OpenJPEG CPU output."""
import argparse
import ctypes as ct
import json
from pathlib import Path

from generate_mq_tables import generate
from test_idwt53 import OpenCL, KernelProgram, HANDLE, UINT, INT, SIZE


def load_blocks(path):
    blocks = [json.loads(line) for line in path.read_text().splitlines()]
    if not blocks:
        raise ValueError("Empty corpus")
    for b in blocks:
        if b["version"] != 2 or b.get("stage") != "post_roi" or b["corrupted"] or b["style"] & ~63:
            raise ValueError("Unsupported capture: expected post-ROI Part 1 reference")
        if not (0 <= b["roi"] <= 30 and b["numbps"] + b["roi"] <= 30):
            raise ValueError("Invalid ROI bitplane range")
        if b["check_pterm"] not in (0, 1) or b["pterm_status"] not in (0, 512, 1024):
            raise ValueError("Invalid PTERM metadata")
        if b["check_pterm"] and not b["style"] & 16:
            raise ValueError("PTERM check without coding flag")
        if not b["check_pterm"] and b["pterm_status"]:
            raise ValueError("PTERM diagnostic without a check")
        w, h = b["width"], b["height"]
        if not (0 < w <= 1024 and 0 < h <= 1024 and w*h <= 4096):
            raise ValueError("Invalid block geometry")
        if not (0 <= b["orientation"] <= 3 and 0 <= b["numbps"] <= 30):
            raise ValueError("Invalid block coding parameters")
        if len(b["coefficients"]) != w*h:
            raise ValueError("Incorrect reference coefficient count")
        b["payload"] = bytes.fromhex(b["bytes"])
        if len(b["payload"]) > 65536 or len(b["segments"]) > 90:
            raise ValueError("Unbounded block input")
        if any(not (0 <= n <= 65536 and 0 < p <= 90) for n, p in b["segments"]):
            raise ValueError("Invalid segment")
        if sum(s[0] for s in b["segments"]) != len(b["payload"]):
            raise ValueError("Segment lengths do not cover payload")
        if sum(s[1] for s in b["segments"]) > max(0, 3*(b["numbps"]+b["roi"])-2):
            raise ValueError("Excess coding passes")
        # RAW and MQ coding cannot share one arithmetic segment. An incomplete
        # quality layer may end a segment early, but not change its coding mode.
        bp, coding_pass = b["numbps"]+b["roi"], 2
        for _, passes in b["segments"]:
            raw = bool(b["style"] & 1 and bp <= b["numbps"]-4 and coding_pass < 2)
            for _ in range(passes):
                mode = bool(b["style"] & 1 and bp <= b["numbps"]-4 and coding_pass < 2)
                if raw != mode:
                    raise ValueError("Mixed RAW and MQ passes in one segment")
                coding_pass += 1
                if coding_pass == 3:
                    coding_pass = 0
                    bp -= 1
    return blocks


def verify_batch(program, blocks):
    cl = program.cl
    desc, segments, reference = [], [], []
    payload = bytearray()
    for b in blocks:
        desc.extend((b["width"], b["height"], b["orientation"], b["numbps"], b["style"],
                     len(payload), len(b["payload"]), len(segments)//2, len(b["segments"]), len(reference),
                     b["roi"], b["check_pterm"]))
        for segment in b["segments"]:
            segments.extend(segment)
        payload.extend(b["payload"])
        reference.extend(b["coefficients"])
    # Distinct output/scratch sentinels expose unwritten values.
    arrays = [(UINT * len(desc))(*desc),
              (UINT * max(1, len(segments)))(*segments),
              (ct.c_ubyte * max(1, len(payload)))(*payload),
              (INT * len(reference))(*([123456789] * len(reference))),
              (ct.c_ubyte * len(reference))(*([255] * len(reference))),
              (UINT * len(blocks))(*([99] * len(blocks)))]
    buffers = []
    try:
        for data in arrays:
            error = INT()
            handle = cl.CreateBuffer(program.context, 1 | 32, ct.sizeof(data), data, ct.byref(error))
            if handle:
                buffers.append(handle)
            cl.check(error.value, "create Tier-1 buffer")
            if not handle:
                raise RuntimeError("Null Tier-1 buffer")
        kernel = program.kernels[0]
        args = [HANDLE(h) for h in buffers] + [UINT(len(blocks))]
        for index, value in enumerate(args):
            cl.check(cl.SetKernelArg(kernel, index, ct.sizeof(value), ct.byref(value)), "Tier-1 argument")
        shape = (SIZE * 1)(len(blocks) + 3)
        cl.check(cl.EnqueueNDRangeKernel(program.queue, kernel, 1, None, shape, None, 0, None, None), "Tier-1 dispatch")
        for index in (3, 5):
            data = arrays[index]
            cl.check(cl.EnqueueReadBuffer(program.queue, buffers[index], 1, 0, ct.sizeof(data), data, 0, None, None), "Tier-1 readback")
        if any(v & 255 for v in arrays[5]):
            failures = [(i, v) for i, v in enumerate(arrays[5]) if v & 255]
            raise AssertionError(f"Kernel status errors: {failures[:10]}")
        program.segsym_diagnostics = getattr(program, "segsym_diagnostics", 0) + sum(bool(v & 256) for v in arrays[5])
        offset = 0
        for index, b in enumerate(blocks):
            if arrays[5][index] & 1536 != b["pterm_status"]:
                raise AssertionError(f"Block {index}: GPU PTERM {arrays[5][index] & 1536} != CPU {b['pterm_status']}")
            expected = b["coefficients"]
            actual = list(arrays[3][offset:offset+len(expected)])
            if actual != expected:
                i = next(i for i, (a, e) in enumerate(zip(actual, expected)) if a != e)
                metadata = {k: v for k, v in b.items() if k not in ("bytes", "payload", "coefficients")}
                raise AssertionError(f"Block {index}, coefficient {i}: GPU={actual[i]}, CPU={expected[i]}; {metadata}")
            offset += len(expected)
        return len(reference)
    finally:
        cl.Finish(program.queue)
        for buffer in reversed(buffers):
            cl.ReleaseMemObject(buffer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--driver")
    args = parser.parse_args()
    blocks = load_blocks(args.corpus)
    cl = OpenCL()
    devices = [(d, n) for d, n in cl.devices() if args.device.lower() in n.lower()
               and (not args.driver or args.driver in cl.device_text(d, 0x102D))]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one matching GPU, found {len(devices)}")
    print(f"Tier-1 GPU: {devices[0][1]}; driver {cl.device_text(devices[0][0], 0x102D)}")
    directory = Path(__file__).resolve().parent
    if (directory / "mq_states.clh").read_text() != generate():
        raise RuntimeError("MQ table is stale; regenerate from upstream")
    source = (directory / "mq_states.clh").read_bytes() + (directory / "t1_decode.cl").read_bytes()
    program = KernelProgram(cl, devices[0][0], source, (b"decode_blocks",))
    count = 0
    try:
        for start in range(0, len(blocks), 128):
            count += verify_batch(program, blocks[start:start+128])
    finally:
        program.close()
    print(f"PASS: {len(blocks)} Tier-1 blocks / {count} coefficients exactly match OpenJPEG")
    raw_segments = 0
    for b in blocks:
        bp, coding_pass = b["numbps"]+b["roi"], 2
        for _, passes in b["segments"]:
            raw_segments += bool(b["style"] & 1 and bp <= b["numbps"]-4 and coding_pass < 2)
            bp -= (coding_pass + passes)//3
            coding_pass = (coding_pass + passes)%3
    print(f"Coverage: {len(set(b['style'] for b in blocks))} styles, {raw_segments} RAW segments, "
          f"{sum(bool(b['style'] & 8) for b in blocks)} VSC blocks")
    print(f"Nonfatal segmentation-symbol diagnostics: {program.segsym_diagnostics}")
    print(f"ROI blocks: {sum(bool(b['roi']) for b in blocks)}; "
          f"PTERM checks: {sum(b['check_pterm'] for b in blocks)}; "
          f"CPU-matched PTERM diagnostics: {sum(bool(b['pterm_status']) for b in blocks)}")


if __name__ == "__main__":
    main()
