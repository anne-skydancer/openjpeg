# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, OpenJPEG OpenCL contributors
"""Execute the experimental kernel on an explicitly selected OpenCL GPU.

Standard-library-only development runner; ctypes is not the production backend.
Uses the system ICD, an in-order queue and independent forward-transform input.
No viewer launch, package replacement or vendor-specific extensions.
"""
import argparse
import ctypes as ct
import ctypes.util
from pathlib import Path
import random
import sys

INT = ct.c_int32
UINT = ct.c_uint32
BITS = ct.c_uint64
SIZE = ct.c_size_t
HANDLE = ct.c_void_p
P = ct.POINTER


class OpenCL:
    def __init__(self):
        if sys.platform == "win32":
            self.lib = ct.WinDLL("OpenCL.dll")
        else:
            name = ctypes.util.find_library("OpenCL")
            if not name:
                raise RuntimeError("OpenCL runtime not found")
            self.lib = ct.CDLL(name)
        signatures = {
            "GetPlatformIDs": (INT, [UINT, P(HANDLE), P(UINT)]),
            "GetDeviceIDs": (INT, [HANDLE, BITS, UINT, P(HANDLE), P(UINT)]),
            "GetDeviceInfo": (INT, [HANDLE, UINT, SIZE, HANDLE, P(SIZE)]),
            "CreateContext": (HANDLE, [HANDLE, UINT, P(HANDLE), HANDLE, HANDLE, P(INT)]),
            "CreateCommandQueue": (HANDLE, [HANDLE, HANDLE, BITS, P(INT)]),
            "CreateProgramWithSource": (HANDLE, [HANDLE, UINT, P(ct.c_char_p), P(SIZE), P(INT)]),
            "BuildProgram": (INT, [HANDLE, UINT, P(HANDLE), ct.c_char_p, HANDLE, HANDLE]),
            "GetProgramBuildInfo": (INT, [HANDLE, HANDLE, UINT, SIZE, HANDLE, P(SIZE)]),
            "CreateKernel": (HANDLE, [HANDLE, ct.c_char_p, P(INT)]),
            "CreateBuffer": (HANDLE, [HANDLE, BITS, SIZE, HANDLE, P(INT)]),
            "SetKernelArg": (INT, [HANDLE, UINT, SIZE, HANDLE]),
            "EnqueueNDRangeKernel": (INT, [HANDLE, HANDLE, UINT, P(SIZE), P(SIZE), P(SIZE), UINT, P(HANDLE), P(HANDLE)]),
            "EnqueueReadBuffer": (INT, [HANDLE, HANDLE, UINT, SIZE, SIZE, HANDLE, UINT, P(HANDLE), P(HANDLE)]),
            "Finish": (INT, [HANDLE]),
        }
        for kind in ("Context", "CommandQueue", "Program", "Kernel", "MemObject"):
            signatures["Release" + kind] = (INT, [HANDLE])
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, "cl" + name)
            function.restype = result
            function.argtypes = args
            setattr(self, name, function)

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"{operation}: OpenCL error {code}")

    def devices(self):
        count = UINT()
        self.check(self.GetPlatformIDs(0, None, ct.byref(count)), "platform count")
        platforms = (HANDLE * count.value)()
        self.check(self.GetPlatformIDs(count, platforms, None), "platform enumeration")
        devices = []
        for platform in platforms:
            code = self.GetDeviceIDs(platform, 4, 0, None, ct.byref(count))  # GPU
            if code == -1:  # CL_DEVICE_NOT_FOUND
                continue
            self.check(code, "GPU count")
            handles = (HANDLE * count.value)()
            self.check(self.GetDeviceIDs(platform, 4, count, handles, None), "GPU enumeration")
            for handle in handles:
                devices.append((handle, self.device_text(handle, 0x102B)))
        return devices

    def device_text(self, device, field):
        length = SIZE()
        self.check(self.GetDeviceInfo(device, field, 0, None, ct.byref(length)), "device info size")
        data = ct.create_string_buffer(length.value)
        self.check(self.GetDeviceInfo(device, field, length, data, None), "device info")
        return data.value.decode(errors="replace")


class KernelProgram:
    def __init__(self, cl, device, source_bytes, kernel_names):
        self.cl = cl
        self.resources = []
        self.queue = None
        error = INT()
        dev = HANDLE(device)

        def keep(value, kind):
            if value:
                self.resources.append((kind, value))
            cl.check(error.value, "create " + kind)
            if not value:
                raise RuntimeError("Null OpenCL " + kind)
            return value

        try:
            self.context = keep(cl.CreateContext(None, 1, ct.byref(dev), None, None, ct.byref(error)), "Context")
            self.queue = keep(cl.CreateCommandQueue(self.context, dev, 0, ct.byref(error)), "CommandQueue")
            source = ct.c_char_p(source_bytes)
            program = keep(cl.CreateProgramWithSource(self.context, 1, ct.byref(source), None, ct.byref(error)), "Program")
            code = cl.BuildProgram(program, 1, ct.byref(dev), b"-cl-std=CL1.2", None, None)
            if code:
                length = SIZE()
                cl.check(cl.GetProgramBuildInfo(program, dev, 0x1183, 0, None, ct.byref(length)), "build log size")
                log = ct.create_string_buffer(length.value)
                cl.check(cl.GetProgramBuildInfo(program, dev, 0x1183, length, log, None), "build log")
                raise RuntimeError(f"Kernel build error {code}: {log.value.decode(errors='replace')}")
            self.kernels = [keep(cl.CreateKernel(program, name, ct.byref(error)), "Kernel")
                            for name in kernel_names]
        except Exception:
            self.close()
            raise

    def close(self):
        if self.queue:
            self.cl.Finish(self.queue)
        for kind, value in reversed(self.resources):
            getattr(self.cl, "Release" + kind)(value)
        self.resources.clear()
        self.queue = None


class Reconstruction(KernelProgram):
    def __init__(self, cl, device):
        super().__init__(cl, device, Path(__file__).with_name("idwt53.cl").read_bytes(),
                         (b"undo_update", b"undo_predict"))

    def decode(self, samples, length, lines, sample_stride, line_stride, parity):
        if not (1 <= length <= 4096 and 1 <= lines <= 4096 and parity in (0, 1)):
            raise ValueError("unsupported geometry")
        if sample_stride <= 0 or line_stride <= 0:
            raise ValueError("invalid stride")
        # The experiment accepts non-overlapping row or column families only.
        if not (line_stride >= length * sample_stride or
                sample_stride >= lines * line_stride):
            raise ValueError("overlapping lines")
        if (lines - 1) * line_stride + (length - 1) * sample_stride >= len(samples):
            raise ValueError("buffer too small")
        if any(abs(x) > 1048576 for x in samples):
            raise ValueError("outside qualified coefficient range")
        cl = self.cl
        data = (INT * len(samples))(*samples)
        error = INT()
        buffer = cl.CreateBuffer(self.context, 1 | 32, ct.sizeof(data), data, ct.byref(error))
        cl.check(error.value, "create coefficient buffer")
        try:
            args = [HANDLE(buffer), UINT(length), UINT(lines), UINT(sample_stride), UINT(line_stride), UINT(parity)]
            # Deliberately over-dispatch to exercise bounds guards.
            shape = (SIZE * 2)(length + 3, lines + 2)
            for kernel in self.kernels:
                for index, value in enumerate(args):
                    cl.check(cl.SetKernelArg(kernel, index, ct.sizeof(value), ct.byref(value)), "kernel argument")
                cl.check(cl.EnqueueNDRangeKernel(self.queue, kernel, 2, None, shape, None, 0, None, None), "dispatch")
            cl.check(cl.EnqueueReadBuffer(self.queue, buffer, 1, 0, ct.sizeof(data), data, 0, None, None), "readback")
            return list(data)
        finally:
            cl.Finish(self.queue)
            cl.ReleaseMemObject(buffer)


def forward53(values, parity):
    """Independent integer forward lifting; GPU must recover original values."""
    result = values.copy()
    length = len(result)
    if length == 1:
        result[0] *= 2 if parity else 1
        return result
    def neighbors(i):
        return result[i - 1 if i else 1] + result[i + 1 if i + 1 < length else i - 1]
    for i in range(length):
        if (i + parity) % 2:
            result[i] -= neighbors(i) // 2
    for i in range(length):
        if not (i + parity) % 2:
            result[i] += (neighbors(i) + 2) // 4
    return result


def verify(backend):
    rng = random.Random(0x53)
    cases = samples = 0
    for length in (1, 2, 3, 4, 5, 7, 16, 31, 32, 33, 63, 64, 65, 127, 256, 511, 1024):
        for parity in (0, 1):
            signals = [[0] * length, [123] * length, [-123] * length,
                       [32767 if i % 2 else -32768 for i in range(length)],
                       [i - length // 2 for i in range(length)],
                       [rng.randint(-32768, 32767) for _ in range(length)]]
            for impulse in (0, length // 2, length - 1):
                signal = [0] * length
                signal[impulse] = -1023
                signals.append(signal)
            for columns in (False, True):
                lines = len(signals)
                stride, pitch = (lines + 3, 1) if columns else (1, length + 3)
                size = (lines - 1) * pitch + (length - 1) * stride + 8
                expected = [777777] * size
                coefficients = expected.copy()
                for line, signal in enumerate(signals):
                    encoded = forward53(signal, parity)
                    for i, (original, coefficient) in enumerate(zip(signal, encoded)):
                        offset = line * pitch + i * stride
                        expected[offset] = original
                        coefficients[offset] = coefficient
                actual = backend.decode(coefficients, length, lines, stride, pitch, parity)
                if actual != expected:
                    i = next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
                    raise AssertionError(f"n={length} parity={parity} columns={columns} at={i}: {actual[i]} != {expected[i]}")
                cases += 1
                samples += length * lines
    print(f"PASS: {cases} GPU batches, {samples} exact reconstructed samples; padding unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list GPUs without running kernels")
    parser.add_argument("--device", help="unique case-insensitive device-name substring")
    parser.add_argument("--driver", help="driver-version substring when multiple ICDs expose the same GPU")
    args = parser.parse_args()
    cl = OpenCL()
    devices = cl.devices()
    for device, name in devices:
        print(f"GPU: {name}; driver {cl.device_text(device, 0x102D)}")
    if args.list:
        return
    if not args.device:
        parser.error("--device is required to select the test GPU explicitly")
    matches = [(d, n) for d, n in devices if args.device.lower() in n.lower()]
    if args.driver:
        matches = [(d, n) for d, n in matches if args.driver in cl.device_text(d, 0x102D)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching GPU, found {len(matches)}")
    print("Testing:", matches[0][1])
    backend = Reconstruction(cl, matches[0][0])
    try:
        verify(backend)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
