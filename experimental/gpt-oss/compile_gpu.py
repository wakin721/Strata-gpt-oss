"""Compile the experimental expert kernels with NVIDIA NVRTC, into build output."""
import argparse
import ctypes as c
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvrtc-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--arch',default='compute_120')
    args=parser.parse_args()
    dlls=list(args.nvrtc_root.rglob('nvrtc64_*.dll'))
    if len(dlls)!=1: raise RuntimeError(f'expected one NVRTC DLL, found {len(dlls)}')
    dll_dir=os.add_dll_directory(str(dlls[0].parent.resolve()))
    # NVRTC loads its builtins by name; preload the wheel's matching sibling.
    builtins=list(dlls[0].parent.glob('nvrtc-builtins64_*.dll'))
    if len(builtins)!=1: raise RuntimeError('expected one NVRTC builtins DLL')
    builtins_handle=c.CDLL(str(builtins[0].resolve()))
    nvrtc=c.CDLL(str(dlls[0].resolve()))
    program=c.c_void_p()
    source=Path(__file__).with_name('gpu_expert.cu').read_bytes()
    nvrtc.nvrtcCreateProgram.argtypes=[c.POINTER(c.c_void_p),c.c_char_p,c.c_char_p,c.c_int,c.c_void_p,c.c_void_p]
    nvrtc.nvrtcCompileProgram.argtypes=[c.c_void_p,c.c_int,c.POINTER(c.c_char_p)]
    nvrtc.nvrtcGetProgramLogSize.argtypes=[c.c_void_p,c.POINTER(c.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes=[c.c_void_p,c.c_void_p]
    nvrtc.nvrtcGetPTXSize.argtypes=[c.c_void_p,c.POINTER(c.c_size_t)]
    nvrtc.nvrtcGetPTX.argtypes=[c.c_void_p,c.c_void_p]
    nvrtc.nvrtcDestroyProgram.argtypes=[c.POINTER(c.c_void_p)]
    def check(code):
        if code: raise RuntimeError(f'NVRTC error {code}')
    check(nvrtc.nvrtcCreateProgram(c.byref(program),source,b'gpu_expert.cu',0,None,None))
    try:
        options=[f'--gpu-architecture={args.arch}'.encode(),b'--std=c++17',b'--fmad=false']
        values=(c.c_char_p*len(options))(*options)
        status=nvrtc.nvrtcCompileProgram(program,len(options),values)
        size=c.c_size_t(); check(nvrtc.nvrtcGetProgramLogSize(program,c.byref(size)))
        log=c.create_string_buffer(size.value); check(nvrtc.nvrtcGetProgramLog(program,log))
        if log.value: print(log.value.decode('utf-8',errors='replace'))
        check(status)
        check(nvrtc.nvrtcGetPTXSize(program,c.byref(size)))
        ptx=c.create_string_buffer(size.value); check(nvrtc.nvrtcGetPTX(program,ptx))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(ptx.raw)
        print(f'Compiled {args.arch} PTX: {args.out} ({size.value} bytes)')
    finally:
        nvrtc.nvrtcDestroyProgram(c.byref(program)); dll_dir.close()


if __name__=='__main__': main()
