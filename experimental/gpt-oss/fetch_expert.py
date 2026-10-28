"""Download one published expert with HTTP ranges, not the entire 63 GB model.

The resulting GGUF is a labelled expert fixture. It is NOT a usable language model.
Only the six selected expert tensors are included, with expert_count set to one.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request


def fetch_range_once(url: str, start: int, size: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start+size-1}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=45) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match or (int(match[1]), int(match[2])) != (start, start+size-1):
            raise RuntimeError("server did not honor the exact HTTP range; refusing a full model download")
        data = response.read(size+1)
        if len(data) != size:
            raise RuntimeError(f"truncated or oversized range: received {len(data)}, expected {size}")
        return data


def fetch_range(url: str, start: int, size: int) -> bytes:
    for attempt in range(3):
        try:
            return fetch_range_once(url,start,size)
        except (OSError,RuntimeError) as error:
            if attempt == 2:
                raise
            print(f"Retrying range {start}+{size}: {type(error).__name__}",flush=True)


class Header:
    def __init__(self, data: bytes):
        self.data, self.offset = data, 0

    def take(self, size: int) -> bytes:
        end = self.offset+size
        if end > len(self.data):
            raise EOFError("GGUF directory is larger than the downloaded prefix")
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def unpack(self, fmt: str):
        return struct.unpack("<"+fmt, self.take(struct.calcsize("<"+fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("Q")).decode("utf-8")

    def value(self, kind: int, keep=True):
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.unpack("I"), self.unpack("Q")
            if count > 1 << 24:
                raise ValueError("invalid GGUF array length")
            # Tokens and merges are intentionally discarded.
            for _ in range(count):
                self.value(element, False)
            return None
        formats = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
        return self.unpack(formats[kind])

    def parse(self):
        if self.take(4) != b"GGUF" or self.unpack("I") != 3:
            raise ValueError("expected GGUF v3")
        tensor_count, metadata_count = self.unpack("Q"), self.unpack("Q")
        if tensor_count > 100000 or metadata_count > 100000:
            raise ValueError("implausible GGUF directory")
        metadata = {}
        for _ in range(metadata_count):
            name, kind = self.string(), self.unpack("I")
            metadata[name] = self.value(kind)
        tensors = {}
        for _ in range(tensor_count):
            name, dimensions = self.string(), self.unpack("I")
            if dimensions not in (1,2,3,4):
                raise ValueError("invalid tensor rank")
            shape = [self.unpack("Q") for _ in range(dimensions)]
            kind, offset = self.unpack("I"), self.unpack("Q")
            tensors[name] = (shape, kind, offset)
        alignment = metadata.get("general.alignment",32)
        if not isinstance(alignment,int) or alignment <= 0 or alignment > 4096:
            raise ValueError("invalid GGUF alignment")
        return metadata, tensors, (self.offset+alignment-1)//alignment*alignment


def write_fixture(path: Path, metadata: dict, tensors: list[tuple[str,list[int],int,bytes]]):
    def string(s):
        encoded=s.encode("utf-8")
        return struct.pack("<Q",len(encoded))+encoded
    header=bytearray(b"GGUF"+struct.pack("<IQQ",3,len(tensors),len(metadata)))
    for name,value in metadata.items():
        header.extend(string(name))
        if isinstance(value,str):
            header.extend(struct.pack("<I",8)+string(value))
        else:
            header.extend(struct.pack("<II",4,value))
    payload=bytearray()
    for name,shape,kind,data in tensors:
        payload.extend(b"\0"*((-len(payload))%32))
        header.extend(string(name)+struct.pack("<I",len(shape)))
        header.extend(struct.pack("<"+"Q"*len(shape),*shape))
        header.extend(struct.pack("<IQ",kind,len(payload)))
        payload.extend(data)
    header.extend(b"\0"*((-len(header))%32))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as output:
        output.write(header)
        output.write(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/7200b74/gpt-oss-120b-MXFP4.gguf")
    parser.add_argument("--layer",type=int,default=0)
    parser.add_argument("--expert",type=int,default=0)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("output already exists; choose another path")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    cache=args.out.parent/("header-"+hashlib.sha256(args.url.encode()).hexdigest()[:16]+".bin")
    cached=cache.read_bytes() if cache.exists() else b""
    for size in (4<<20,8<<20,16<<20,32<<20):
        print(f"Reading {size>>20} MiB of GGUF directory",flush=True)
        try:
            prefix=cached if len(cached)>=size else fetch_range(args.url,0,size)
            cache.write_bytes(prefix)
            meta,info,data_start=Header(prefix).parse()
            break
        except EOFError:
            continue
    else:
        raise ValueError("GGUF directory exceeds 32 MiB")
    if meta.get("general.architecture") != "gpt-oss":
        raise ValueError("source is not GPT-OSS")
    h,f,layers,experts=[meta[f"gpt-oss.{key}"] for key in ("embedding_length","expert_feed_forward_length","block_count","expert_count")]
    if not (0 <= args.layer < layers and 0 <= args.expert < experts):
        raise ValueError("layer or expert out of range")
    slices,provenance=[],[]
    for projection,cols,rows in (("gate",h,f),("up",h,f),("down",f,h)):
        for suffix in ("weight","bias"):
            name=f"blk.{args.layer}.ffn_{projection}_exps.{suffix}"
            shape,kind,offset=info[name]
            if suffix == "weight":
                if kind != 39 or shape != [cols,rows,experts] or cols%32:
                    raise ValueError(f"unsupported MXFP4 layout: {name}: {shape}, {kind}")
                size=cols//32*17*rows
                fixture_shape=[cols,rows,1]
            else:
                if kind != 0 or shape != [rows,experts]:
                    raise ValueError(f"unsupported bias layout: {name}: {shape}, {kind}")
                size=rows*4
                fixture_shape=[rows,1]
            start=data_start+offset+args.expert*size
            print(f"Fetching {name}: {size:,} bytes",flush=True)
            data=fetch_range(args.url,start,size)
            fixture_name=f"blk.0.ffn_{projection}_exps.{suffix}"
            slices.append((fixture_name,fixture_shape,kind,data))
            provenance.append({"tensor":name,"start":start,"bytes":size,"sha256":hashlib.sha256(data).hexdigest()})
    fixture_meta={"general.architecture":"gpt-oss","general.name":"GPT-OSS expert-only fixture; NOT a language model",
        "gpt-oss.embedding_length":h,"gpt-oss.expert_feed_forward_length":f,"gpt-oss.block_count":1,"gpt-oss.expert_count":1,
        "strata.probe.source_layer":args.layer,"strata.probe.source_expert":args.expert,"strata.probe.source_url":args.url}
    write_fixture(args.out,fixture_meta,slices)
    args.out.with_suffix(".json").write_text(json.dumps({"generation_supported":False,"source_url":args.url,
        "source_layers":layers,"source_experts":experts,"slices":provenance},indent=2)+"\n",encoding="utf-8")
    print(f"Saved expert fixture: {args.out} ({args.out.stat().st_size:,} bytes). It cannot generate text.",flush=True)


if __name__ == "__main__":
    main()
