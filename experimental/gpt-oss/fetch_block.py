"""Build a labelled single-activation block fixture from GPT-OSS-120B ranges.

Select top-4 from the ORIGINAL 128-way router on the fixed input, then download
only those four experts. Compacting these four router rows preserves this ONE
activation's top-k weights. The result cannot evaluate arbitrary inputs or text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from fetch_expert import Header,fetch_range,write_fixture


def decode(data,shape,kind):
    if kind==0: return np.frombuffer(data,dtype='<f4').astype(np.float64).reshape(tuple(reversed(shape)))
    if kind==8:
        blocks=np.frombuffer(data,dtype=np.uint8).reshape(-1,34)
        scale=blocks[:,:2].copy().view('<f2').reshape(-1).astype(np.float64)
        values=blocks[:,2:].copy().view(np.int8).astype(np.float64)*scale[:,None]
    elif kind==39:
        blocks=np.frombuffer(data,dtype=np.uint8).reshape(-1,17)
        codes=np.concatenate((blocks[:,1:]&15,blocks[:,1:]>>4),axis=-1)
        table=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
        values=table[codes]*np.exp2(blocks[:,0].astype(np.float64)[:,None]-127)
    else: raise ValueError(f'unsupported real fixture tensor type: {kind}')
    return values.reshape(tuple(reversed(shape)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--url',default='https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/7200b74/gpt-oss-120b-MXFP4.gguf')
    parser.add_argument('--layer',type=int,default=0)
    args=parser.parse_args()
    if args.out.exists(): parser.error('output exists; choose a new path')
    meta,info,data_start=Header(args.header.read_bytes()).parse()
    if meta.get('general.architecture')!='gpt-oss': raise ValueError('not GPT-OSS')
    h=meta['gpt-oss.embedding_length']; f=meta['gpt-oss.expert_feed_forward_length']
    ne=meta['gpt-oss.expert_count']; top=meta['gpt-oss.expert_used_count']
    if not 0<=args.layer<meta['gpt-oss.block_count']: raise ValueError('invalid layer')
    args.cache.mkdir(parents=True,exist_ok=True)
    tensors=[]; weights={}; provenance=[]
    prefix=f'blk.{args.layer}.'
    def download(name,expert=None):
        shape,kind,offset=info[name]
        if kind not in (0,8,39): raise ValueError(f'unsupported tensor {name}')
        elements=int(np.prod(shape)); size=elements*4 if kind==0 else elements//32*(34 if kind==8 else 17)
        if expert is not None:
            if shape[-1]!=ne: raise ValueError('invalid expert axis')
            size//=ne
        start=data_start+offset+(expert*size if expert is not None else 0)
        key=hashlib.sha256(f'{args.url}:{start}:{size}'.encode()).hexdigest()
        path=args.cache/(key+'.bin'); digest_path=args.cache/(key+'.sha256')
        data=path.read_bytes() if path.exists() else b''
        valid=len(data)==size and digest_path.exists() and hashlib.sha256(data).hexdigest()==digest_path.read_text().strip()
        if not valid:
            print(f'Fetching {name} expert={expert}: {size:,} bytes',flush=True)
            data=fetch_range(args.url,start,size)
            path.write_bytes(data); digest_path.write_text(hashlib.sha256(data).hexdigest()+'\n')
        provenance.append({'tensor':name,'expert':expert,'start':start,'bytes':size,'sha256':hashlib.sha256(data).hexdigest()})
        return shape,kind,data
    for name in info:
        if name.startswith(prefix) and not any(f'ffn_{p}_exps.' in name for p in ('gate','up','down')):
            shape,kind,data=download(name)
            fixture_name='blk.0.'+name[len(prefix):]
            tensors.append((fixture_name,shape,kind,data)); weights[fixture_name]=decode(data,shape,kind)
    # Independent float64 single-token attention. At position zero the RoPE angle
    # is zero, but YaRN magnitude still scales Q and K.
    inp=((np.arange(h)%17)-8)/16.
    eps=meta['gpt-oss.attention.layer_norm_rms_epsilon']
    norm=lambda x,w: x/np.sqrt(np.mean(x*x)+eps)*w
    x=norm(inp,weights['blk.0.attn_norm.weight'])
    heads=meta['gpt-oss.attention.head_count']; kvheads=meta['gpt-oss.attention.head_count_kv']; d=meta['gpt-oss.attention.key_length']
    def project(p): return (weights[f'blk.0.attn_{p}.weight']@x+weights[f'blk.0.attn_{p}.bias'])
    amplitude=1.
    if meta.get('gpt-oss.rope.scaling.type')=='yarn': amplitude=1+.1*np.log(meta['gpt-oss.rope.scaling.factor'])
    q=project('q').reshape(heads,d)*amplitude
    k=np.repeat(project('k').reshape(kvheads,d),heads//kvheads,axis=0)*amplitude
    v=np.repeat(project('v').reshape(kvheads,d),heads//kvheads,axis=0)
    scores=np.sum(q*k,axis=-1)/np.sqrt(d)
    # Stable two-way softmax: one key plus the per-head attention sink.
    p=1/(1+np.exp(np.clip(weights['blk.0.attn_sinks.weight']-scores,-700,700)))
    mixed=(v*p[:,None]).reshape(-1)
    residual=inp+weights['blk.0.attn_output.weight']@mixed+weights['blk.0.attn_output.bias']
    x=norm(residual,weights['blk.0.post_attention_norm.weight'])
    scores=weights['blk.0.ffn_gate_inp.weight']@x+weights['blk.0.ffn_gate_inp.bias']
    selected=np.argsort(-scores,kind='stable')[:top]
    probabilities=np.exp(scores[selected]-np.max(scores[selected])); probabilities/=probabilities.sum()
    print(f'Original router top-{top}: {selected.tolist()}',flush=True)
    router_meta={k:v for k,v in meta.items() if k.startswith('gpt-oss.') and isinstance(v,(str,int,float))}
    router_meta.update({'general.architecture':'gpt-oss','general.name':'GPT-OSS attention/router-only fixture; no experts',
        'gpt-oss.block_count':1,'gpt-oss.attention.sliding_window_pattern':[1 if args.layer%2==0 else 0]})
    router_path=args.out.with_suffix('.router-only.gguf')
    if not router_path.exists():
        write_fixture(router_path,router_meta,tensors+[('token_embd.weight',[h,1],0,inp.astype('<f4').tobytes())])
    args.out.with_suffix('.router-expected.f32').write_bytes(x.astype('<f4').tobytes())
    # Replace only the router rows and expert axis, preserving selected values.
    tensors=[t for t in tensors if t[0] not in ('blk.0.ffn_gate_inp.weight','blk.0.ffn_gate_inp.bias')]
    for suffix in ('weight','bias'):
        name=f'blk.0.ffn_gate_inp.{suffix}'; array=weights[name][selected].astype('<f4')
        shape=[h,top] if suffix=='weight' else [top]
        tensors.append((name,shape,0,array.tobytes()))
    mixture=np.zeros(h,dtype=np.float64)
    for p,expert in zip(probabilities,selected):
        ew={}
        for projection in ('gate','up','down'):
            for suffix in ('weight','bias'):
                name=prefix+f'ffn_{projection}_exps.{suffix}'
                shape,kind,data=download(name,int(expert))
                ew[(projection,suffix)]=(shape,kind,data,decode(data,shape[:-1],kind))
        gate=ew['gate','weight'][3]@x+ew['gate','bias'][3]
        up=ew['up','weight'][3]@x+ew['up','bias'][3]
        gate=np.minimum(gate,7); up=np.clip(up,-7,7)
        hidden=gate/(1+np.exp(-1.702*gate))*(up+1)
        mixture+=p*(ew['down','weight'][3]@hidden+ew['down','bias'][3])
        for key,(shape,kind,data,_) in ew.items():
            name=f'blk.0.ffn_{key[0]}_exps.{key[1]}'
            existing=next((t for t in tensors if t[0]==name),None)
            if existing:
                tensors.remove(existing); tensors.append((name,existing[1],kind,existing[3]+data))
            else: tensors.append((name,shape[:-1]+[top],kind,data))
    expected=(residual+mixture).astype('<f4')
    fixture_meta={k:v for k,v in meta.items() if k.startswith('gpt-oss.') and isinstance(v,(str,int,float))}
    fixture_meta.update({'general.architecture':'gpt-oss','general.name':'GPT-OSS selected-expert block fixture; one synthetic activation only',
        'gpt-oss.block_count':1,'gpt-oss.expert_count':top,'strata.probe.source_layer':args.layer,
        'strata.probe.kind':'single-activation-selected-expert-block','strata.probe.source_url':args.url})
    # Source layer 0 is sliding, layer 1 global; preserve it when remapping to 0.
    fixture_meta['gpt-oss.attention.sliding_window_pattern']=[1 if args.layer%2==0 else 0]
    tensors.append(('token_embd.weight',[h,1],0,inp.astype('<f4').tobytes()))
    write_fixture(args.out,fixture_meta,tensors)
    expected_path=args.out.with_suffix('.expected.f32'); expected_path.write_bytes(expected.tobytes())
    report={'generation_supported':False,'scope':'ONE fixed synthetic activation, original full router selection',
        'source_url':args.url,'source_layer':args.layer,'original_expert_ids':selected.tolist(),'top_k_weights':probabilities.tolist(),
        'reference':'NumPy float64 attention and routed experts, F32 output','slices':provenance}
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {args.out}: {args.out.stat().st_size:,} bytes; expected {expected_path}',flush=True)


if __name__=='__main__': main()
