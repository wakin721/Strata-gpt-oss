"""Independent batch NumPy oracle versus the native streaming decoder."""
import importlib.util
from pathlib import Path
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location("fetch",Path(__file__).with_name("fetch_expert.py"))
fetch=importlib.util.module_from_spec(spec); spec.loader.exec_module(fetch)
CPU=Path(sys.argv.pop(1)).resolve()
TEMP=Path.cwd()/"build-gpt-oss-decoder-test-temp"; TEMP.mkdir(exist_ok=True)


def encode(array,kind,rng):
    array=np.asarray(array,dtype=np.float32)
    if kind==0: return array.astype('<f4').tobytes(),array.astype(np.float64)
    if kind==1:
        rounded=array.astype('<f2')
        return rounded.tobytes(),rounded.astype(np.float64)
    if kind==30:
        # Round-to-nearest-even BF16.
        bits=array.view(np.uint32)
        rounded=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
        return rounded.tobytes(),(rounded.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    blocks=array.reshape(-1,32)
    if kind==8:
        output=bytearray(); decoded=[]
        for block in blocks:
            scale=np.float16(np.max(np.abs(block))/127)
            q=np.rint(block/float(scale)).clip(-127,127).astype(np.int8)
            output.extend(struct.pack('<e',scale)+q.tobytes())
            decoded.extend(q.astype(np.float64)*float(scale))
        return bytes(output),np.asarray(decoded).reshape(array.shape)
    if kind==39:
        codes=rng.integers(0,16,(len(blocks),32),dtype=np.uint8)
        exponents=rng.integers(119,124,len(blocks),dtype=np.uint8)
        table=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
        decoded=table[codes]*np.exp2(exponents.astype(np.float64)[:,None]-127)
        output=np.empty((len(blocks),17),dtype=np.uint8)
        output[:,0]=exponents; output[:,1:]=codes[:,:16]|(codes[:,16:]<<4)
        return output.tobytes(),decoded.reshape(array.shape)
    raise ValueError(kind)


def fixture(path,*,dense_kind=0,expert_kind=0,yarn=False,pattern=None):
    rng=np.random.default_rng(109)
    h,f,heads,kv,d,ne,k,vocab,layers=32,32,4,2,8,4,2,11,3
    meta={'general.architecture':'gpt-oss','gpt-oss.embedding_length':h,'gpt-oss.expert_feed_forward_length':f,
        'gpt-oss.block_count':layers,'gpt-oss.expert_count':ne,'gpt-oss.expert_used_count':k,
        'gpt-oss.attention.head_count':heads,'gpt-oss.attention.head_count_kv':kv,'gpt-oss.attention.key_length':d,
        'gpt-oss.attention.value_length':d,'gpt-oss.context_length':256,'gpt-oss.attention.sliding_window':3,
        'gpt-oss.rope.freq_base':150000.,'gpt-oss.attention.layer_norm_rms_epsilon':1e-5}
    if yarn:
        meta.update({'gpt-oss.rope.scaling.type':'yarn','gpt-oss.rope.scaling.factor':32.,
            'gpt-oss.rope.scaling.original_context_length':4096,'gpt-oss.rope.scaling.yarn_beta_fast':32.,
            'gpt-oss.rope.scaling.yarn_beta_slow':1.})
    if pattern is not None: meta['gpt-oss.attention.sliding_window_pattern']=pattern
    tensors=[]; weights={}
    def put(name,shape,kind=0,array=None):
        if array is None: array=rng.normal(0,.12,tuple(reversed(shape)))
        data,decoded=encode(array,kind,rng)
        tensors.append((name,shape,kind,data)); weights[name]=decoded
    put('token_embd.weight',[h,vocab],dense_kind)
    put('output.weight',[h,vocab],dense_kind)
    put('output_norm.weight',[h],array=rng.uniform(.5,1.5,h))
    for layer in range(layers):
        p=f'blk.{layer}.'
        for proj,size in (('q',heads*d),('k',kv*d),('v',kv*d)):
            put(p+f'attn_{proj}.weight',[h,size],dense_kind)
            put(p+f'attn_{proj}.bias',[size])
        put(p+'attn_output.weight',[heads*d,h],dense_kind)
        put(p+'attn_output.bias',[h])
        put(p+'attn_sinks.weight',[heads],array=np.linspace(-5,5,heads))
        for name in ('attn_norm.weight','post_attention_norm.weight'):
            put(p+name,[h],array=rng.uniform(.5,1.5,h))
        put(p+'ffn_gate_inp.weight',[h,ne])
        put(p+'ffn_gate_inp.bias',[ne])
        for projection,cols,rows in (('gate',h,f),('up',h,f),('down',f,h)):
            put(p+f'ffn_{projection}_exps.weight',[cols,rows,ne],expert_kind)
            put(p+f'ffn_{projection}_exps.bias',[rows,ne],array=rng.normal(0,4,(ne,rows)))
    fetch.write_fixture(path,meta,tensors)
    return meta,weights


def reference(ids,meta,w):
    """Compute all tokens as a batch; no streaming/ring-buffer implementation."""
    heads,kv,d=meta['gpt-oss.attention.head_count'],meta['gpt-oss.attention.head_count_kv'],meta['gpt-oss.attention.key_length']
    state=w['token_embd.weight'][ids].copy()
    epsilon=meta['gpt-oss.attention.layer_norm_rms_epsilon']
    def norm(x,weight): return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon)*weight
    n=len(ids); positions=np.arange(n)
    frequency=meta['gpt-oss.rope.freq_base']**(-np.arange(0,d,2)/d)
    amplitude=1.
    if meta.get('gpt-oss.rope.scaling.type')=='yarn':
        original=meta['gpt-oss.rope.scaling.original_context_length']; base=meta['gpt-oss.rope.freq_base']
        low=max(0,math.floor(d*math.log(original/(32*2*math.pi))/(2*math.log(base))))
        high=min(d-1,math.ceil(d*math.log(original/(2*math.pi))/(2*math.log(base))))
        ramp=1-np.clip((np.arange(d//2)-low)/max(.001,high-low),0,1)
        factor=meta['gpt-oss.rope.scaling.factor']
        frequency=frequency/factor*(1-ramp)+frequency*ramp
        amplitude=1+.1*math.log(factor)
    cos=np.cos(positions[:,None]*frequency[None,:])[:,None,:]*amplitude
    sin=np.sin(positions[:,None]*frequency[None,:])[:,None,:]*amplitude
    def rotate(x):
        a,b=np.split(x,2,axis=-1)
        return np.concatenate((a*cos-b*sin,a*sin+b*cos),axis=-1)
    pattern=meta.get('gpt-oss.attention.sliding_window_pattern',2)
    for layer in range(meta['gpt-oss.block_count']):
        p=f'blk.{layer}.'; x=norm(state,w[p+'attn_norm.weight'])
        def projection(name): return x@w[p+f'attn_{name}.weight'].T+w[p+f'attn_{name}.bias']
        q=rotate(projection('q').reshape(n,heads,d)); k=rotate(projection('k').reshape(n,kv,d))
        v=projection('v').reshape(n,kv,d)
        k=np.repeat(k,heads//kv,axis=1); v=np.repeat(v,heads//kv,axis=1)
        scores=np.einsum('thd,shd->hts',q,k)/math.sqrt(d)
        valid=positions[None,:]<=positions[:,None]
        sliding=bool(pattern[layer]) if isinstance(pattern,list) else pattern==0 or (layer+1)%pattern!=0
        if sliding: valid &= positions[None,:]>positions[:,None]-meta['gpt-oss.attention.sliding_window']
        scores=np.where(valid[None,:,:],scores,-np.inf)
        sinks=np.broadcast_to(w[p+'attn_sinks.weight'][:,None,None],(heads,n,1))
        scores=np.concatenate((scores,sinks),axis=-1)
        probabilities=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
        probabilities/=np.sum(probabilities,axis=-1,keepdims=True)
        mixed=np.einsum('hts,shd->thd',probabilities[:,:,:n],v).reshape(n,heads*d)
        state=state+mixed@w[p+'attn_output.weight'].T+w[p+'attn_output.bias']
        normalized=norm(state,w[p+'post_attention_norm.weight'])
        logits=normalized@w[p+'ffn_gate_inp.weight'].T+w[p+'ffn_gate_inp.bias']
        top=np.argsort(-logits,axis=-1,kind='stable')[:,:meta['gpt-oss.expert_used_count']]
        selected=np.take_along_axis(logits,top,axis=-1)
        probabilities=np.exp(selected-np.max(selected,axis=-1,keepdims=True)); probabilities/=probabilities.sum(axis=-1,keepdims=True)
        mixture=np.zeros_like(state)
        for token in range(n):
            for index,expert in enumerate(top[token]):
                gate=w[p+'ffn_gate_exps.weight'][expert]@normalized[token]+w[p+'ffn_gate_exps.bias'][expert]
                up=w[p+'ffn_up_exps.weight'][expert]@normalized[token]+w[p+'ffn_up_exps.bias'][expert]
                gate=np.minimum(gate,7); up=np.clip(up,-7,7)
                hidden=gate/(1+np.exp(-1.702*gate))*(up+1)
                value=w[p+'ffn_down_exps.weight'][expert]@hidden+w[p+'ffn_down_exps.bias'][expert]
                mixture[token]+=probabilities[token,index]*value
        state+=mixture
    return norm(state,w['output_norm.weight'])@w['output.weight'].T


class DecoderTests(unittest.TestCase):
    def run_cpu(self,path,ids,out,extra=()):
        if os.environ.get('GPT_OSS_FAST_CPU'):
            extra=(*extra,'--fast-cpu')
        if os.environ.get('GPT_OSS_GPU_PTX'):
            extra=(*extra,'--expert-ptx',os.environ['GPT_OSS_GPU_PTX'],'--expert-cache-mib','1')
            if os.environ.get('GPT_OSS_CPU_WORKERS'):
                extra=(*extra,'--cpu-expert-workers',os.environ['GPT_OSS_CPU_WORKERS'])
        result=subprocess.run([str(CPU),'--model',str(path),'--tokens',','.join(map(str,ids)),
            '--context','256','--logits-out',str(out),*extra],capture_output=True,text=True,timeout=30)
        if os.environ.get('GPT_OSS_FAST_CPU') and 'AVX2/F16C kernels are unsupported' in result.stderr:
            self.skipTest('AVX2/F16C unavailable')
        return result

    def test_batch_oracle_against_streaming_and_ring_wrap(self):
        cases=[(0,0,False,None),(8,39,True,None),(1,39,False,[0,1,0]),(30,0,True,[1,1,0])]
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            for index,(dense,expert,yarn,pattern) in enumerate(cases):
                path=Path(folder)/f'model-{index}.gguf'; out=Path(folder)/f'logits-{index}.bin'
                meta,w=fixture(path,dense_kind=dense,expert_kind=expert,yarn=yarn,pattern=pattern)
                ids=[0,1,2,4,5,6,7,9,3,0,2,8,10,9,2,1,4]
                result=self.run_cpu(path,ids,out,('--reset-check',))
                self.assertEqual(result.returncode,0,result.stderr)
                actual=np.fromfile(out,dtype='<f4').reshape(len(ids),-1)
                expected=reference(ids,meta,w)
                np.testing.assert_allclose(actual,expected,atol=8e-5,rtol=8e-5)
                self.assertIn('reset_check=passed',result.stdout)
                if os.environ.get('GPT_OSS_GPU_PTX'):
                    counters=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
                    self.assertGreater(int(counters['gpu_expert_calls']),0)
                    self.assertGreater(int(counters['cpu_expert_calls']),0)
                    self.assertLessEqual(int(counters['expert_cache_bytes']),1024*1024)
                if os.environ.get('GPT_OSS_PRIMITIVES'):
                    primitive=subprocess.run([os.environ['GPT_OSS_PRIMITIVES'],str(path)],capture_output=True,text=True,timeout=30)
                    self.assertEqual(primitive.returncode,0,primitive.stderr)
                    print(primitive.stdout.strip())
                print(f'decoder case {index}: dense={dense} expert={expert} yarn={yarn} max_abs={np.max(np.abs(actual-expected)):.3g}')

    def test_autoregressive_tokens_and_context_guards(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path=Path(folder)/'model.gguf'; out=Path(folder)/'logits.bin'
            meta,w=fixture(path,yarn=True,expert_kind=39)
            ids=[2,3,5]; expected=[]
            for _ in range(4):
                next_id=int(np.argmax(reference(ids,meta,w)[-1])); expected.append(next_id); ids.append(next_id)
            result=self.run_cpu(path,[2,3,5],out,('--predict','4'))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('predicted_token_ids='+','.join(map(str,expected)),result.stdout)
            actual=np.fromfile(out,dtype='<f4').reshape(6,-1)
            np.testing.assert_allclose(actual,reference(ids[:-1],meta,w),atol=8e-5,rtol=8e-5)
            stopped=self.run_cpu(path,[2,3,5],Path(folder)/'stopped.bin',('--predict','4','--eos',str(expected[0])))
            self.assertEqual(stopped.returncode,0,stopped.stderr)
            self.assertIn('predicted_token_ids='+str(expected[0])+'\n',stopped.stdout)
            self.assertIn('logit_records=3',stopped.stdout)
            # Invalid input and an oversized generation must fail before any output file is created.
            for bad_ids,extra in (([11],()),([0],('--context','0')),([0],('--predict','300'))):
                bad=Path(folder)/'bad.bin'
                result=self.run_cpu(path,bad_ids,bad,extra)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(bad.exists())

    def test_last_prompt_logits_preserves_generation_and_reset(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            path=Path(folder)/'model.gguf'
            fixture(path,dense_kind=8,expert_kind=39,yarn=True)
            ids=[0,1,4,2,5,9,3]
            full=Path(folder)/'all.bin'
            result=self.run_cpu(path,ids,full,('--predict','4'))
            self.assertEqual(result.returncode,0,result.stderr)
            full_logits=np.fromfile(full,dtype='<f4').reshape(10,-1)
            predicted=next(line for line in result.stdout.splitlines() if line.startswith('predicted_token_ids='))
            for reset in (False,True):
                out=Path(folder)/f'last-{reset}.bin'
                extra=('--predict','4','--prompt-logits','last',*(['--reset-check'] if reset else []))
                result=self.run_cpu(path,ids,out,extra)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn(predicted+'\n',result.stdout)
                actual=np.fromfile(out,dtype='<f4').reshape(5 if reset else 4,-1)
                expected=full_logits[[0,6,7,8,9] if reset else [6,7,8,9]]
                np.testing.assert_array_equal(actual,expected)
                self.assertIn('evaluated_positions=10',result.stdout)
                if reset: self.assertIn('reset_check=passed',result.stdout)


if __name__=='__main__': unittest.main()
