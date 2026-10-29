"""Local, sequential native-engine ablations; no downloads or server changes."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--baseline-engine',type=Path,help='Optional saved pre-optimization executable')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--ptx',type=Path,required=True)
    parser.add_argument('--tokens-json',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache-mib',type=int,default=4096)
    parser.add_argument('--case',action='append',choices=['short-scalar','short-avx2','short-last','short-workers','template-fast','short-grouped','template-grouped',
        'short-baseline','template-baseline'],help='Run only selected cases; may be repeated')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    ids=json.loads(args.tokens_json.read_text())
    if isinstance(ids,dict): ids=ids['prompt_token_ids']
    cases=[('short-scalar',ids[-13:],8,[]),
        ('short-avx2',ids[-13:],8,['--fast-cpu']),
        ('short-last',ids[-13:],8,['--fast-cpu','--prompt-logits','last']),
        ('short-workers',ids[-13:],8,['--fast-cpu','--prompt-logits','last','--cpu-expert-workers','4']),
        ('template-fast',ids,2,['--fast-cpu','--prompt-logits','last','--cpu-expert-workers','4']),
        ('short-grouped',ids[-13:],8,['--fast-cpu','--prompt-logits','last','--cpu-expert-workers','4','--group-hot-experts']),
        ('template-grouped',ids,2,['--fast-cpu','--prompt-logits','last','--cpu-expert-workers','4','--group-hot-experts'])]
    cases=[(name,prompt,predict,extra,args.engine) for name,prompt,predict,extra in cases]
    if args.baseline_engine:
        cases=[('short-baseline',ids[-13:],8,[],args.baseline_engine),
            ('template-baseline',ids,2,[],args.baseline_engine),*cases]
    if args.case:
        if set(args.case)-{case[0] for case in cases}: parser.error('baseline cases require --baseline-engine')
        cases=[case for case in cases if case[0] in args.case]
    for name,_,_,_,_ in cases:
        if (args.out/f'{name}.f32').exists() or (args.out/f'{name}.log').exists():
            raise FileExistsError(f'refusing to overwrite {name}')
    for name,prompt,predict,extra,engine in cases:
        output=args.out/f'{name}.f32'
        log=args.out/f'{name}.log'
        if output.exists() or log.exists(): raise FileExistsError(f'refusing to overwrite {name}')
        command=[str(engine.resolve()),'--model',str(args.model.resolve()),'--tokens',','.join(map(str,prompt)),
            '--predict',str(predict),'--context','128','--expert-ptx',str(args.ptx.resolve()),
            '--expert-cache-mib',str(args.cache_mib),'--logits-out',str(output.resolve()),*extra]
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        print(name,result.returncode,log.read_text(),flush=True)
        if result.returncode: raise RuntimeError(f'{name} failed')


if __name__=='__main__': main()
