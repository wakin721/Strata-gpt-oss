"""One-shot native text probe using the downloaded GGUF's own Harmony template.

This is a prototype command, not an API server. All model compute runs in the
native decoder; Python only renders the template and encodes/decodes token IDs.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--system',default='Answer the user clearly and accurately.')
    parser.add_argument('--max-new-tokens',type=int,default=32)
    parser.add_argument('--context',type=int,default=256)
    parser.add_argument('--expert-ptx',type=Path)
    parser.add_argument('--expert-cache-mib',type=int,default=4096)
    parser.add_argument('--fast',action='store_true',help='Opt in to AVX2/F16C, last-prompt logits and persistent CPU expert workers')
    parser.add_argument('--cpu-expert-workers',type=int,default=4,help='With --fast and --expert-ptx: persistent CPU workers (1..16)')
    parser.add_argument('--thinking',action='store_true',help='Allow analysis before final; otherwise request the final channel directly')
    parser.add_argument('--output',type=Path,help='Optional local JSON record')
    args=parser.parse_args()
    if args.max_new_tokens<1 or args.context<1: parser.error('token limits must be positive')
    if not 1<=args.cpu_expert_workers<=16: parser.error('CPU expert workers must be between 1 and 16')
    if args.output and args.output.exists(): parser.error('output already exists; choose a new path')
    from fetch_expert import Header
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template
    with args.model.open('rb') as source:
        for size in (4<<20,8<<20,16<<20,32<<20):
            source.seek(0)
            try: metadata,_,_=Header(source.read(size)).parse();break
            except EOFError: continue
        else: raise ValueError('GGUF directory exceeds 32 MiB')
    if metadata.get('general.architecture')!='gpt-oss': raise ValueError('expected gpt-oss GGUF')
    template=_compile_jinja_template(metadata['tokenizer.chat_template'])
    today=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()
    messages=[{'role':'system','content':args.system},{'role':'user','content':args.prompt}]
    prompt=template.render(messages=messages,add_generation_prompt=True,tools=None,reasoning_effort='low',current_date=today)
    if not args.thinking: prompt+='<|channel|>final<|message|>'
    tokenizer=Tokenizer.from_file(str(args.tokenizer/'tokenizer.json'))
    ids=tokenizer.encode(prompt,add_special_tokens=False).ids
    if len(ids)+args.max_new_tokens-1>args.context: parser.error('prompt and requested output exceed --context')
    command=[str(args.engine.resolve()),'--model',str(args.model.resolve()),'--tokens',','.join(map(str,ids)),
        '--context',str(args.context),'--predict',str(args.max_new_tokens),'--eos','200002,199999,200012']
    if args.expert_ptx:
        command+=['--expert-ptx',str(args.expert_ptx.resolve()),'--expert-cache-mib',str(args.expert_cache_mib)]
    if args.fast:
        command+=['--fast-cpu','--prompt-logits','last']
        if args.expert_ptx: command+=['--cpu-expert-workers',str(args.cpu_expert_workers)]
    print(f'Native GPT-OSS prototype: {len(ids)} prompt tokens. Processing may take several minutes.',flush=True)
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=None,text=True,encoding='utf-8')
    try: stdout,_=process.communicate()
    except KeyboardInterrupt:
        process.terminate();process.wait();raise
    if process.returncode: raise RuntimeError(f'native engine exited with {process.returncode}')
    fields=dict(line.split('=',1) for line in stdout.splitlines() if '=' in line)
    generated=[int(value) for value in fields.get('predicted_token_ids','').split(',') if value]
    raw=tokenizer.decode(generated,skip_special_tokens=False)
    answer=raw
    if '<|channel|>final<|message|>' in answer: answer=answer.rsplit('<|channel|>final<|message|>',1)[1]
    elif args.thinking: answer='[No final channel was reached within the token limit.]'
    for stop in ('<|return|>','<|endoftext|>','<|call|>'): answer=answer.split(stop,1)[0]
    print(answer,flush=True)
    print(f"[engine elapsed: {fields.get('elapsed_seconds','?')} seconds]",flush=True)
    if args.output:
        record={'prototype':True,'prompt_tokens':len(ids),'generated_ids':generated,'raw_output':raw,'answer':answer,'engine_metrics':fields}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    main()
