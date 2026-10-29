"""Loopback chat-completions/SSE adapter for the experimental native decoder."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid
from stream_text import StreamingAnswer
from template_text import compile_template

MODEL_ID='gpt-oss-120b-native'


class RequestError(ValueError): pass
class EngineError(RuntimeError): pass


class TextStops:
    def __init__(self,stops): self.stops=stops;self.pending='';self.matched=False
    def feed(self,text):
        if self.matched: return ''
        self.pending+=text
        hits=[index for stop in self.stops if (index:=self.pending.find(stop))>=0]
        if hits:
            result=self.pending[:min(hits)];self.pending='';self.matched=True;return result
        keep=0
        for stop in self.stops:
            for size in range(min(len(stop)-1,len(self.pending)),0,-1):
                if self.pending.endswith(stop[:size]): keep=max(keep,size);break
        result=self.pending[:-keep] if keep else self.pending
        self.pending=self.pending[-keep:] if keep else ''
        return result
    def finish(self):
        result='' if self.matched else self.pending;self.pending='';return result


@dataclass
class Plan:
    ids:list
    predict:int
    stream:bool
    stops:list
    include_usage:bool


def integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high: raise RequestError(f'{name} must be an integer between {low} and {high}')
    return value


class NativeRun:
    def __init__(self,command,timeout,scratch_dir):
        self.stderr=tempfile.TemporaryFile(dir=scratch_dir)
        try:
            self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',bufsize=1)
        except BaseException: self.stderr.close();raise
        self.deadline=time.monotonic()+timeout
        self.lines=queue.Queue();self.ids=[];self.fields={};self.eos=False
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
    def _read(self):
        try:
            for line in self.process.stdout: self.lines.put(line)
        finally: self.lines.put(None)
    def cancel(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill();self.process.wait()
    def close(self):
        self.cancel();self.reader.join(timeout=2)
        self.process.stdout.close();self.stderr.close()
    def events(self):
        while True:
            remaining=self.deadline-time.monotonic()
            if remaining<=0: raise EngineError('native request timed out')
            try: line=self.lines.get(timeout=min(1,remaining))
            except queue.Empty: yield None;continue
            if line is None: break
            if line.startswith('{'):
                event=json.loads(line)
                if event.get('event')!='token' or event.get('index')!=len(self.ids) or type(event.get('id')) is not int:
                    raise EngineError('invalid native token stream')
                self.ids.append(event['id']);self.eos=bool(event.get('eos'))
                yield event
            elif '=' in line:
                key,value=line.strip().split('=',1);self.fields[key]=value
        self.process.wait(timeout=max(.1,self.deadline-time.monotonic()))
        if self.process.returncode:
            self.stderr.seek(0,2);end=self.stderr.tell();self.stderr.seek(max(0,end-4096))
            detail=self.stderr.read().decode('utf-8',errors='replace').strip()
            raise EngineError(f'native engine failed: {detail or self.process.returncode}')
        summary=[int(value) for value in self.fields.get('predicted_token_ids','').split(',') if value]
        if summary!=self.ids: raise EngineError('native stream and summary disagree')


class NativeBackend:
    def __init__(self,args,engine_command=None):
        from fetch_expert import Header
        from tokenizers import Tokenizer
        self.args=args;self.gate=threading.Lock();self.active=None
        self.scratch_dir=Path(getattr(args,'scratch_dir',Path(__file__).resolve().parents[2]/'build-gpt-oss-api-temp'))
        self.scratch_dir.mkdir(parents=True,exist_ok=True)
        self.engine_command=engine_command or [str(args.engine.resolve())]
        with args.model.open('rb') as source:
            for size in (4<<20,8<<20,16<<20,32<<20):
                source.seek(0)
                try: metadata,_,_=Header(source.read(size)).parse();break
                except EOFError: continue
            else: raise ValueError('GGUF directory exceeds 32 MiB')
        if metadata.get('general.architecture')!='gpt-oss': raise ValueError('expected gpt-oss GGUF')
        if not 1<=args.context<=min(metadata['gpt-oss.context_length'],4096):
            raise ValueError('API context must fit the model and be at most 4096 in this prototype')
        self.template=compile_template(metadata['tokenizer.chat_template'])
        self.tokenizer=Tokenizer.from_file(str(args.tokenizer/'tokenizer.json'))
        self.vocab=self.tokenizer.get_vocab_size()
    def prepare(self,body):
        if not isinstance(body,dict): raise RequestError('request body must be a JSON object')
        if body.get('model',MODEL_ID)!=MODEL_ID: raise RequestError(f'model must be {MODEL_ID}')
        messages=body.get('messages')
        if not isinstance(messages,list) or not messages or len(messages)>256: raise RequestError('messages must contain 1..256 text messages')
        clean=[]
        for message in messages:
            if not isinstance(message,dict) or message.get('role') not in ('system','developer','user','assistant'):
                raise RequestError('supported roles: system, developer, user, assistant')
            if message.get('tool_calls') or message.get('function_call'): raise RequestError('tool calls are not supported by this prototype')
            content=message.get('content')
            if isinstance(content,list):
                if not all(isinstance(part,dict) and part.get('type')=='text' and isinstance(part.get('text'),str) for part in content):
                    raise RequestError('only text content parts are supported')
                content=''.join(part['text'] for part in content)
            if not isinstance(content,str): raise RequestError('message content must be text')
            clean.append({'role':message['role'],'content':content})
        # Never silently pretend to apply sampling/features absent in the native
        # decoder. Clients can select greedy decoding with temperature=0.
        for key,expected in {'temperature':0,'top_p':1,'top_k':0,'frequency_penalty':0,'presence_penalty':0,'repetition_penalty':1,'min_p':0}.items():
            value=body.get(key,expected)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or value!=expected or not math.isfinite(value):
                raise RequestError(f'this native prototype requires {key}={expected}')
        integer(body.get('n',1),'n',1,1)
        for key in ('tools','functions','logit_bias','logprobs'):
            if body.get(key): raise RequestError(f'{key} is not supported by this prototype')
        if body.get('response_format') not in (None,{'type':'text'}): raise RequestError('only response_format type=text is supported')
        if body.get('tool_choice','none')!='none': raise RequestError('tool_choice must be none')
        if body.get('reasoning_effort','low') not in ('none','low'): raise RequestError('reasoning_effort must be none or low')
        stream=body.get('stream',False)
        if type(stream) is not bool: raise RequestError('stream must be a boolean')
        options=body.get('stream_options',{})
        if options is None: options={}
        if not isinstance(options,dict) or type(options.get('include_usage',False)) is not bool: raise RequestError('invalid stream_options')
        stops=body.get('stop',[])
        if stops is None: stops=[]
        if isinstance(stops,str): stops=[stops]
        if not isinstance(stops,list) or len(stops)>4 or not all(isinstance(stop,str) and 1<=len(stop)<=1024 for stop in stops):
            raise RequestError('stop must be a nonempty string or up to four nonempty strings')
        count=body.get('max_completion_tokens',body.get('max_tokens',min(128,self.args.context)))
        predict=integer(count,'max_tokens',1,self.args.context)
        if 'max_tokens' in body and 'max_completion_tokens' in body and body['max_tokens']!=body['max_completion_tokens']:
            raise RequestError('max_tokens and max_completion_tokens disagree')
        try:
            prompt=self.template.render(messages=clean,tools=None,add_generation_prompt=True,reasoning_effort='low')+'<|channel|>final<|message|>'
            ids=self.tokenizer.encode(prompt,add_special_tokens=False).ids
        except Exception as error: raise RequestError(f'invalid chat template input: {error}') from error
        if not ids or len(ids)+predict-1>self.args.context:
            raise RequestError(f'prompt ({len(ids)} tokens) and output ({predict}) exceed context {self.args.context}')
        return Plan(ids,predict,stream,stops,options.get('include_usage',False))
    @contextmanager
    def run(self,plan):
        a=self.args
        command=[*self.engine_command,'--model',str(a.model.resolve()),'--tokens',','.join(map(str,plan.ids)),
            '--context',str(a.context),'--predict',str(plan.predict),'--eos','200002,199999,200012,200007',
            '--prompt-logits','last','--stream']
        if not a.scalar: command+=['--fast-cpu']
        if a.expert_ptx:
            command+=['--expert-ptx',str(a.expert_ptx.resolve()),'--expert-cache-mib',str(a.expert_cache_mib),
                '--cpu-expert-workers',str(a.cpu_expert_workers)]
            if a.group_hot_experts: command+=['--group-hot-experts']
        run=NativeRun(command,a.request_timeout,self.scratch_dir);self.active=run
        try: yield run
        finally: run.close();self.active=None
    def cancel(self):
        run=self.active
        if run: run.cancel()


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,backend,api_key=None):
        self.backend=backend;self.api_key=api_key
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def _cors(self):
        origin=self.headers.get('Origin','')
        try: parsed=urlsplit(origin)
        except ValueError: return False
        if parsed.scheme in ('http','https') and parsed.hostname in ('localhost','127.0.0.1','::1'):
            self.send_header('Access-Control-Allow-Origin',origin);self.send_header('Vary','Origin')
            return True
        return False
    def _json(self,status,body):
        data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Connection','close');self._cors();self.end_headers()
        self.close_connection=True;self.wfile.write(data);self.wfile.flush()
    def _error(self,status,message):
        self._json(status,{'error':{'message':message,'type':'invalid_request_error' if status<500 else 'server_error','code':status}})
    def _authorized(self):
        key=self.server.api_key
        if key and not hmac.compare_digest(self.headers.get('Authorization','').encode(),('Bearer '+key).encode()):
            self._error(401,'invalid API key');return False
        return True
    def do_GET(self):
        if not self._authorized(): return
        path=urlsplit(self.path).path.rstrip('/')
        if path=='/v1/models':
            self._json(200,{'object':'list','data':[{'id':MODEL_ID,'object':'model','created':0,'owned_by':'local',
                'context_length':self.server.backend.args.context,
                'modalities':{'input':['text'],'output':['text']},
                # The template is currently fixed to low; other official model
                # effort levels are not yet implemented by this adapter.
                'supported_reasoning_levels':[{'effort':'low'}]}]})
        elif path in ('','/health'):
            self._json(200,{'status':'busy' if self.server.backend.gate.locked() else 'ok','model':MODEL_ID})
        else: self._error(404,'unknown endpoint')
    def do_OPTIONS(self):
        if urlsplit(self.path).path.rstrip('/') not in ('/v1/models','/v1/chat/completions','/health'):
            self._error(404,'unknown endpoint');return
        self.send_response(204)
        self._cors();self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Authorization, Content-Type')
        self.send_header('Content-Length','0');self.send_header('Connection','close');self.end_headers();self.close_connection=True
    def _sse(self,value):
        data=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)
        self.wfile.write(('data: '+data+'\n\n').encode('utf-8'));self.wfile.flush()
    def do_POST(self):
        if not self._authorized(): return
        if urlsplit(self.path).path.rstrip('/')!='/v1/chat/completions': self._error(404,'unknown endpoint');return
        try:
            if self.headers.get('Transfer-Encoding'): raise RequestError('chunked request bodies are not supported')
            size=int(self.headers.get('Content-Length','0'))
            if not 1<=size<=2*1024*1024: raise RequestError('request body must be 1 byte to 2 MiB')
            self.connection.settimeout(15)
            body=json.loads(self.rfile.read(size).decode('utf-8'),parse_constant=lambda value:(_ for _ in ()).throw(RequestError('nonfinite JSON number')))
            plan=self.server.backend.prepare(body)
        except (ValueError,TypeError,UnicodeError,TimeoutError) as error:
            self._error(400,str(error));return
        backend=self.server.backend
        if not backend.gate.acquire(blocking=False): self._error(429,'native engine is busy; retry after the active request finishes');return
        streaming_started=False
        try:
            with backend.run(plan) as run:
                self.connection.settimeout(30)
                cid='chatcmpl-'+uuid.uuid4().hex;created=int(time.time())
                def chunk(delta,finish=None):
                    return {'id':cid,'object':'chat.completion.chunk','created':created,'model':MODEL_ID,
                        'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}
                if plan.stream:
                    self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control','no-cache');self.send_header('X-Accel-Buffering','no')
                    self.send_header('Connection','close');self._cors();self.end_headers();self.close_connection=True
                    streaming_started=True;self._sse(chunk({'role':'assistant','content':''}))
                renderer=StreamingAnswer(backend.tokenizer);stopper=TextStops(plan.stops);parts=[];heartbeat=time.monotonic()
                def emit(text):
                    if text:
                        parts.append(text)
                        if plan.stream: self._sse(chunk({'content':text}))
                for event in run.events():
                    if event is None:
                        if plan.stream and time.monotonic()-heartbeat>=5:
                            self.wfile.write(b': waiting\n\n');self.wfile.flush();heartbeat=time.monotonic()
                        continue
                    if event['id']<0 or event['id']>=backend.vocab: raise EngineError('native token outside tokenizer vocabulary')
                    emit(stopper.feed(renderer.feed(event['id'])))
                    if stopper.matched: run.cancel();break
                if not stopper.matched:
                    if not run.ids or (not run.eos and len(run.ids)!=plan.predict): raise EngineError('native generation ended unexpectedly')
                    emit(stopper.feed(renderer.finish()));emit(stopper.finish())
                finish='stop' if run.eos or stopper.matched else 'length'
                usage={'prompt_tokens':len(plan.ids),'completion_tokens':len(run.ids),'total_tokens':len(plan.ids)+len(run.ids)}
                if plan.stream:
                    self._sse(chunk({},finish))
                    if plan.include_usage:
                        self._sse({'id':cid,'object':'chat.completion.chunk','created':created,'model':MODEL_ID,'choices':[],'usage':usage})
                    self._sse('[DONE]')
                else:
                    timings={key:float(run.fields[key]) for key in ('prefill_seconds','first_token_seconds','decode_seconds',
                        'elapsed_seconds','decode_tokens_per_second') if key in run.fields}
                    self._json(200,{'id':cid,'object':'chat.completion','created':created,'model':MODEL_ID,
                        'choices':[{'index':0,'message':{'role':'assistant','content':''.join(parts)},'finish_reason':finish}],
                        'usage':usage,'timings':timings})
        except (BrokenPipeError,ConnectionResetError,TimeoutError): pass
        except Exception as error:
            try:
                if streaming_started:
                    self._sse({'error':{'message':str(error),'type':'server_error'}});self._sse('[DONE]')
                else: self._error(500,str(error))
            except (BrokenPipeError,ConnectionResetError,TimeoutError): pass
        finally: backend.gate.release()


def main():
    root=Path(__file__).resolve().parents[2];model_dir=root.parent/'strata_data/models/gpt-oss-120b-MXFP4'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=model_dir/'gpt-oss-120b-MXFP4.gguf')
    parser.add_argument('--tokenizer',type=Path,default=model_dir/'tokenizer')
    parser.add_argument('--engine',type=Path,default=root/'build-gpt-oss-ninja/experimental/gpt-oss/strata-gpt-oss-cpu.exe')
    parser.add_argument('--expert-ptx',type=Path,default=root/'build-gpt-oss/gpu_expert.ptx')
    parser.add_argument('--port',type=int,default=8097)
    parser.add_argument('--context',type=int,default=2048)
    parser.add_argument('--expert-cache-mib',type=int,default=4096)
    parser.add_argument('--cpu-expert-workers',type=int,default=4)
    parser.add_argument('--request-timeout',type=float,default=900)
    parser.add_argument('--scratch-dir',type=Path,default=root/'build-gpt-oss-api-temp')
    parser.add_argument('--group-hot-experts',action='store_true')
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--scalar',action='store_true')
    parser.add_argument('--api-key-env',help='Optional environment variable holding the bearer key; key is never printed')
    args=parser.parse_args()
    if args.cpu_only: args.expert_ptx=None
    if not 1<=args.port<=65535 or not 1<=args.cpu_expert_workers<=16 or not 1<=args.expert_cache_mib<=1048576:
        parser.error('invalid port, worker count or expert cache size')
    if not math.isfinite(args.request_timeout) or args.request_timeout<=0: parser.error('request timeout must be positive')
    for path in [args.engine,args.model,args.tokenizer/'tokenizer.json',*([args.expert_ptx] if args.expert_ptx else [])]:
        if not path.is_file(): parser.error(f'file not found: {path}')
    key=os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key: parser.error('API key environment variable is empty')
    backend=NativeBackend(args);server=Server(('127.0.0.1',args.port),backend,key)
    print(f'GPT-OSS experimental API ready: http://127.0.0.1:{args.port}/v1; model={MODEL_ID}; temperature=0',flush=True)
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally: backend.cancel();server.server_close()


if __name__=='__main__': main()
