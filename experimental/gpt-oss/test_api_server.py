"""Exercise real HTTP/SSE and native-process lifetime with a tiny fake engine."""
from argparse import Namespace
import http.client
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from api_server import MODEL_ID,NativeBackend,Server,TextStops
from fetch_expert import write_fixture
from test_stream_text import tokenizer

FAKE='''import argparse,json,sys,time
from pathlib import Path
config=json.loads(Path(sys.argv.pop(1)).read_text())
parser=argparse.ArgumentParser();parser.add_argument('--predict',type=int);args,_=parser.parse_known_args()
if config.get('fail'):
 sys.stderr.write('test native failure '+('x'*100000));sys.exit(4)
time.sleep(config.get('prefill',0))
ids=config['ids'][:args.predict]
for index,token in enumerate(ids):
 print(json.dumps({'event':'token','index':index,'id':token,'seconds':index*.01,'eos':token==config['eos']}),flush=True)
 if index==2 and config.get('ack'):
  deadline=time.monotonic()+5
  while not Path(config['ack']).exists():
   if time.monotonic()>deadline:sys.exit(3)
   time.sleep(.01)
 time.sleep(config.get('delay',0))
print('predicted_token_ids='+','.join(map(str,ids)),flush=True)
print('elapsed_seconds=0.1',flush=True)
'''


class ApiTests(unittest.TestCase):
    def setUp(self):
        temp_root=Path.cwd()/'build-gpt-oss-api-test-temp';temp_root.mkdir(exist_ok=True)
        self.folder=tempfile.TemporaryDirectory(dir=temp_root);root=Path(self.folder.name)
        self.config=root/'fake.json';fake=root/'fake.py';fake.write_text(FAKE,encoding='utf-8')
        tok=tokenizer();(root/'tokenizer').mkdir();tok.save(str(root/'tokenizer/tokenizer.json'))
        self.defaults={'ids':tok.encode('雨夜🙂<|return|>',add_special_tokens=False).ids,'eos':tok.token_to_id('<|return|>')}
        self.configure()
        model=root/'model.gguf'
        write_fixture(model,{'general.architecture':'gpt-oss','gpt-oss.context_length':2048,
            'tokenizer.chat_template':"{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"},[])
        args=Namespace(model=model,tokenizer=root/'tokenizer',engine=Path(sys.executable),context=2048,
            expert_ptx=None,scalar=True,expert_cache_mib=1,cpu_expert_workers=1,group_hot_experts=False,request_timeout=5)
        self.backend=NativeBackend(args,[sys.executable,'-u',str(fake),str(self.config)])
        self.http=Server(('127.0.0.1',0),self.backend)
        self.thread=threading.Thread(target=self.http.serve_forever,kwargs={'poll_interval':.02},daemon=True);self.thread.start()
    def tearDown(self):
        self.backend.cancel();self.http.shutdown();self.http.server_close();self.thread.join();self.folder.cleanup()
    def configure(self,**kwargs): self.config.write_text(json.dumps({**self.defaults,**kwargs}),encoding='utf-8')
    def connection(self): return http.client.HTTPConnection(*self.http.server_address,timeout=8)
    def request(self,body=None,path='/v1/chat/completions',headers=None):
        connection=self.connection()
        connection.request('POST' if body is not None else 'GET',path,json.dumps(body).encode() if body is not None else None,
            {'Content-Type':'application/json',**(headers or {})})
        response=connection.getresponse();status=response.status;data=response.read();connection.close()
        return status,json.loads(data)
    def body(self,**kwargs): return {'model':MODEL_ID,'messages':[{'role':'user','content':'你好'}],'max_tokens':16,**kwargs}
    def wait_idle(self):
        end=time.monotonic()+3
        while self.backend.gate.locked() and time.monotonic()<end: threading.Event().wait(.01)
        self.assertFalse(self.backend.gate.locked());self.assertIsNone(self.backend.active)
    def test_models_auth_and_buffered_multiturn(self):
        status,data=self.request(path='/v1/models');self.assertEqual(status,200);self.assertEqual(data['data'][0]['id'],MODEL_ID)
        self.http.api_key='secret'
        self.assertEqual(self.request(path='/v1/models')[0],401)
        self.assertEqual(self.request(path='/v1/models',headers={'Authorization':'Bearer secret'})[0],200)
        self.http.api_key=None
        body=self.body(response_format={'type':'text'},messages=[{'role':'system','content':'用中文回答'},{'role':'user','content':[{'type':'text','text':'你好'}]},
            {'role':'assistant','content':'你好'},{'role':'user','content':'继续'}])
        status,data=self.request(body);self.assertEqual(status,200,data)
        self.assertEqual(data['choices'][0]['message']['content'],'雨夜🙂');self.assertEqual(data['choices'][0]['finish_reason'],'stop')
        self.assertEqual(data['usage']['completion_tokens'],len(self.defaults['ids']))
        self.assertEqual(data['timings'],{'elapsed_seconds':.1})
        self.wait_idle()
    def test_local_browser_preflight_and_cors(self):
        self.http.api_key='secret'
        connection=self.connection();connection.request('OPTIONS','/v1/chat/completions',headers={'Origin':'http://localhost:5173',
            'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'})
        response=connection.getresponse();self.assertEqual(response.status,204)
        self.assertEqual(response.getheader('Access-Control-Allow-Origin'),'http://localhost:5173');response.read();connection.close()
        connection=self.connection();connection.request('GET','/v1/models',headers={'Origin':'https://elsewhere.example','Authorization':'Bearer secret'})
        response=connection.getresponse();self.assertIsNone(response.getheader('Access-Control-Allow-Origin'));response.read();connection.close()
    def test_sse_delivers_text_before_engine_may_finish(self):
        ack=Path(self.folder.name)/'ack';self.configure(ack=str(ack))
        connection=self.connection();connection.request('POST','/v1/chat/completions',json.dumps(self.body(stream=True,stream_options={'include_usage':True})))
        response=connection.getresponse();self.assertEqual(response.status,200)
        self.assertIn('text/event-stream',response.getheader('Content-Type'))
        chunks=[];done=False
        while line:=response.readline():
            if not line.startswith(b'data: '):continue
            value=line[6:].strip()
            if value==b'[DONE]':done=True;break
            chunk=json.loads(value);chunks.append(chunk)
            for choice in chunk.get('choices',[]):
                if choice['delta'].get('content') and not ack.exists():
                    self.assertIsNone(self.backend.active.process.poll())
                    ack.touch()
        connection.close();self.assertTrue(done)
        text=''.join(choice['delta'].get('content','') for chunk in chunks for choice in chunk['choices'])
        self.assertEqual(text,'雨夜🙂');self.assertTrue(any(chunk.get('usage') for chunk in chunks))
        self.wait_idle()
    def test_stop_across_tokens_and_length(self):
        status,data=self.request(self.body(stop='夜🙂'));self.assertEqual(status,200,data)
        self.assertEqual(data['choices'][0]['message']['content'],'雨');self.assertEqual(data['choices'][0]['finish_reason'],'stop')
        self.wait_idle()
        status,data=self.request(self.body(max_tokens=2));self.assertEqual(status,200,data)
        self.assertEqual(data['choices'][0]['finish_reason'],'length');self.assertEqual(data['usage']['completion_tokens'],2)
        self.wait_idle()
    def test_invalid_requests_fail_before_engine_start(self):
        for change in ({'temperature':.7},{'temperature':10**400},{'n':True},{'stream':'yes'},{'stop':''},{'stop':['']},
            {'stream_options':[]},{'model':'other'},{'tools':[{}]},{'max_tokens':True},{'messages':[]},
            {'messages':[{'role':'user','content':[{'type':'image_url','image_url':'x'}]}]},
            {'max_tokens':2048},{'max_tokens':2,'max_completion_tokens':3}):
            status,data=self.request(self.body(**change));self.assertEqual(status,400,(change,data));self.assertIsNone(self.backend.active)
    def test_failure_timeout_and_disconnect_release_single_flight(self):
        self.configure(fail=True)
        status,data=self.request(self.body());self.assertEqual(status,500);self.assertIn('native engine failed',data['error']['message']);self.wait_idle()
        connection=self.connection();connection.request('POST','/v1/chat/completions',json.dumps(self.body(stream=True)))
        response=connection.getresponse();self.assertEqual(response.status,200)
        wire=response.read();connection.close()
        self.assertIn(b'"error"',wire);self.assertIn(b'data: [DONE]',wire);self.wait_idle()
        self.configure(prefill=2);self.backend.args.request_timeout=.1
        status,data=self.request(self.body());self.assertEqual(status,500);self.assertIn('timed out',data['error']['message']);self.wait_idle()
        self.backend.args.request_timeout=5;self.configure(delay=.1)
        connection=self.connection();connection.request('POST','/v1/chat/completions',json.dumps(self.body(stream=True)))
        stream_socket=connection.sock
        response=connection.getresponse();self.assertEqual(response.status,200)
        self.assertEqual(self.request(self.body())[0],429)
        # Close after the role event. The next SSE write detects the dead client
        # and must stop the child instead of holding the GPU slot indefinitely.
        while line:=response.readline():
            if line.startswith(b'data: '):break
        stream_socket.shutdown(socket.SHUT_RDWR);response.close();connection.close();self.wait_idle()
        self.configure();self.assertEqual(self.request(self.body())[0],200);self.wait_idle()


if __name__=='__main__': unittest.main()
