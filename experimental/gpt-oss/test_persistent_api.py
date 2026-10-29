"""Real HTTP/SSE with a fake persistent process: reuse, progress and recovery."""
import json
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import patch
from test_api_server import ApiFixture,TOOL

FAKE_WORKER='''import json,sys,time
from pathlib import Path
config_path=Path(sys.argv[1])
config=json.loads(config_path.read_text())
if config.get('ready_delay'):time.sleep(config['ready_delay'])
print(json.dumps({'event':'ready','protocol':1}),flush=True)
previous=[]
for line in sys.stdin:
 if line.strip()=='quit':break
 rid,count,eos,reuse,raw=line.strip().split('\\t');count=int(count);eos=[int(x) for x in eos.split(',')];prompt=[int(x) for x in raw.split(',')]
 config=json.loads(config_path.read_text())
 if config.get('fail'):
  sys.stderr.write('test persistent native failure '+('x'*100000));sys.exit(4)
 common=0
 while reuse=='1' and common<min(len(previous),len(prompt)-1) and previous[common]==prompt[common]:common+=1
 def emit(value):
  if config.get('bad_timing') and value['event']=='prefill':value['seconds']=float('nan')
  print(json.dumps({'request_id':rid,**value}),flush=True)
 if config.get('bad_request_id'):rid='wrong'
 emit({'event':'prefill','done':common,'total':len(prompt),'reused':common,'seconds':0})
 time.sleep(config.get('prefill',0))
 emit({'event':'prefill','done':len(prompt)+(1 if config.get('bad_progress') else 0),'total':len(prompt),'reused':common,'seconds':.1})
 ids=config['ids'][:count];produced=[]
 for index,token in enumerate(ids):
  produced.append(token)
  emit({'event':'token','index':index,'id':token,'seconds':.1+index*.01,'eos':token in eos})
  if index==2 and config.get('ack'):
   deadline=time.monotonic()+5
   while not Path(config['ack']).exists():
    if time.monotonic()>deadline:sys.exit(3)
    time.sleep(.01)
  time.sleep(config.get('delay',0))
  if token in eos:break
 previous=prompt
 if config.get('bad_summary'):produced=[]
 emit({'event':'done','ids':produced,'fields':{'elapsed_seconds':.1,'prefill_seconds':.08,'prompt_reused_tokens':common,'prompt_evaluated_tokens':len(prompt)-common}})
 if config.get('exit_after_done'):sys.exit(0)
'''


class PersistentApiTests(ApiFixture):
    def setUp(self):
        super().setUp()
        fake=Path(self.folder.name)/'persistent_fake.py';fake.write_text(FAKE_WORKER,encoding='utf-8')
        self.backend.engine_command=[sys.executable,'-u',str(fake),str(self.config)]
        self.backend.args.persistent_engine=True
    def wait_idle(self):
        import time,threading
        end=time.monotonic()+3
        while self.backend.gate.locked() and time.monotonic()<end: threading.Event().wait(.01)
        self.assertFalse(self.backend.gate.locked());self.assertIsNone(self.backend.active)
        self.assertFalse(list(self.backend.scratch_dir.glob('*.tokens')))

    def test_worker_reuse_usage_monitor_and_shutdown(self):
        body=self.body()
        status,first=self.request(body);self.assertEqual(status,200,first);self.wait_idle()
        worker=self.backend.worker;self.assertIsNone(worker.process.poll())
        self.assertEqual(first['usage']['prompt_tokens_details']['cached_tokens'],0)
        status,second=self.request(body);self.assertEqual(status,200,second);self.wait_idle()
        self.assertIs(self.backend.worker,worker)
        self.assertEqual(second['choices'][0]['message'],first['choices'][0]['message'])
        self.assertEqual(second['usage']['prompt_tokens_details']['cached_tokens'],first['usage']['prompt_tokens']-1)
        _,metrics=self.request(path='/metrics')
        self.assertEqual(metrics['requests'][0]['reused'],first['usage']['prompt_tokens']-1)
        _,state=self.request(path='/status');self.assertTrue(state['loaded']);self.assertFalse(state['busy'])
        status,uncached=self.request(self.body(cache_prompt=False));self.assertEqual(status,200,uncached);self.wait_idle()
        self.assertEqual(uncached['usage']['prompt_tokens_details']['cached_tokens'],0)
        self.assertIs(self.backend.worker,worker)
        self.assertEqual(self.request(self.body(cache_prompt='yes'))[0],400)
        self.backend.cancel();self.assertIsNotNone(worker.process.poll());self.assertTrue(worker.closed)

    def test_progress_and_stream_before_worker_can_finish(self):
        ack=Path(self.folder.name)/'ack';self.configure(prefill=.3,ack=str(ack))
        connection=self.connection();connection.request('POST','/v1/chat/completions',json.dumps(self.body(stream=True,stream_options={'include_usage':True})))
        response=connection.getresponse();self.assertEqual(response.status,200)
        _,metrics=self.request(path='/metrics');self.assertEqual(metrics['live']['state'],'reading')
        deadline=time.monotonic()+2
        while metrics['live']['prompt_read'] is None and time.monotonic()<deadline:
            time.sleep(.01);_,metrics=self.request(path='/metrics')
        self.assertEqual(metrics['live']['prompt_read'],0)
        chunks=[]
        while line:=response.readline():
            if not line.startswith(b'data: '):continue
            if line[6:].strip()==b'[DONE]':break
            item=json.loads(line[6:]);chunks.append(item)
            if any(c['delta'].get('content') for c in item.get('choices',[])) and not ack.exists():
                self.assertIsNone(self.backend.active.process.poll());ack.touch()
        connection.close();self.wait_idle()
        self.assertTrue(any('usage' in x for x in chunks));self.assertIsNone(self.backend.worker.process.poll())

    def test_corrupt_events_timeout_and_crash_discard_state_then_recover(self):
        for change in ({'bad_request_id':True},{'bad_progress':True},{'bad_summary':True},{'bad_timing':True},{'fail':True}):
            self.configure(**change)
            status,data=self.request(self.body());self.assertEqual(status,500,(change,data));self.wait_idle()
            self.assertTrue(self.backend.worker.closed)
            self.configure();status,data=self.request(self.body());self.assertEqual(status,200,data);self.wait_idle()
            self.assertEqual(data['usage']['prompt_tokens_details']['cached_tokens'],0)
        broken=self.backend.worker
        with patch.object(broken.process.stdin,'write',side_effect=BrokenPipeError('test pipe closed')):
            status,data=self.request(self.body());self.assertEqual(status,500,data);self.wait_idle()
        self.assertTrue(broken.closed)
        self.assertEqual(self.request(self.body())[0],200);self.wait_idle()
        crashed=self.backend.worker;crashed.process.terminate();crashed.process.wait(timeout=5)
        status,data=self.request(self.body());self.assertEqual(status,200,data);self.wait_idle()
        self.assertIsNot(self.backend.worker,crashed);self.assertTrue(crashed.closed)
        self.assertEqual(data['usage']['prompt_tokens_details']['cached_tokens'],0)
        self.backend.cancel();self.configure(ready_delay=1);self.backend.args.request_timeout=.1
        status,data=self.request(self.body());self.assertEqual(status,500);self.assertIn('timed out',data['error']['message']);self.wait_idle()
        self.assertTrue(self.backend.worker.closed)
        self.backend.args.request_timeout=5;self.configure();self.assertEqual(self.request(self.body())[0],200);self.wait_idle()

    def test_stop_disconnect_and_abandoned_stream_restart_worker(self):
        self.configure(delay=.05)
        connection=self.connection();connection.request('POST','/v1/chat/completions',json.dumps(self.body(stream=True)))
        stream_socket=connection.sock;response=connection.getresponse();self.assertEqual(response.status,200)
        self.assertEqual(self.request(self.body())[0],429)
        while line:=response.readline():
            if line.startswith(b'data: '):break
        stream_socket.shutdown(socket.SHUT_RDWR);response.close();connection.close();self.wait_idle()
        self.assertTrue(self.backend.worker.closed)
        self.configure();status,data=self.request(self.body(stop='夜🙂'));self.assertEqual(status,200,data);self.wait_idle()
        self.assertEqual(data['choices'][0]['message']['content'],'雨');self.assertTrue(self.backend.worker.closed)
        self.assertEqual(self.request(self.body())[0],200);self.wait_idle()

    def test_reasoning_and_tools_keep_completed_worker(self):
        self.harmony_output('<|channel|>analysis<|message|>想<|end|><|start|>assistant<|channel|>final<|message|>好<|return|>')
        status,data=self.request(self.body(reasoning_effort='high',max_tokens=200));self.assertEqual(status,200,data);self.wait_idle()
        worker=self.backend.worker;self.assertEqual(data['choices'][0]['message']['reasoning_content'],'想')
        self.harmony_output('{"city":"台北"}<|call|>','<|call|>')
        status,data=self.request(self.body(tools=[TOOL],tool_choice={'type':'function','function':{'name':'weather'}},max_tokens=64))
        self.assertEqual(status,200,data);self.assertEqual(data['choices'][0]['finish_reason'],'tool_calls');self.wait_idle()
        self.assertIs(self.backend.worker,worker);self.assertIsNone(worker.process.poll())


if __name__=='__main__': unittest.main()
