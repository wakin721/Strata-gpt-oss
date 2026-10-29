"""Measured request state and Strata-compatible read-only monitor endpoints."""
from collections import deque
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from serve.telemetry import Telemetry
from strata_cli.client import normalize_legacy


def number(fields,key):
    try: return float(fields[key])
    except (KeyError,ValueError,TypeError): return None


class Monitor:
    def __init__(self,backend,model):
        self.backend=backend;self.model=model;self.lock=threading.RLock()
        self.boot=time.time();self.instance=uuid.uuid4().hex
        self.current=None;self.recent=deque(maxlen=50);self.activity=deque(maxlen=32);self.sequence=0
        self.totals={'since':self.boot,'requests':0,'prompt_tokens':0,'output_tokens':0}
        self.disconnects=0;self.timeouts=0;self.rejected=0
        self.hardware=Telemetry(extra=self.rates)
    def close(self):
        self.hardware.close();self.hardware._thread.join(timeout=2)
    def _event(self,kind,outcome=None):
        self.sequence+=1
        self.activity.append({'seq':self.sequence,'at':time.time(),'request_id':self.current['id'],
            'event':kind,'outcome':outcome})
    def begin(self,request_id,plan):
        with self.lock:
            self.current={'id':request_id,'time':time.time(),'started_mono':time.monotonic(),
                'prompt_tokens':len(plan.ids),'max_tokens':plan.predict,'generated':0,'first_mono':None,
                'first_engine_token_s':None}
            self.totals['requests']+=1;self.totals['prompt_tokens']+=len(plan.ids)
            self._event('reading')
    def token(self,event):
        with self.lock:
            if not self.current: return
            if self.current['first_mono'] is None:
                self.current['first_mono']=time.monotonic()
                self.current['first_engine_token_s']=event.get('seconds')
                self._event('first_token')
            self.current['generated']+=1;self.totals['output_tokens']+=1
    def rates(self):
        with self.lock:
            c=self.current
            if not c or c['first_mono'] is None or c['generated']<2: return {'tok_s':None}
            duration=time.monotonic()-c['first_mono']
            return {'tok_s':(c['generated']-1)/duration if duration>0 else None}
    def finish(self,outcome,fields):
        with self.lock:
            c=self.current
            if not c: return
            duration=time.monotonic()-c['started_mono']
            first=(c['first_mono']-c['started_mono']) if c['first_mono'] is not None else None
            prefill=number(fields,'prefill_seconds')
            decode=number(fields,'decode_seconds')
            rate=number(fields,'decode_tokens_per_second')
            if rate is None and c['generated']>=2 and first is not None and duration>first:
                rate=(c['generated']-1)/(duration-first)
            self.recent.appendleft({'id':c['id'],'time':c['time'],'duration_s':duration,'finish':outcome,
                'prompt_tokens':c['prompt_tokens'],'reused':0,'output_tokens':c['generated'],
                'first_engine_token_s':c['first_engine_token_s'],'first_token_s':first,
                'prompt_ms':prefill*1000 if prefill is not None else None,
                'decode_ms':decode*1000 if decode is not None else None,'decode_tok_s':rate,
                'error':outcome in ('error','timeout')})
            if outcome=='disconnect': self.disconnects+=1
            if outcome=='timeout': self.timeouts+=1
            self._event({'stop':'finished','length':'finished','tool_calls':'finished','disconnect':'disconnected','timeout':'timeout'}.get(outcome,'error'),outcome)
            self.current=None
    def status(self):
        with self.lock:
            c=dict(self.current) if self.current else None
        run=self.backend.active
        return {'service':'strata','backend':'gpt-oss-native','model':self.model,'started':self.boot,
            'loaded':bool(run and run.process.poll() is None),
            'cache_max_tokens':self.backend.args.context,'busy':bool(c),
            'generated':c['generated'] if c else 0,
            'concurrency':{'serving':1,'requested':1,'active':int(bool(c)),'queued':0}}
    def metrics(self):
        with self.lock:
            c=dict(self.current) if self.current else None
            recent=list(self.recent);totals=dict(self.totals)
            rate=self.rates()['tok_s'];now=time.monotonic()
        state='generating' if c and c['first_mono'] is not None else 'reading' if c else 'idle'
        slot={'slot':0,'group':0,'state':'decoding' if state=='generating' else state,
            'request_id':c['id'] if c else None,'prompt_tokens':c['prompt_tokens'] if c else None,
            'held_tokens':c['prompt_tokens']+c['generated'] if c else 0,
            'generated':c['generated'] if c else 0,'prompt_read':None,
            'elapsed_s':now-c['started_mono'] if c else None,'tok_s':rate}
        hw=self.hardware.snapshot()
        return {'time':time.time(),'engine':{'model':self.model,'max_context':self.backend.args.context,
            'backend':'gpt-oss-native','expert_cache_mib':self.backend.args.expert_cache_mib if self.backend.args.expert_ptx else 0,
            'experts':getattr(self.backend,'total_experts',None),'expert_slots':None,'batch_groups':1},
            'live':{'state':state,'phase':'prefill' if state=='reading' else 'decode' if c else None,
                'queued':0,'running':int(bool(c)),'prompt_tokens':c['prompt_tokens'] if c else None,
                'prompt_read':None,'prompt_total':None,'generated':c['generated'] if c else None,
                'max_tokens':c['max_tokens'] if c else None,'elapsed_s':slot['elapsed_s'],
                'tok_s':rate,'tok_s_mean':rate,'tok_s_window_s':None,'slots':[slot]},
            'requests':recent,'requests_kept':len(recent),'totals':totals,
            'hardware':hw['now'],'hardware_static':hw['static'],'history':hw['history']}
    def snapshot(self):
        metrics=self.metrics();status=self.status()
        result=normalize_legacy(metrics,status)
        with self.lock:
            result['source']='gpt-oss-native'
            result['instance'].update(id=self.instance,state='ready')
            result['rates']['output_tokens_server_counter']=self.totals['output_tokens']
            result['work'].update(slots_requested=1,admission_limit=1,waiting=0,outside_slots=0,
                admission_rejected=self.rejected,queue_timeouts=self.timeouts,client_disconnect_cancels=self.disconnects)
            result['work']['requests']=[{'id':self.current['id'],'request_id':self.current['id'],
                'state':'decoding' if self.current['first_mono'] is not None else 'reading',
                'prompt_tokens':self.current['prompt_tokens'],'generated':self.current['generated']}] if self.current else []
            result['activity']={'last_seq':self.sequence,'events':list(self.activity)}
            result['quality']['request_correlation']='measured'
        return result
