"""Single-flight native worker lifetime and validated per-request JSON events."""
import json
import math
import queue
import subprocess
import tempfile
import threading
import time
import uuid


class PersistentError(RuntimeError): pass
class PersistentTimeout(PersistentError): pass


class PersistentWorker:
    def __init__(self,command,scratch_dir):
        self.stderr=tempfile.TemporaryFile(dir=scratch_dir)
        try:
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
                text=True,encoding='utf-8',bufsize=1)
        except BaseException: self.stderr.close();raise
        self.lines=queue.Queue();self.ready=False;self.closed=False;self.fields={}
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
    def _read(self):
        try:
            for line in self.process.stdout: self.lines.put(line)
        finally: self.lines.put(None)
    def close(self):
        if self.closed: return
        self.closed=True
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill();self.process.wait()
        self.reader.join(timeout=2)
        for pipe in (self.process.stdin,self.process.stdout,self.stderr):
            try: pipe.close()
            except OSError: pass
        self.ready=False
    def line(self,deadline):
        if self.closed: raise PersistentError('native worker was canceled')
        remaining=deadline-time.monotonic()
        if remaining<=0: raise PersistentTimeout('native request timed out')
        try: line=self.lines.get(timeout=min(1,remaining))
        except queue.Empty: return None
        if line is None:
            if self.closed: raise PersistentError('native worker was canceled')
            self.stderr.seek(0,2);end=self.stderr.tell();self.stderr.seek(max(0,end-4096))
            detail=self.stderr.read().decode('utf-8',errors='replace').strip()
            raise PersistentError('native worker exited: '+(detail or str(self.process.poll())))
        try: value=json.loads(line)
        except ValueError as error: raise PersistentError('invalid native worker JSON') from error
        if not isinstance(value,dict): raise PersistentError('invalid native worker event')
        return value


class PersistentRun:
    def __init__(self,worker,plan,timeout,eos_ids):
        self.worker=worker;self.process=worker.process;self.plan=plan
        self.request_id=uuid.uuid4().hex;self.deadline=time.monotonic()+timeout
        self.ids=[];self.fields={};self.eos=False;self.complete=False
        self.progress=-1;self.reused=None
        self.eos_ids=eos_ids
        self.seconds=-1
    def cancel(self): self.worker.close()
    def close(self):
        # Keep only a fully consumed, validated request. Cancellation, parser
        # failure and timeouts discard the worker and all KV/expert state.
        if not self.complete: self.worker.close()
    def events(self):
        w=self.worker
        while not w.ready:
            event=w.line(self.deadline)
            if event is None: yield None;continue
            if event!={'event':'ready','protocol':1}: raise PersistentError('native worker handshake failed')
            w.ready=True
        eos=self.eos_ids
        command='\t'.join((self.request_id,str(self.plan.predict),','.join(map(str,eos)),
            '1' if self.plan.cache_prompt else '0',','.join(map(str,self.plan.ids))))+'\n'
        try: w.process.stdin.write(command);w.process.stdin.flush()
        except OSError as error: raise PersistentError('cannot send request to native worker') from error
        while True:
            event=w.line(self.deadline)
            if event is None: yield None;continue
            if event.get('request_id')!=self.request_id: raise PersistentError('native worker request id mismatch')
            kind=event.get('event')
            if kind in ('prefill','token'):
                seconds=event.get('seconds')
                if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0 or seconds<self.seconds:
                    raise PersistentError('invalid native worker event timing')
                self.seconds=seconds
            if kind=='prefill':
                done,total,reused=(event.get(key) for key in ('done','total','reused'))
                if any(type(x) is not int for x in (done,total,reused)) or total!=len(self.plan.ids) or not 0<=reused<=done<=total or done<self.progress or self.ids:
                    raise PersistentError('invalid native prefill progress')
                if self.reused is not None and reused!=self.reused: raise PersistentError('native reuse count changed')
                self.progress=done;self.reused=reused;self.fields['prompt_reused_tokens']=str(reused)
                yield event
            elif kind=='token':
                token=event.get('id')
                if self.progress!=len(self.plan.ids) or type(event.get('index')) is not int or event['index']!=len(self.ids) or type(token) is not int or len(self.ids)>=self.plan.predict or self.eos:
                    raise PersistentError('invalid native worker token order')
                if type(event.get('eos')) is not bool or event['eos']!=(token in eos): raise PersistentError('invalid native worker EOS')
                self.ids.append(token);self.eos=event['eos'];yield event
            elif kind=='done':
                fields=event.get('fields')
                if event.get('ids')!=self.ids or not self.ids or not isinstance(fields,dict): raise PersistentError('native worker stream and summary disagree')
                if not self.eos and len(self.ids)!=self.plan.predict: raise PersistentError('native worker ended unexpectedly')
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in fields.values()): raise PersistentError('invalid native worker metrics')
                if fields.get('prompt_reused_tokens')!=self.reused or fields.get('prompt_evaluated_tokens')!=len(self.plan.ids)-self.reused:
                    raise PersistentError('native worker reuse summary disagrees')
                self.fields={key:str(value) for key,value in fields.items()};w.fields=dict(self.fields)
                self.complete=True;return
            elif kind=='error': raise PersistentError('native worker failed: '+str(event.get('message','unknown error')))
            else: raise PersistentError('unknown native worker event')
