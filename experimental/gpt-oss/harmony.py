"""Incremental Harmony headers/channels to OpenAI content and tool-call deltas."""
import uuid
from chat_protocol import json_object
from stream_text import ByteLevelDecoder

MARKERS=('<|channel|>','<|message|>','<|start|>','<|end|>','<|return|>','<|call|>','<|endoftext|>')


class HarmonyAnswer:
    def __init__(self,tokenizer,validators,header='assistant',expose_reasoning=True):
        self.bytes=ByteLevelDecoder(tokenizer);self.validators=validators;self.header=header
        self.state='header';self.pending='';self.channel=None;self.current_tool=None
        self.done=False;self.called=False;self.expose_reasoning=expose_reasoning
        self.content='';self.reasoning='';self.tool_calls=[]
        if header.endswith('<|message|>'):
            self.header=header[:-len('<|message|>')]
            self.initial=self._open()
        else: self.initial=[]

    def _open(self):
        fields=self.header.replace('<|channel|>',' ').split()
        if not fields or fields[0]!='assistant': raise ValueError('engine generated a non-assistant Harmony role')
        recipients=[x[3:] for x in fields[1:] if x.startswith('to=')]
        if len(recipients)>1: raise ValueError('multiple Harmony recipients')
        channels=[x for x in fields[1:] if x in ('analysis','commentary','final')]
        if len(channels)!=1: raise ValueError('missing or invalid Harmony channel')
        self.channel=channels[0];self.state='body';self.current_tool=None
        if not recipients: return []
        recipient=recipients[0]
        if not recipient.startswith('functions.') or recipient[10:] not in self.validators:
            raise ValueError('engine called an unavailable function')
        if self.tool_calls: raise ValueError('only one tool call per native turn is supported')
        call={'id':'call_'+uuid.uuid4().hex[:24],'type':'function','function':{'name':recipient[10:],'arguments':''}}
        self.current_tool=call;self.tool_calls.append(call)
        return [{'tool_calls':[{'index':0,**call,'function':dict(call['function'])}]}]

    def _text(self,text):
        if not text: return []
        if self.state=='header':
            self.header+=text
            if len(self.header)>1024: raise ValueError('Harmony header exceeds 1024 characters')
            return []
        if self.state=='gap':
            if text.strip(): raise ValueError('expected next Harmony message')
            return []
        if self.current_tool is not None:
            self.current_tool['function']['arguments']+=text
            return [{'tool_calls':[{'index':0,'function':{'arguments':text}}]}]
        if self.channel=='analysis':
            self.reasoning+=text
            return [{'reasoning_content':text}] if self.expose_reasoning else []
        self.content+=text
        return [{'content':text}]

    def _marker(self,marker):
        if self.state=='header':
            if marker=='<|channel|>': self.header+=marker;return []
            if marker=='<|message|>': return self._open()
            raise ValueError('unexpected Harmony marker inside header')
        if self.state=='gap':
            if marker!='<|start|>': raise ValueError('expected Harmony start marker')
            self.state='header';self.header='';return []
        if marker=='<|call|>':
            if self.current_tool is None: raise ValueError('tool handoff without a function call')
            f=self.current_tool['function'];args=json_object(f['arguments'])
            self.validators[f['name']].validate(args)
            self.called=True;self.done=True
        elif marker in ('<|return|>','<|endoftext|>'):
            if self.current_tool is not None: raise ValueError('function call ended without handoff')
            if self.channel=='analysis': raise ValueError('analysis ended without a final answer')
            self.done=True
        elif marker=='<|end|>':
            if self.current_tool is not None: raise ValueError('function call ended without handoff')
            if self.channel=='final': self.done=True
            else: self.state='gap'
        else: raise ValueError('unexpected Harmony marker inside message')
        return []

    def _consume(self,text,final=False):
        result=self.initial;self.initial=[];self.pending+=text
        while self.pending and not self.done:
            hits=[(self.pending.find(m),m) for m in MARKERS if m in self.pending]
            if hits:
                index,marker=min(hits)
                result+=self._text(self.pending[:index]);self.pending=self.pending[index+len(marker):]
                result+=self._marker(marker)
            else:
                keep=0
                if not final:
                    for m in MARKERS:
                        for size in range(1,min(len(m),len(self.pending)+1)):
                            if self.pending.endswith(m[:size]): keep=max(keep,size)
                chunk=self.pending[:-keep] if keep else self.pending
                self.pending=self.pending[-keep:] if keep else ''
                result+=self._text(chunk);break
        return result

    def feed(self,token_id): return self._consume(self.bytes.feed(token_id))
    def finish(self): return self._consume(self.bytes.finish(),final=True)
