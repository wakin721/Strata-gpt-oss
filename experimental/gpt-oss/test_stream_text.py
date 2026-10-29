"""Streaming UTF-8, Harmony filtering and a producer/consumer flush handshake."""
import json
import subprocess
import sys
import threading
import unittest
from tokenizers import Tokenizer,models,decoders,pre_tokenizers
from stream_text import StreamingAnswer,answer_from_raw,consume_stream
from harmony import HarmonyAnswer
from chat_protocol import prepare_tools,prepare_messages,reasoning

TOOL={'type':'function','function':{'name':'weather','parameters':{'type':'object',
    'properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False},'strict':True}}


class HarmonyTests(unittest.TestCase):
    def parse(self,raw,header='assistant',expose=True):
        tok=tokenizer();_,validators,_=prepare_tools({'tools':[TOOL]})
        renderer=HarmonyAnswer(tok,validators,header,expose);deltas=[]
        for token in tok.encode(raw,add_special_tokens=False).ids: deltas+=renderer.feed(token)
        deltas+=renderer.finish()
        return renderer,deltas

    def test_reasoning_final_unicode_and_no_control_leaks(self):
        raw='<|channel|>analysis<|message|>想一想🙂<|end|><|start|>assistant<|channel|>final<|message|>下雨。<|return|>'
        r,d=self.parse(raw)
        self.assertTrue(r.done);self.assertFalse(r.called)
        self.assertEqual(''.join(x.get('reasoning_content','') for x in d),'想一想🙂')
        self.assertEqual(''.join(x.get('content','') for x in d),'下雨。')
        self.assertNotIn('<|',json.dumps(d,ensure_ascii=False));self.assertNotIn('\ufffd',str(d))
        r,d=self.parse(raw,expose=False);self.assertNotIn('reasoning_content',str(d))

    def test_tool_streaming_both_header_orders_and_validation(self):
        for header in (' to=functions.weather<|channel|>commentary json','<|channel|>commentary to=functions.weather json'):
            raw=header+'<|message|>{"city":"台北🙂"}<|call|>'
            r,d=self.parse(raw)
            self.assertTrue(r.called);self.assertTrue(r.done)
            deltas=[c for x in d for c in x.get('tool_calls',[])]
            self.assertEqual(deltas[0]['function']['name'],'weather');self.assertEqual(deltas[0]['index'],0)
            self.assertEqual(''.join(x['function'].get('arguments','') for x in deltas),'{"city":"台北🙂"}')
            self.assertEqual(r.tool_calls[0]['function']['arguments'],'{"city":"台北🙂"}')
        r,d=self.parse('{"city":"台北"}<|call|>',header='assistant to=functions.weather<|channel|>commentary json<|message|>')
        self.assertTrue(r.called)
        r,d=self.parse('weather<|channel|>commentary json<|message|>{"city":"台北"}<|call|>',header='assistant to=functions.')
        self.assertTrue(r.called)

    def test_analysis_then_tool_and_length_truncation(self):
        raw='<|channel|>analysis<|message|>查天气<|end|><|start|>assistant to=functions.weather<|channel|>commentary json<|message|>{"city":"台北"}<|call|>'
        r,d=self.parse(raw);self.assertTrue(r.called);self.assertEqual(r.reasoning,'查天气')
        r,d=self.parse(' to=functions.weather<|channel|>commentary json<|message|>{"city":')
        self.assertFalse(r.done);self.assertFalse(r.called)
        self.assertEqual(r.tool_calls[0]['function']['arguments'],'{"city":')
        r,d=self.parse('<|channel|>ana');self.assertFalse(r.done);self.assertEqual(d,[])

    def test_invalid_tool_output_never_completes_successfully(self):
        for raw in (' to=functions.missing<|channel|>commentary<|message|>{}<|call|>',
            ' to=functions.weather<|channel|>commentary<|message|>{bad}<|call|>',
            ' to=functions.weather<|channel|>commentary<|message|>{"city":123}<|call|>',
            ' to=functions.weather<|channel|>commentary<|message|>[]<|call|>',
            ' to=functions.weather<|channel|>commentary<|message|>{"city":"x"}<|return|>',
            '<|channel|>analysis<|message|>thinking<|return|>'):
            with self.assertRaises(Exception): self.parse(raw)

    def test_history_ids_and_json_are_preserved(self):
        call={'id':'call_1','type':'function','function':{'name':'weather','arguments':'{"city":"台北"}'}}
        history=[{'role':'user','content':'天气?'},{'role':'assistant','content':None,'reasoning_content':'查一下','tool_calls':[call]},
            {'role':'tool','tool_call_id':'call_1','content':'下雨'}]
        result=prepare_messages(history)
        self.assertEqual(result[1]['tool_calls'][0]['function']['arguments'],{'city':'台北'})
        self.assertEqual(result[1]['thinking'],'查一下')
        for bad in (history[:-1],[history[-1]],history[:-1]+[{**history[-1],'tool_call_id':'wrong'}]):
            with self.assertRaises(ValueError): prepare_messages(bad)

    def test_reasoning_and_offline_schema_inputs(self):
        for level in ('low','medium','high'):
            self.assertEqual(reasoning({'reasoning_effort':level}),(level,True))
            self.assertEqual(reasoning({'reasoning':{'effort':level}}),(level,True))
        with self.assertRaises(ValueError): reasoning({'reasoning_effort':'ultra'})
        with self.assertRaises(ValueError): reasoning({'reasoning_effort':'low','reasoning':{'effort':'high'}})
        for schema in ({'type':'object','$ref':'https://example.invalid/schema'}, {'type':'object','$id':'file:///tmp/schema'}):
            with self.assertRaises(ValueError): prepare_tools({'tools':[{'type':'function','function':{'name':'x','parameters':schema}}]})
        schema={'type':'object','properties':{'city':{'$ref':'#/$defs/city'}},'$defs':{'city':{'type':'string'}}}
        _,v,_=prepare_tools({'tools':[{'type':'function','function':{'name':'x','parameters':schema}}]})
        v['x'].validate({'city':'台北'})


def tokenizer():
    # One token per byte forces multi-token Chinese/emoji, independent of a
    # downloaded model or BPE merges.
    alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())
    result=Tokenizer(models.BPE(vocab={char:i for i,char in enumerate(alphabet)},merges=[]))
    result.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    result.decoder=decoders.ByteLevel()
    result.add_special_tokens(['<|channel|>','<|message|>','<|return|>','<|endoftext|>','<|call|>','<|end|>','<|start|>'])
    return result


class StreamTests(unittest.TestCase):
    def test_split_unicode_controls_and_literal_prefixes(self):
        tok=tokenizer()
        cases=[('雨夜，灯光🌧️。é\0',False,'雨夜，灯光🌧️。é\0'),('雨夜<|return|>ignored',False,'雨夜'),
            ('<|channel|>analysis<|message|>secret<|channel|>final<|message|>雨夜🌧️<|return|>',True,'雨夜🌧️'),
            ('<|channel|>analysis<|message|>secret',True,'[No final channel was reached within the token limit.]'),
            ('ends with <|ret',False,'ends with <|ret'),
            ('雨夜<|end|><|start|>assistant<|channel|>analysis<|message|>secret',False,'雨夜'),
            ('<|channel|>final<|message|>雨夜<|end|><|start|>assistant<|channel|>final<|message|>different',True,'雨夜')]
        for text,thinking,expected in cases:
            renderer=StreamingAnswer(tok,thinking)
            chunks=[renderer.feed(i) for i in tok.encode(text,add_special_tokens=False).ids]
            chunks.append(renderer.finish())
            self.assertEqual(''.join(chunks),expected)
            self.assertEqual(answer_from_raw(text,thinking),expected)
            self.assertEqual(renderer.raw,text)
            self.assertNotIn('\ufffd',''.join(chunks))
            if thinking: self.assertNotIn('secret',''.join(chunks))
        # A literal replacement character and an incomplete UTF-8 sequence.
        for ids in [tok.encode('\ufffd',add_special_tokens=False).ids,[tok.token_to_id('Ã')]]:
            renderer=StreamingAnswer(tok)
            text=''.join(renderer.feed(i) for i in ids)+renderer.finish()
            self.assertEqual(text,tok.decode(ids,skip_special_tokens=False))

    def test_text_arrives_before_child_is_allowed_to_finish(self):
        tok=tokenizer();ids=tok.encode('雨夜<|return|>',add_special_tokens=False).ids
        producer='''import json,sys
ids=json.loads(sys.argv[1])
for index,token in enumerate(ids):
 print(json.dumps({'event':'token','index':index,'id':token,'seconds':index*.01,'eos':index==len(ids)-1}),flush=True)
 if index==2:
  if sys.stdin.readline().strip()!='visible': sys.exit(2)
print('predicted_token_ids='+','.join(map(str,ids)),flush=True)
'''
        process=subprocess.Popen([sys.executable,'-u','-c',producer,json.dumps(ids)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1)
        timer=threading.Timer(5,process.kill);timer.start();parts=[]
        def emit(text):
            if not parts:
                self.assertIsNone(process.poll(),'text arrived only after producer exit')
                process.stdin.write('visible\n');process.stdin.flush()
            parts.append(text)
        try:
            fields,events=consume_stream(process,StreamingAnswer(tok),emit)
            self.assertEqual(''.join(parts),'雨夜')
            self.assertEqual(len(events),len(ids))
            self.assertEqual(fields['predicted_token_ids'],','.join(map(str,ids)))
        finally:
            timer.cancel()
            if process.poll() is None: process.kill()
            process.wait()
            for pipe in (process.stdin,process.stdout,process.stderr): pipe.close()


if __name__=='__main__': unittest.main()
