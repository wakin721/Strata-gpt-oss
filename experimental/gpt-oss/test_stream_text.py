"""Streaming UTF-8, Harmony filtering and a producer/consumer flush handshake."""
import json
import subprocess
import sys
import threading
import unittest
from tokenizers import Tokenizer,models,decoders,pre_tokenizers
from stream_text import StreamingAnswer,answer_from_raw,consume_stream


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
