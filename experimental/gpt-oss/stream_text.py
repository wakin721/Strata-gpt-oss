"""ByteLevel UTF-8 decoding and final-channel display for the local text probe."""
import codecs
import json
import time

FINAL='<|channel|>final<|message|>'
STOPS=('<|return|>','<|endoftext|>','<|call|>','<|end|>')
CONTROLS=('<|channel|>','<|start|>','<|message|>')


def answer_from_raw(raw,thinking=False,partial=False):
    if thinking:
        if FINAL not in raw: return '' if partial else '[No final channel was reached within the token limit.]'
        answer=raw.split(FINAL,1)[1]
    else: answer=raw[len(FINAL):] if raw.startswith(FINAL) else raw
    # Display one final message. A later generated role/header must never
    # replace already displayed text or expose a new analysis channel.
    for stop in (*STOPS,*CONTROLS): answer=answer.split(stop,1)[0]
    if partial:
        # Hide incomplete control headers until a complete final header arrives.
        for marker in (FINAL,*STOPS,*CONTROLS):
            for size in range(min(len(answer),len(marker)-1),0,-1):
                if answer.endswith(marker[:size]):
                    answer=answer[:-size];break
    return answer


class ByteLevelDecoder:
    """Decode token bytes without losing UTF-8 characters split across tokens."""
    def __init__(self,tokenizer):
        config=json.loads(tokenizer.to_str())
        if config.get('decoder',{}).get('type')!='ByteLevel':
            raise ValueError('streaming currently requires the GPT-OSS ByteLevel decoder')
        # Inverse of the ByteLevel alphabet. Bytes outside the visible ranges
        # map to consecutive codepoints starting at U+0100.
        visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
        alphabet={chr(value):value for value in visible}
        missing=0
        for value in range(256):
            if value not in visible:
                alphabet[chr(256+missing)]=value;missing+=1
        self.alphabet=alphabet
        self.added={token['id']:token['content'] for token in config.get('added_tokens',[])}
        self.tokenizer=tokenizer
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace')
    def feed(self,token_id):
        if token_id in self.added: data=self.added[token_id].encode('utf-8')
        else:
            token=self.tokenizer.id_to_token(token_id)
            if token is None: raise ValueError(f'invalid streamed token ID: {token_id}')
            data=bytes(self.alphabet[char] for char in token)
        return self.decoder.decode(data,final=False)
    def finish(self): return self.decoder.decode(b'',final=True)


class StreamingAnswer:
    def __init__(self,tokenizer,thinking=False):
        self.bytes=ByteLevelDecoder(tokenizer)
        self.thinking=thinking
        self.raw=''
        self.printed=''

    def _update(self,text,partial):
        self.raw+=text
        answer=answer_from_raw(self.raw,self.thinking,partial=partial)
        if not answer.startswith(self.printed):
            raise ValueError('generated channel change would revise text already streamed')
        delta=answer[len(self.printed):]
        self.printed=answer
        return delta

    def feed(self,token_id):
        return self._update(self.bytes.feed(token_id),partial=True)

    def finish(self):
        return self._update(self.bytes.finish(),partial=False)


def consume_stream(process,renderer,emit):
    """Consume flushed token lines while the child is running; return metrics."""
    lines=[]; events=[]
    for line in process.stdout:
        if line.startswith('{'):
            event=json.loads(line)
            if event.get('event')!='token' or event.get('index')!=len(events):
                raise ValueError('invalid/out-of-order native token event')
            event['received_monotonic']=time.perf_counter()
            events.append(event)
            delta=renderer.feed(event['id'])
            if delta: emit(delta)
        else: lines.append(line)
    process.wait()
    if process.returncode: raise RuntimeError(f'native engine exited with {process.returncode}')
    fields=dict(line.strip().split('=',1) for line in lines if '=' in line)
    ids=[int(value) for value in fields.get('predicted_token_ids','').split(',') if value]
    if ids!=[event['id'] for event in events]: raise ValueError('stream and final token IDs disagree')
    tail=renderer.finish()
    if tail: emit(tail)
    return fields,events
