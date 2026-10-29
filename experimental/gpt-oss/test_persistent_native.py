"""Independent NumPy checks of KV rollback, ring growth and native worker reuse."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from test_decoder import fixture,reference,CPU,TEMP

CHECKPOINT=Path(sys.argv.pop(1)).resolve()


class PersistentNativeTests(unittest.TestCase):
    def test_sliding_checkpoint_restore_and_context_growth_logits(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            model=Path(folder)/'model.gguf';out=Path(folder)/'checkpoint.f32'
            meta,w=fixture(model,dense_kind=8,expert_kind=39,yarn=True)
            result=subprocess.run([str(CHECKPOINT),str(model),str(out)],capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            sequences=[[0,1],[0,1,2,4,5,6,7,9],[0,1,2,4,5,6,7,9,3,0,2,8,10],
                [0,1,2,4,5,6,7,9,1,4,2],[0,1,3,5,8,1],[0,1,9,2]]
            expected=np.stack([reference(ids,meta,w)[-1] for ids in sequences])
            actual=np.fromfile(out,dtype='<f4').reshape(len(sequences),-1)
            np.testing.assert_allclose(actual,expected,atol=8e-5,rtol=8e-5)
            self.assertIn('expired_checkpoint_rejected=true',result.stdout)

    def test_worker_generation_repeated_appended_divergent_and_invalid_prompts(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as folder:
            model=Path(folder)/'model.gguf';meta,w=fixture(model,dense_kind=8,expert_kind=39,yarn=True,context=4096)
            original=[0,1,2,4,5,6,7,9,3]
            def predict(prompt,count):
                ids=list(prompt);result=[]
                for _ in range(count):
                    token=int(np.argmax(reference(ids,meta,w)[-1]));result.append(token);ids.append(token)
                return result
            first=predict(original,4)
            prompts=[original,original,original+first+[2,1],original[:-1]+[10],[0,10,8,3]]
            prompts += [[0,1,2,4,5,6,7,9,3]+[i%11 for i in range(n)] for n in range(2,42,2)]
            # Exercise chunk checkpoints as well as the per-request boundaries.
            prompts.append([i%11 for i in range(520)])
            commands=[];expected=[]
            for i,prompt in enumerate(prompts):
                count=4 if i<5 else 1;expected.append(predict(prompt,count))
                commands.append(f'r{i}\t{count}\t-\t1\t'+','.join(map(str,prompt)))
            commands+=['uncached\t4\t-\t0\t'+','.join(map(str,original)),
                'bad\t2\t-\t1\t999','afterbad\t2\t-\t1\t0,1,2','quit']
            command=[str(CPU),'--model',str(model),'--context','4096','--serve','--fast-cpu']
            if os.environ.get('GPT_OSS_GPU_PTX'):
                command+=['--expert-ptx',os.environ['GPT_OSS_GPU_PTX'],'--expert-cache-mib','1',
                    '--cpu-expert-workers','4','--group-hot-experts']
            result=subprocess.run(command,input='\n'.join(commands)+'\n',capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            events=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(events[0],{'event':'ready','protocol':1})
            done={e['request_id']:e for e in events if e['event']=='done'}
            for i,wanted in enumerate(expected):
                self.assertEqual(done[f'r{i}']['ids'],wanted,f'request {i}')
                self.assertLessEqual(done[f'r{i}']['fields']['checkpoints'],8)
                progress=[e for e in events if e.get('request_id')==f'r{i}' and e['event']=='prefill']
                self.assertEqual(progress[-1]['done'],len(prompts[i]))
                self.assertTrue(all(a['done']<=b['done'] for a,b in zip(progress,progress[1:])))
            self.assertEqual(done['r0']['fields']['prompt_reused_tokens'],0)
            self.assertEqual(done['r1']['fields']['prompt_reused_tokens'],len(original)-1)
            self.assertGreater(done['r2']['fields']['prompt_reused_tokens'],len(original)-1)
            self.assertEqual(done['r4']['fields']['prompt_reused_tokens'],0)
            self.assertEqual(done['uncached']['fields']['prompt_reused_tokens'],0)
            self.assertEqual(done['uncached']['ids'],first)
            self.assertTrue(any(e['event']=='error' and e['request_id']=='bad' for e in events))
            self.assertEqual(done['afterbad']['fields']['prompt_reused_tokens'],0)
            self.assertEqual(done['afterbad']['ids'],predict([0,1,2],2))


if __name__=='__main__': unittest.main()
