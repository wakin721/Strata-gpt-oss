"""Probe integration and failure cases, with tiny synthetic GGUF fixtures."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("fetch_expert",Path(__file__).with_name("fetch_expert.py"))
fetch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
PROBE=Path(sys.argv.pop(1)).resolve() if len(sys.argv)>1 else None
TEST_TEMP=Path.cwd()/"build-gpt-oss-test-temp"
TEST_TEMP.mkdir(exist_ok=True)


def temp_directory():
    return tempfile.TemporaryDirectory(dir=TEST_TEMP)


class ProbeTests(unittest.TestCase):
    def fixture(self,path,kind=39,arch="gpt-oss"):
        meta={"general.architecture":arch,"gpt-oss.embedding_length":64,
              "gpt-oss.expert_feed_forward_length":32,"gpt-oss.block_count":1,"gpt-oss.expert_count":2}
        tensors=[]
        for projection,cols,rows in (("gate",64,32),("up",64,32),("down",32,64)):
            size=cols*rows*2*4 if kind==0 else cols//32*17*rows*2
            tensors.append((f"blk.0.ffn_{projection}_exps.weight",[cols,rows,2],kind,bytes(size)))
            bias=[1.0 if projection=="down" else 0.0]*(rows*2)
            tensors.append((f"blk.0.ffn_{projection}_exps.bias",[rows,2],0,struct.pack("<"+"f"*len(bias),*bias)))
        fetch.write_fixture(path,meta,tensors)

    def run_probe(self,*args):
        return subprocess.run([str(PROBE),*map(str,args)],capture_output=True,text=True,timeout=15)

    def test_real_file_layouts_and_expert_offsets(self):
        with temp_directory() as directory:
            for kind in (0,39):
                path=Path(directory)/f"fixture-{kind}.gguf"
                self.fixture(path,kind)
                for expert in (0,1):
                    result=self.run_probe("--expert",path,0,expert)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn("generation_supported=false",result.stdout)
                    self.assertIn("expert_output_sum=64",result.stdout)

    def test_architecture_and_index_guards(self):
        with temp_directory() as directory:
            path=Path(directory)/"wrong.gguf"
            self.fixture(path,arch="qwen4exp")
            self.assertNotEqual(self.run_probe("--inspect",path).returncode,0)
            path=Path(directory)/"valid.gguf"
            self.fixture(path)
            for layer,expert in ((1,0),(0,2),("-1",0),(0,"0junk")):
                self.assertNotEqual(self.run_probe("--expert",path,layer,expert).returncode,0)

    def test_truncated_weight_file(self):
        with temp_directory() as directory:
            path=Path(directory)/"short.gguf"
            self.fixture(path)
            with path.open("r+b") as output:
                output.truncate(path.stat().st_size-100)
            self.assertNotEqual(self.run_probe("--expert",path,0,1).returncode,0)

    def test_directory_parser_and_exclusive_output(self):
        with temp_directory() as directory:
            path=Path(directory)/"fixture.gguf"
            self.fixture(path)
            metadata,tensors,data_start=fetch.Header(path.read_bytes()).parse()
            self.assertEqual(metadata["gpt-oss.embedding_length"],64)
            self.assertEqual(tensors["blk.0.ffn_gate_exps.weight"][0],[64,32,2])
            self.assertEqual(data_start%32,0)
            with self.assertRaises(FileExistsError):
                self.fixture(path)
            with self.assertRaises(EOFError):
                fetch.Header(path.read_bytes()[:100]).parse()


if __name__=="__main__":
    unittest.main()
