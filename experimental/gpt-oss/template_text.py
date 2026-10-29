"""Lightweight renderer for the downloaded GPT-OSS GGUF's Harmony template."""
import datetime
import json
from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


def compile_template(source):
    environment=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
    def fail(message): raise TemplateError(message)
    def tojson(value,ensure_ascii=False,indent=None,separators=None,sort_keys=False):
        return json.dumps(value,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
    environment.filters['tojson']=tojson
    environment.globals['raise_exception']=fail
    environment.globals['strftime_now']=lambda fmt:datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime(fmt)
    return environment.from_string(source)
