"""Compare lightweight rendering with the existing Transformers renderer."""
import os
from pathlib import Path
import unittest
from jinja2 import TemplateError
from jinja2.exceptions import SecurityError
from template_text import compile_template


class TemplateTests(unittest.TestCase):
    def test_reference_rendering(self):
        try: from transformers.utils.chat_template_utils import _compile_jinja_template
        except ImportError: self.skipTest('optional Transformers reference unavailable')
        sources=["{% for item in messages %}{{ item.role }}:{{ item.content }}\n{% endfor %}",
            "{{ tools|tojson }}|{{ {'text':'雨<>&'}|tojson }}|{{ strftime_now('%Y-%m-%d') }}",
            "{% for x in [1,2,3] %}{% if x==2 %}{% continue %}{% endif %}{{ x }}{% endfor %}"]
        if os.environ.get('GPT_OSS_TEMPLATE_FILE'):
            sources.append(Path(os.environ['GPT_OSS_TEMPLATE_FILE']).read_text(encoding='utf-8'))
        cases=[{'messages':[{'role':'system','content':'Answer clearly.'},{'role':'user','content':text}],
                'tools':None,'add_generation_prompt':True,'reasoning_effort':'low'} for text in ['2+2?','雨夜🌧️','literal <tag> & "quote"']]
        for source in sources:
            for kwargs in cases:
                old=_compile_jinja_template(source);new=compile_template(source)
                # Compare a fixed date independent of execution time/timezone.
                for template in (old,new): template.environment.globals['strftime_now']=lambda fmt:'2026-10-29'
                self.assertEqual(new.render(**kwargs),old.render(**kwargs))

    def test_sandbox_and_explicit_template_error(self):
        with self.assertRaises(TemplateError): compile_template("{{ raise_exception('invalid message') }}").render()
        with self.assertRaises(SecurityError): compile_template("{{ [].__class__.__mro__ }}").render()


if __name__=='__main__': unittest.main()
