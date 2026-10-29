"""Validate OpenAI chat inputs and adapt them to the downloaded Harmony template."""
import json
import re

LEVELS=('low','medium','high')
NAME=re.compile(r'[A-Za-z0-9_.-]{1,128}\Z')


def json_object(value):
    if isinstance(value,str):
        value=json.loads(value,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite tool argument')))
    if not isinstance(value,dict): raise ValueError('function arguments must be a JSON object')
    return value


def check_refs(value):
    # Schema validation must never fetch URLs or read files supplied by clients.
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('$ref','$dynamicRef','$recursiveRef') and (not isinstance(item,str) or not item.startswith('#')):
                raise ValueError('tool schemas support only local fragment references')
            if key=='$id': raise ValueError('$id is not supported in tool schemas')
            check_refs(item)
    elif isinstance(value,list):
        for item in value: check_refs(item)


def prepare_tools(body):
    tools=body.get('tools',[])
    if tools is None: tools=[]
    if not isinstance(tools,list) or len(tools)>128: raise ValueError('tools must be a list of at most 128 functions')
    clean=[];validators={}
    if tools:
        from jsonschema import validators as schema_validators
        from referencing import Registry
        def no_retrieval(uri): raise ValueError('remote schema retrieval is disabled')
    for tool in tools:
        if not isinstance(tool,dict) or tool.get('type')!='function' or not isinstance(tool.get('function'),dict):
            raise ValueError('each tool must have type=function and a function definition')
        f=tool['function'];name=f.get('name')
        if not isinstance(name,str) or not NAME.fullmatch(name) or name in validators:
            raise ValueError('function names must be unique and contain only letters, digits, _, -, .')
        description=f.get('description','');schema=f.get('parameters',{'type':'object','properties':{}})
        if not isinstance(description,str) or type(f.get('strict',False)) is not bool:
            raise ValueError('invalid function description or strict flag')
        if not isinstance(schema,dict) or schema.get('type')!='object': raise ValueError('function parameters must be an object schema')
        check_refs(schema)
        cls=schema_validators.validator_for(schema);cls.check_schema(schema)
        # The template iterates properties/required; supply empty defaults.
        schema={**schema,'properties':schema.get('properties',{}),'required':schema.get('required',[])}
        validators[name]=cls(schema,registry=Registry(retrieve=no_retrieval))
        clean.append({'type':'function','function':{**f,'description':description,'parameters':schema}})
    choice=body.get('tool_choice','auto' if clean else 'none')
    if isinstance(choice,dict):
        f=choice.get('function');name=f.get('name') if isinstance(f,dict) else None
        if choice.get('type')!='function' or name not in validators: raise ValueError('tool_choice names an unavailable function')
        prefix=' to=functions.'+name+'<|channel|>commentary json<|message|>'
    elif choice in ('none','auto','required'):
        if choice=='required' and not clean: raise ValueError('tool_choice=required needs tools')
        prefix=' to=functions.' if choice=='required' else ''
    else: raise ValueError('tool_choice must be auto, none, required or a named function')
    if type(body.get('parallel_tool_calls',False)) is not bool: raise ValueError('parallel_tool_calls must be a boolean')
    if choice=='none': return [],{},''
    return clean,validators,prefix


def prepare_messages(messages):
    if not isinstance(messages,list) or not messages or len(messages)>256: raise ValueError('messages must contain 1..256 text messages')
    clean=[];pending=None;seen=set()
    for i,m in enumerate(messages):
        if not isinstance(m,dict) or m.get('role') not in ('system','developer','user','assistant','tool'):
            raise ValueError('supported roles: system, developer, user, assistant, tool')
        role=m['role']
        if role in ('system','developer') and i!=0: raise ValueError('system/developer message must be first for this template')
        if m.get('function_call'): raise ValueError('use tool_calls instead of legacy function_call')
        content=m.get('content')
        if content is None and role=='assistant' and m.get('tool_calls'): content=''
        if isinstance(content,list):
            if not all(isinstance(p,dict) and p.get('type')=='text' and isinstance(p.get('text'),str) for p in content):
                raise ValueError('only text content parts are supported')
            content=''.join(p['text'] for p in content)
        if not isinstance(content,str): raise ValueError('message content must be text')
        item={'role':role,'content':content}
        if pending and role!='tool': raise ValueError('a tool result must follow its assistant tool call')
        if role=='tool':
            if pending is None or m.get('tool_call_id')!=pending: raise ValueError('tool_call_id does not match the pending call')
            pending=None
        calls=m.get('tool_calls')
        if calls:
            if role!='assistant' or not isinstance(calls,list) or len(calls)!=1:
                raise ValueError('this Harmony template supports one tool call per assistant message')
            call=calls[0]
            if not isinstance(call,dict) or call.get('type')!='function' or not isinstance(call.get('function'),dict):
                raise ValueError('invalid assistant tool call')
            f=call['function'];name=f.get('name');cid=call.get('id')
            if not isinstance(name,str) or not NAME.fullmatch(name) or not isinstance(cid,str) or not cid or cid in seen:
                raise ValueError('invalid function name or duplicate/missing tool call id')
            arguments=json_object(f.get('arguments'))
            item['tool_calls']=[{'type':'function','id':cid,'function':{'name':name,'arguments':arguments}}]
            thinking=m.get('reasoning_content',m.get('thinking','')) or ''
            if not isinstance(thinking,str): raise ValueError('reasoning_content must be text')
            item['thinking']='\n'.join(x for x in (thinking,content) if x);item['content']=''
            pending=cid;seen.add(cid)
        clean.append(item)
    if pending: raise ValueError('missing tool result for the last assistant tool call')
    return clean


def reasoning(body):
    nested=body.get('reasoning',{})
    if nested is None: nested={}
    if not isinstance(nested,dict): raise ValueError('reasoning must be an object')
    effort=body.get('reasoning_effort',nested.get('effort','low'))
    if 'reasoning_effort' in body and 'effort' in nested and effort!=nested['effort']:
        raise ValueError('reasoning_effort and reasoning.effort disagree')
    if effort not in (*LEVELS,'none'): raise ValueError('reasoning_effort must be low, medium or high (none forces final text)')
    explicit='reasoning_effort' in body or 'effort' in nested
    return effort,explicit
