import json, hashlib, zipfile, subprocess, time, re
from pathlib import Path
from datetime import datetime, timezone

OUT = Path('cleo-v6-15')
OUT.mkdir(exist_ok=True)
START = time.monotonic()
DEADLINE = datetime.fromisoformat('2026-10-13T14:05:07.802624+00:00')
def now(): return datetime.now(timezone.utc).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def save(n, x): (OUT/n).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def load(z,n): return json.loads(z.read(n))
gatez = zipfile.ZipFile('inputs/KAGGLE-62-recovery-v15.zip')
oldz = zipfile.ZipFile('inputs/kaggle49-v6-12-cleo-issue-identities.zip')
bindings=[]
gate_parts={}
for n in ['global-manifest.json','dedup-and-gates.json','body-prescreen-exclusions.json','owner-queue.json','research-source-registry.json','cleo614-next-queue.json']:
    b=gatez.read(n); gate_parts[n]=json.loads(b)
    bindings.append({'archive':'KAGGLE-62-recovery-v15.zip','member':n,'bytes':len(b),'sha256':sha(b)})
    save('input-'+n,gate_parts[n])
for n in ['manifest.json','next-queue.json','selection.json','index-response.json']:
    b=oldz.read(n); bindings.append({'archive':'kaggle49-v6-12-cleo-issue-identities.zip','member':n,'bytes':len(b),'sha256':sha(b)})
old=load(oldz,'manifest.json')
known_text=json.dumps(gate_parts)+json.dumps(old)
excluded=set(re.findall(r'python-poetry/cleo#\d+',known_text)) | {'python-poetry/cleo#454','python-poetry/cleo#341','python-poetry/cleo#84','python-poetry/cleo#231','python-poetry/cleo#366','python-poetry/cleo#359','python-poetry/cleo#446'}
nodes=load(oldz,'index-response.json')['data']['search']['nodes']
selected=[x for x in nodes if f"python-poetry/cleo#{x['number']}" not in excluded]
save('input-bindings.json',{'as_of':now(),'archives':[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in Path('inputs').glob('*')], 'selected_members':bindings,'coverage':'Selected available metadata only; no hidden/dev/sealed/root clearance'})
save('selection.json',{'cache_source':'v6.12 index-response.json; only issue identities previously listed, not timeline-qualified','known_ids_excluded':sorted(excluded),'selected':[x['number'] for x in selected[:30]],'remaining':[x['number'] for x in selected[30:]],'index_refresh_calls':0})
LOG=[]
def call(name, query):
    if time.monotonic()-START>1500 or datetime.now(timezone.utc)>=DEADLINE: raise RuntimeError('TIME_LIMIT')
    request={'query':query}; save(name+'-request.json',request)
    # Exclusive pre-request journal: any timeout/error consumes this attempt.
    with (OUT/(name+'-attempt.json')).open('x',encoding='utf-8') as f: json.dump({'started_at':now(),'timeout_seconds':30,'automatic_retries':0},f)
    t=time.monotonic()
    try:
        p=subprocess.run(['gh','api','graphql','--input',str(OUT/(name+'-request.json'))],capture_output=True,timeout=30)
    except subprocess.TimeoutExpired:
        LOG.append({'name':name,'status':'TIMEOUT','wall_seconds':time.monotonic()-t}); save('requests.json',LOG); raise RuntimeError('TIMEOUT')
    status={'name':name,'completed_at':now(),'exit_code':p.returncode,'wall_seconds':time.monotonic()-t,'response_bytes':len(p.stdout),'response_sha256':sha(p.stdout),'timeout_seconds':30,'automatic_retries':0}
    LOG.append(status);save('requests.json',LOG)
    # stderr can contain authentication material; record only type/status.
    if p.returncode: status['error_type']='GH_EXIT_NONZERO';save('requests.json',LOG);raise RuntimeError('GH_EXIT_NONZERO')
    data=json.loads(p.stdout)
    if data.get('errors'): status['error_type']='GRAPHQL_SEMANTIC_ERROR';save('requests.json',LOG);raise RuntimeError('GRAPHQL_SEMANTIC_ERROR')
    save(name+'-response.json',data)
    return data['data']['repository']

pin_cache={}
for r in old['records']:
    for p in r['candidate_pin_bindings']: pin_cache[p['pr']]=p
records=[]; batches=[]
failure=None
try:
  for bi in range(3):
    batch=selected[bi*10:(bi+1)*10]
    if not batch: break
    fields=[]
    for x in batch:
      n=x['number']
      fields.append('i%d:issue(number:%d){id number url repository{nameWithOwner} timelineItems(first:100,itemTypes:[CLOSED_EVENT]){totalCount pageInfo{hasNextPage endCursor} nodes{... on ClosedEvent{id createdAt closer{__typename ... on PullRequest{id databaseId number url repository{nameWithOwner}} ... on Commit{oid repository{nameWithOwner}}}}}}}'%(n,n))
    events=call('batch%d-events'%(bi+1),'{repository(owner:"python-poetry",name:"cleo"){'+ ' '.join(fields)+'}}')
    prs=set(); br=[]
    for x in batch:
      n=x['number']; e=events['i'+str(n)]
      if not e or e['number']!=n or e['url']!=x['url'] or e['repository']['nameWithOwner']!='python-poetry/cleo': raise RuntimeError('IDENTITY_MISMATCH')
      timeline=e['timelineItems']; good=[]
      for c in timeline['nodes']:
        closer=c.get('closer')
        if closer and closer.get('repository',{}).get('nameWithOwner')=='python-poetry/cleo' and closer['__typename']=='PullRequest' and c['createdAt']<'2026-01-01T00:00:00Z':
          good.append(c); prs.add(closer['number'])
      r={'repo_issue':f'python-poetry/cleo#{n}','identity':x,'closed_timeline':timeline,'same_repo_pre_cutoff_pr_events':good,'candidate_pin_bindings':[], 'role':'RESEARCH_SOURCE_QUALIFICATION_METADATA_ONLY','effective_split':'UNASSIGNED','body_read':False,'source_snapshot_acquired':False,'behavior_contract':'UNKNOWN_NOT_READ','text_use':'UNKNOWN','root':'UNKNOWN','actor_authorized':False,'training_released':False}
      br.append(r)
    newprs=sorted(prs-set(pin_cache))
    if newprs:
      fields=['p%d:pullRequest(number:%d){id databaseId number url mergedAt mergeCommit{oid tree{oid} parents(first:10){totalCount pageInfo{hasNextPage endCursor} nodes{oid tree{oid}}}}}'%(p,p) for p in newprs]
      pins=call('batch%d-pins'%(bi+1),'{repository(owner:"python-poetry",name:"cleo"){'+ ' '.join(fields)+'}}')
      bases={}
      for p in newprs:
        v=pins['p'+str(p)]
        if not v or v['number']!=p: raise RuntimeError('PR_IDENTITY_MISMATCH')
        m=v.get('mergeCommit')
        if not v.get('mergedAt') or v['mergedAt']>='2026-01-01T00:00:00Z' or not m or not m['parents']['nodes']:
          pin_cache[p]={'pr':p,'status':'UNKNOWN_NO_ELIGIBLE_MERGE_PARENT'};continue
        parent=m['parents']['nodes'][0]
        pin_cache[p]={'pr':p,'pr_url':v['url'],'candidate_first_parent':parent['oid'],'candidate_base_tree':parent['tree']['oid'],'landing':m['oid'],'landing_tree':m['tree']['oid'],'merged_at':v['mergedAt'],'broken':'NOT_VERIFIED'}
        bases[p]=parent['oid']
      if bases:
        fields=['p%d:object(expression:"%s:LICENSE"){... on Blob{oid byteSize text}}'%(p,b) for p,b in bases.items()]
        ls=call('batch%d-licenses'%(bi+1),'{repository(owner:"python-poetry",name:"cleo"){'+ ' '.join(fields)+'}}')
        for p,b in bases.items():
          v=ls['p'+str(p)]
          if not v or v.get('text') is None: pin_cache[p]['license_status']='UNKNOWN';continue
          raw=v['text'].encode('utf-8'); oid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
          if oid!=v['oid'] or len(raw)!=v['byteSize']:raise RuntimeError('LICENSE_BYTE_BINDING_MISMATCH')
          filename='license-'+oid+'.txt';(OUT/filename).write_bytes(raw)
          pin_cache[p].update({'license_blob':oid,'license_bytes':len(raw),'license_sha256':sha(raw),'license_file':filename,'git_blob_and_length_verified':True,'license_url':f'https://github.com/python-poetry/cleo/blob/{b}/LICENSE'})
    for r in br:
      for e in r['same_repo_pre_cutoff_pr_events']:
        p=pin_cache[e['closer']['number']].copy(); r['candidate_pin_bindings'].append(p)
        lf=p.get('license_file')
        if lf and not (OUT/lf).exists(): (OUT/lf).write_bytes(oldz.read(lf))
      r['known_relationship_matches']=sorted({s for p in r['candidate_pin_bindings'] for s in [str(p.get('candidate_first_parent','')),str(p.get('landing','')),str(p.get('candidate_base_tree','')),f"python-poetry/cleo#{p['pr']}"] if s and s in known_text})
      r['remaining_gaps']=['initial behavior/text-use scope not granted','broken and trusted acceptance unverified','root/family and retained coverage incomplete','candidate source permission not implied']
    records.extend(br)
    batches.append({'batch':bi+1,'metadata_issue_count':len(br),'issues_with_same_repo_pr_closer':sum(bool(r['same_repo_pre_cutoff_pr_events']) for r in br),'pr_events':sorted(prs),'new_pr_pin_queries':newprs,'new_materials':0,'direction':'Previously index-listed issues with genuinely unexamined closure identities; not repeated PR/template sweep'})
    save('manifest.json',{'schema':'kaggle49-cleo-metadata/6.15','as_of':now(),'records':records})
    save('batches.json',batches)
except Exception as e:
  failure=type(e).__name__+':'+str(e)
  save('failure.json',{'at':now(),'failure':failure,'retry':False})
save('counts.json',{'completed_metadata_issues':len(records),'batches':len(batches),'network_requests':len(LOG),'ordinary_new_packages':0,'research_preparation_packages':0,'source_ready':0,'independent_roots':0,'environment_function_runs':0,'training_release':0,'trajectories':0,'ordinary_cumulative_increment':15,'remaining_to100_at_least':85,'failure':failure})
save('next-queue.json',{'metadata_only_unqueried_closure_ids':[f"python-poetry/cleo#{x['number']}" for x in selected[len(records):]],'completed_no_auto_requery':[r['repo_issue'] for r in records], 'excluded':sorted(excluded),'no_body_or_source_permission':True,'remaining_direction':'Use next unexamined closure identities, or obtain concrete neutral research scope for newly bound pairs; do not repeat empty closers','index_cursor_preserved':load(oldz,'selection.json')['pageInfo']})
save('coverage.json',{'cutoff_exclusive':'2026-01-01T00:00:00Z','scope':'Issue metadata, closed-event closer identity, PR merge parent/tree, exact LICENSE blob only','not_acquired':['title','body','comments','PR body','patch','diff','commit message','future tests','source snapshots','foreign repository content'],'gate':'v15 selected metadata plus v6.12 detailed identities and v6.14 stops; no complete hidden clearance','two_zero_material_batches_direction':'Each batch moved to different unexamined issue closure IDs; no failed/empty identity was requeried','historical_license_not_text_training_right':True})
print(json.dumps({'completed':len(records),'batches':batches,'calls':len(LOG),'failure':failure}))
