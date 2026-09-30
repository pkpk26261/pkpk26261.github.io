#!/usr/bin/env python3
"""Verify generated pages, source preservation, and local resource integrity."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,hashlib,zipfile,re
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'content/site.json').read_text())
class Links(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=set();self.h1=0;self.title=0
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if attrs.get('id'):self.ids.add(attrs['id'])
  if tag=='h1':self.h1+=1
  if tag=='title':self.title+=1
  for key in ('src','href'):
   if attrs.get(key):self.refs.append((tag,key,attrs[key]))
def parse(html):
 p=Links();p.feed(html);return p
pages=sorted(p for p in ROOT.rglob('*.html') if p.name in ('index.html','404.html') and 'lib' not in p.relative_to(ROOT).parts)
errors=[];missing=[]
for p in pages:
 parsed=parse(p.read_text())
 if parsed.h1!=1:errors.append(f'{p.relative_to(ROOT)} has {parsed.h1} h1 headings')
 if parsed.title!=1:errors.append(f'{p.relative_to(ROOT)} has {parsed.title} titles')
 for tag,key,ref in parsed.refs:
  u=urlsplit(ref)
  if u.scheme or u.netloc or not u.path:continue
  target=ROOT/unquote(u.path).lstrip('/') if u.path.startswith('/') else p.parent/unquote(u.path)
  if target.is_dir():target=target/'index.html'
  if not target.exists():missing.append((str(p.relative_to(ROOT)),ref))
  elif u.fragment and target.suffix=='.html':
   ids=parsed.ids if target==p else parse(target.read_text()).ids
   if unquote(u.fragment) not in ids:errors.append(f'Invalid anchor: {p.relative_to(ROOT)} -> {ref}')
for a in data['articles']:
 body=ROOT/a['body']
 if hashlib.sha256(body.read_bytes()).hexdigest()!=data.get('current_body_hashes',data['legacy_body_hashes'])[a['id']]:errors.append('Preserved article source changed: '+a['title'])
 if not (ROOT/a['path']).exists():errors.append('Legacy URL missing: '+a['url'])
# Compare missing original article references against their exact preserved HTML.
legacy_refs=set()
for a in data['articles']:
 for tag,key,ref in parse((ROOT/a['body']).read_text()).refs:
  u=urlsplit(ref)
  if u.scheme or u.netloc or not u.path:continue
  target=ROOT/unquote(u.path).lstrip('/') if u.path.startswith('/') else (ROOT/a['path']).parent/unquote(u.path)
  if target.is_dir():target=target/'index.html'
  if not target.exists():legacy_refs.add((a['path'],ref))
new_missing=[(p,r) for p,r in missing if (p,r) not in legacy_refs]
errors.extend('New missing local reference: '+p+' -> '+r for p,r in new_missing)
report={'pages_checked':len(pages),'articles_checked':len(data['articles']),'original_article_records':len(data['legacy_body_hashes']),'legacy_routes':len(data['pages'])+len(data['legacy_body_hashes']),'search_entries':len(json.loads((ROOT/'assets/search-index.json').read_text())),'new_errors':errors,'existing_missing_references':sorted(legacy_refs),'backup':data['backup']}
print(json.dumps(report,ensure_ascii=False,indent=2))
(ROOT/'design/static-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
raise SystemExit(bool(errors))
