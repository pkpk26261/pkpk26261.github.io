#!/usr/bin/env python3
"""Import existing Hexo output once; do not run against redesigned pages."""
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
from urllib.parse import unquote, urlsplit, quote
from datetime import datetime
import xml.etree.ElementTree as ET
import json, re, zipfile, hashlib
ROOT = Path(__file__).resolve().parents[1]
class Body(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.html=html; self.lines=[0]; self.start=None; self.end=None; self.depth=0
        for m in re.finditer('\n',html): self.lines.append(m.end())
        self.feed(html)
    def position(self):
        line,col=self.getpos(); return self.lines[line-1]+col
    def handle_starttag(self,t,a):
        if t!='div': return
        if self.start is not None and self.end is None: self.depth+=1
        elif self.start is None and 'post-body' in dict(a).get('class','').split():
            self.start=self.position()+len(self.get_starttag_text()); self.depth=1
    def handle_endtag(self,t):
        if t=='div' and self.start is not None and self.end is None:
            self.depth-=1
            if self.depth==0: self.end=self.position()
    def value(self):
        assert self.start is not None and self.end is not None
        return self.html[self.start:self.end]
class Text(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self,d): self.parts.append(d)
def text(html):
    p=Text(); p.feed(html); return re.sub(r'\s+',' ',' '.join(p.parts)).strip()
def links(html, prefix):
    return [{'name':text(m.group(2)).strip().removeprefix('# '),'url':unescape(m.group(1))} for m in re.finditer(r'<a\b[^>]*href="('+prefix+r'[^"]+)"[^>]*>(.*?)</a>',html,re.S)]
def canonical(html):
    return urlsplit(re.search(r'<link rel="canonical" href="([^"]+)"',html).group(1)).path

def main():
    destination=ROOT/'content/site.json'
    if destination.exists(): raise SystemExit('Content already imported. Use build_site.py to rebuild safely.')
    originals=sorted(p for p in ROOT.rglob('index.html') if 'lib' not in p.relative_to(ROOT).parts)
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    backup=ROOT.parent/f'pkpk26261-blog-original-{stamp}.zip'
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
        for p in originals: z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'search.xml','search.xml'); z.write(ROOT/'css/main.css','css/main.css')
    entries=[]; pages=[]; source_hashes={}
    for entry in ET.parse(ROOT/'search.xml').getroot():
        url=entry.findtext('url'); path=ROOT/unquote(url).lstrip('/')/'index.html'
        html=path.read_text(); body=Body(html).value()
        key=hashlib.sha256(url.encode()).hexdigest()[:12]
        (ROOT/f'content/articles/{key}.html').write_text(body)
        source_hashes[key]=hashlib.sha256(body.encode()).hexdigest()
        created=re.search(r'<time[^>]*itemprop="dateCreated datePublished"[^>]*datetime="([^"]+)"',html)
        modified=re.search(r'<time[^>]*itemprop="dateModified"[^>]*datetime="([^"]+)"',html)
        cats=list({v['url']:v for v in links(html.split('class="post-body"')[0],'/categories/')}.values())
        tags=list({v['url']:v for v in links(html,'/tags/') if 'rel="tag"' in html[html.find(v['url']):html.find(v['url'])+120]}.values())
        # XML has complete tag names; derive canonical tag URL from existing markup.
        tags=[{'name':n.text,'url':next((v['url'] for v in links(html,'/tags/') if v['name']==n.text),'/tags/'+quote(n.text,safe='')+'/')} for n in entry.findall('tags/tag')]
        entries.append({'id':key,'title':entry.findtext('title'),'url':url,'path':str(path.relative_to(ROOT)), 'date':created.group(1)[:10] if created else url[1:11].replace('/','-'),'updated':modified.group(1)[:10] if modified else None,'categories':cats,'tags':tags,'text':text(body),'body':f'content/articles/{key}.html'})
    for p in originals:
        html=p.read_text(); url=canonical(html)
        if any(unquote(a['url'])==unquote(url) for a in entries): continue
        listed=links(html,r'/(?:2019|2020|2022|2023)/')
        body=None
        if url in ('/about/','/pen/'):
            key=url.strip('/'); body=f'content/{key}.html'; (ROOT/body).write_text(Body(html).value())
        pages.append({'path':str(p.relative_to(ROOT)),'url':url,'title':text(re.search(r'<title>(.*?)</title>',html,re.S).group(1)).split('|')[0].strip(),'articles':list(dict.fromkeys(v['url'].split('#')[0] for v in listed)), 'body':body})
    entries.sort(key=lambda p:(p['date'],p['url']),reverse=True)
    data={'articles':entries,'pages':pages,'legacy_body_hashes':source_hashes,'backup':str(backup)}
    destination.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'articles':len(entries),'pages':len(pages),'backup':str(backup),'categories':sorted({c['name'] for a in entries for c in a['categories']}),'tags':sorted({c['name'] for a in entries for c in a['tags']})},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
