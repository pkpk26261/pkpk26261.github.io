#!/usr/bin/env python3
"""Loopback-only article editor and static preview server; stdlib only."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, parse_qs, quote, unquote
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, json, secrets, hashlib, re, subprocess, sys, threading, shutil, base64, webbrowser, urllib.request
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'content/site.json'; DRAFTS=ROOT/'content/drafts'; REVISIONS=ROOT/'content/revisions'
TOKEN=secrets.token_urlsafe(32); LOCK=threading.Lock()
def now():return datetime.now(ZoneInfo('Asia/Taipei'))
def load():return json.loads(SITE.read_text())
def write_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
class Plain(HTMLParser):
 def __init__(self):super().__init__();self.parts=[]
 def handle_data(self,d):self.parts.append(d)
class SafeHTML(HTMLParser):
 def handle_starttag(self,tag,attrs):
  if tag in {'script','object','embed','base','meta','link','style'}:raise ValueError('文章包含不支援的可執行 HTML，請移除腳本或頁面設定。')
  for name,value in attrs:
   if name.startswith('on'):raise ValueError('文章不支援事件屬性，請移除該屬性。')
   if name in ('href','src') and re.match(r'\s*(?:javascript|vbscript|data:text/html):',value or '',re.I):raise ValueError('請使用有效的圖片或連結網址。')
def plain(body):
 p=Plain();p.feed(body);return re.sub(r'\s+',' ',' '.join(p.parts)).strip()
def article_fields(a):
 return {k:a.get(k) for k in ('id','title','url','date','updated','summary','categories','tags')}
def get_article(key):
 draft=DRAFTS/(key+'.json')
 if draft.exists():return json.loads(draft.read_text())
 a=next((a for a in load()['articles'] if a['id']==key),None)
 if not a:return None
 return {**article_fields(a),'body':(ROOT/a['body']).read_text(),'status':'published'}
def term_links(raw,kind,data):
 names=list(dict.fromkeys(v.strip() for v in raw.split(',') if v.strip()))
 if len(names)>20:raise ValueError('分類或標籤最多 20 個。')
 existing={v['name']:v['url'] for a in data['articles'] for v in a[kind]}
 return [{'name':n,'url':existing.get(n,'/'+kind+'/'+quote(n,safe='')+'/')} for n in names]
def save_article(payload,publish=False):
 data=load();key=payload.get('id') or secrets.token_hex(6)
 if not re.fullmatch(r'[a-f0-9]{12}',key):raise ValueError('文章識別碼無效。')
 original=next((a for a in data['articles'] if a['id']==key),None)
 title=str(payload.get('title','')).strip();body=str(payload.get('body',''))
 if not title or len(title)>180:raise ValueError('請輸入 1–180 個字的文章標題。')
 if not plain(body).strip():raise ValueError('請輸入文章內容。')
 if len(body)>2_000_000:raise ValueError('文章內容過大；圖片請以檔案路徑插入。')
 SafeHTML().feed(body)
 date=str(payload.get('date',''));datetime.strptime(date,'%Y-%m-%d')
 if original:date=original['date'];url=original['url']
 else:
  slug=str(payload.get('slug','')).strip().lower()
  if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,99}',slug):raise ValueError('網址代稱請使用英文字母、數字與連字號。')
  url='/'+date.replace('-','/')+'/'+slug+'/'
  if any(unquote(a['url'])==unquote(url) for a in data['articles']):raise ValueError('此文章網址已經存在，請換一個網址代稱。')
  route=ROOT/unquote(url).lstrip('/')/'index.html'
  if route.exists():raise ValueError('此網址已有頁面，請換一個網址代稱。')
 categories=term_links(str(payload.get('categories','')),'categories',data);tags=term_links(str(payload.get('tags','')),'tags',data)
 record={'id':key,'title':title,'url':url,'date':date,'updated':now().strftime('%Y-%m-%d'),'summary':str(payload.get('summary','')).strip()[:300],'categories':categories,'tags':tags,'body':body,'status':'draft','savedAt':now().isoformat()}
 draftfile=DRAFTS/(key+'.json')
 if not publish:
  write_json(draftfile,record);return {'id':key,'status':'draft','url':url,'message':'草稿已儲存，網站公開內容尚未變更。'}
 source=ROOT/f'content/articles/{key}.html'
 snapshot=REVISIONS/(now().strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3))
 snapshot.mkdir(parents=True)
 shutil.copy2(SITE,snapshot/'site.json')
 if source.exists():shutil.copy2(source,snapshot/(key+'.html'))
 saved_site=SITE.read_bytes();saved_body=source.read_bytes() if source.exists() else None
 saved_draft=draftfile.read_bytes() if draftfile.exists() else None
 current={**record,'path':unquote(url).lstrip('/')+'index.html','body':str(source.relative_to(ROOT)),'text':plain(body)}
 current.pop('status');current.pop('savedAt')
 if original:data['articles'][data['articles'].index(original)]=current
 else:data['articles'].append(current)
 data['articles'].sort(key=lambda a:(a['date'],a['url']),reverse=True)
 data.setdefault('current_body_hashes',dict(data['legacy_body_hashes']))[key]=hashlib.sha256(body.encode()).hexdigest()
 source.write_text(body);write_json(SITE,data)
 result=subprocess.run([sys.executable,str(ROOT/'scripts/build_site.py')],cwd=ROOT,capture_output=True,text=True)
 if result.returncode:
  SITE.write_bytes(saved_site)
  if saved_body is not None:source.write_bytes(saved_body)
  else:source.unlink(missing_ok=True)
  subprocess.run([sys.executable,str(ROOT/'scripts/build_site.py')],cwd=ROOT,capture_output=True)
  if not original:shutil.rmtree((ROOT/current['path']).parent,ignore_errors=True)
  raise RuntimeError('網站產生失敗，已還原原內容。')
 draftfile.unlink(missing_ok=True)
 return {'id':key,'status':'published','url':url,'message':'已更新本機網站。上傳 GitHub 後才會更新線上網站。'}
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def send_json(self,data,status=200):
  raw=json.dumps(data,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def valid_host(self):
  return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
 def do_GET(self):
  url=urlsplit(self.path)
  if not self.valid_host():self.send_error(403);return
  if url.path.startswith('/api/'):
   if self.headers.get('Sec-Fetch-Site')=='cross-site':self.send_json({'error':'只允許本機同來源的編輯器。'},403);return
   if url.path=='/api/state':
    data=load();drafts=[json.loads(p.read_text()) for p in DRAFTS.glob('*.json')]
    articles=[{**article_fields(a),'status':'published','hasDraft':any(d['id']==a['id'] for d in drafts)} for a in data['articles']]
    existing={a['id'] for a in articles}
    articles += [{**article_fields(d),'status':'draft','hasDraft':True} for d in drafts if d['id'] not in existing]
    self.send_json({'articles':articles,'token':TOKEN,'date':now().strftime('%Y-%m-%d')});return
   if url.path=='/api/article':
    key=parse_qs(url.query).get('id',[''])[0]
    if not re.fullmatch(r'[a-f0-9]{12}',key):self.send_json({'error':'文章識別碼無效。'},400);return
    article=get_article(key);self.send_json(article or {'error':'找不到文章。'},200 if article else 404);return
   self.send_json({'error':'找不到 API。'},404);return
  # Private backups and drafts are not served as website content.
  if url.path.startswith(('/content/drafts/','/content/revisions/','/design/')):self.send_error(404);return
  super().do_GET()
 def do_POST(self):
  origin=self.headers.get('Origin');allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
  if not self.valid_host() or origin not in allowed or self.headers.get('X-Editor-Token')!=TOKEN:self.send_json({'error':'儲存請求未通過本機來源檢查。'},403);return
  if self.path not in ('/api/save-draft','/api/publish','/api/upload-image'):self.send_json({'error':'找不到 API。'},404);return
  if not self.headers.get('Content-Type','').startswith('application/json'):self.send_json({'error':'請使用 JSON 格式。'},400);return
  try:
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<(7_100_000 if self.path=='/api/upload-image' else 3_000_000):raise ValueError('請求內容過大或為空。')
   payload=json.loads(self.rfile.read(size))
   if self.path=='/api/upload-image':
    blob=base64.b64decode(payload.get('data',''),validate=True)
    if not blob or len(blob)>5_000_000:raise ValueError('圖片大小需小於 5 MB。')
    extension='png' if blob.startswith(b'\x89PNG\r\n\x1a\n') else 'jpg' if blob.startswith(b'\xff\xd8\xff') else 'gif' if blob.startswith((b'GIF87a',b'GIF89a')) else 'webp' if blob.startswith(b'RIFF') and blob[8:12]==b'WEBP' else None
    if not extension:raise ValueError('支援 PNG、JPEG、GIF 和 WebP 圖片。')
    name=secrets.token_hex(10)+'.'+extension;folder=ROOT/'images/uploads';folder.mkdir(parents=True,exist_ok=True);(folder/name).write_bytes(blob)
    result={'url':'/images/uploads/'+name,'message':'圖片已儲存到專案。'}
   else:
    with LOCK:result=save_article(payload,self.path=='/api/publish')
   self.send_json(result)
  except (ValueError,KeyError,TypeError) as e:self.send_json({'error':str(e)},400)
  except Exception:self.send_json({'error':'儲存未完成，請保留編輯中的內容後重試。'},500)
 def log_message(self,format,*args):
  if len(args)>1 and str(args[1]) not in ('200','304'):super().log_message(format,*args)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=4174);parser.add_argument('--open',action='store_true');args=parser.parse_args()
 try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
 except OSError:
  if args.open:
   try:
    state=json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/state',timeout=2))
    if 'articles' in state and 'token' in state:
     webbrowser.open(f'http://127.0.0.1:{args.port}/editor/');return
   except Exception:pass
  raise SystemExit(f'連接埠 {args.port} 已被占用。若文章工作室已啟動，請開啟 http://127.0.0.1:{args.port}/editor/；或使用 --port 4175。')
 print(f'網站：http://127.0.0.1:{args.port}/\n文章編輯器：http://127.0.0.1:{args.port}/editor/',flush=True)
 if args.open:threading.Timer(0.5,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}/editor/')).start()
 try:server.serve_forever()
 except KeyboardInterrupt:server.server_close()
if __name__=='__main__':main()
