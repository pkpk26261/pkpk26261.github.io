#!/usr/bin/env python3
"""Build a dependency-free GitHub Pages site from preserved article HTML."""
from pathlib import Path
from html import escape, unescape
from urllib.parse import unquote, quote
import json, re, math
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'content/site.json').read_text())
POSTS=DATA['articles']
WRITTEN=set()
E=lambda x:escape(str(x),quote=True)
ARROW='<span aria-hidden="true">↗</span>'
ICONS={
 'search':'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>',
 'menu':'<path d="M4 7h16M4 12h16M4 17h16"/>',
 'code':'<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>',
 'book':'<path d="M12 5v16M3 3h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5v16h-5a4 4 0 0 0-4 2 4 4 0 0 0-4-2H3Z"/>',
 'terminal':'<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m6 9 3 3-3 3m6 0h6"/>',
 'spark':'<path d="m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3Z"/>',
 'github':'<path d="M9 19c-4 1-4-2-6-2m12 5v-4a3.5 3.5 0 0 0-1-2.7c3.3-.4 6.8-1.6 6.8-7.3a5.7 5.7 0 0 0-1.5-4 5.3 5.3 0 0 0-.1-4S17.9-.4 15 1.5a13.5 13.5 0 0 0-7 0C5.1-.4 3.8 0 3.8 0a5.3 5.3 0 0 0-.1 4 5.7 5.7 0 0 0-1.5 4c0 5.7 3.5 6.9 6.8 7.3A3.5 3.5 0 0 0 8 18v4"/>'}
def icon(name): return f'<svg class="icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'

def group(a):
    names=' '.join(v['name'] for v in a['categories']+a['tags'])
    if any(x in names for x in ('YOLO','OpenAI','chatGPT')) or 'AI辨識' in a['title']: return 'ai'
    if 'Ubuntu' in names: return 'systems'
    if '勤益-上課筆記' in names or '程式語言學習' in names and 'Python' not in names: return 'notes'
    return 'python' if 'Python' in names else 'web'
LABELS={'ai':'AI 與電腦視覺','python':'Python 開發','systems':'系統與工具','notes':'課程筆記','web':'網站建置'}
SUMMARIES={
 'ChatGPT實務上所遇到的問題':'從實際使用經驗出發，整理 ChatGPT 的應用方式、常見問題與思考。',
 'QT5 Python的GUI介面 使用紀錄':'從環境安裝到介面設計，記錄使用 PyQt5 打造圖形化應用程式的過程。',
 'YOLOv5 的 使用紀錄':'整理模型訓練、資料準備與工業相機應用，留下物件偵測的實作紀錄。',
 'Python封裝exe應用程式':'透過 auto-py-to-exe，將 Python 程式打包為可執行的桌面應用程式。',
 'YOLOv7 的 初學使用紀錄':'記錄 YOLOv7 的環境設定、操作指令，以及初次實作時遇到的問題。',
 '下載相關圖片數據庫':'使用 Python 工具收集圖片資料，為後續的影像辨識實驗準備素材。',
 'Ubuntu 與 Windows 雙系統安裝引導':'從安裝準備到系統設定，整理 Ubuntu 與 Windows 雙系統的建置步驟。',
 '摸索如何自己建立Github的部落格':'從零開始搭建自己的學習部落格，記錄 GitHub 與 Hexo 的設定過程。',
 'OpenCV 程式指令整理':'整理影像處理實作中使用的 OpenCV 指令與操作範例。'}
SUMMARIES.update({
 'C++的勤益上課學習筆記':'整理 C++ 課堂中的資料型態、程式語法與練習範例，逐步建立程式設計基礎。',
 'C# 入門者快速教學文章（資料整理）':'從開發環境與專案建立開始，收集 C# 入門學習所需的教學與操作筆記。',
 'Ubuntu 執行 MATLAB 錯誤訊息 {Failed to load module "canberra-gtk-module"}':'記錄 MATLAB 在 Ubuntu 啟動時的模組錯誤，以及安裝相依套件的處理方式。',
 '勤益(二下)課程-上課筆記 專業軟體應用及實習':'整理 MATLAB 與專業軟體應用課程中的練習，保留每個操作步驟與學習重點。',
 '使用shutil、os模組協助複製、移動、刪除、新增目錄或檔案':'運用 shutil 與 os 模組處理檔案和資料夾，整理常用的自動化操作範例。',
 '使用相機攝影截圖取AI辨識樣本':'透過 OpenCV 擷取相機畫面，整理影像辨識樣本的收集與儲存流程。',
 'Tensorflow-gpu 安裝引導':'記錄在 Anaconda 環境中安裝 TensorFlow GPU 版本的指令與設定。',
 'Anaconda 安裝引導':'整理 Windows 與 Linux 上的 Anaconda 安裝步驟，準備 Python 開發環境。',
 'Anaconda 基本指令介紹':'從套件管理到虛擬環境，整理 conda 與 pip 的日常使用指令。',
 '勤益 Python 教育訓練研習':'收藏 Python 教育訓練的課程檔案、實作範例與研習中的學習紀錄。',
 '勤益(二上)課程-上課筆記 MATLAB 程式設計':'記錄 MATLAB 的基礎語法、矩陣運算與課堂練習，累積程式設計的理解。',
 '勤益(二上)課程-上課筆記 Arduino 程式設計':'從 Arduino 與 Tinkercad 的操作開始，留下電路設計和程式練習的過程。'
})
CARD_TITLES={
 'Ubuntu 執行 MATLAB 錯誤訊息 {Failed to load module "canberra-gtk-module"}':'Ubuntu 執行 MATLAB：啟動錯誤排除',
 '使用shutil、os模組協助複製、移動、刪除、新增目錄或檔案':'Python 檔案管理：shutil 與 os 模組',
 '勤益(二下)課程-上課筆記 專業軟體應用及實習':'專業軟體應用及實習：課堂筆記',
 '勤益(二上)課程-上課筆記 MATLAB 程式設計':'MATLAB 程式設計：課堂筆記',
 '勤益(二上)課程-上課筆記 Arduino 程式設計':'Arduino 程式設計：課堂筆記'
}

for a in POSTS:
    a['group']=group(a); a['label']=LABELS[a['group']]
    a['topics']=[a['group']]
    if any(v['name']=='Python' for v in a['tags']) and 'python' not in a['topics']: a['topics'].append('python')
    a['summary']=a.get('summary') or SUMMARIES.get(a['title'],re.sub(r'⬇.*?⬇|文章開始|\s+',' ',a['text'])[:95].strip()+'…')
    a['minutes']=max(1,math.ceil(len(a['text'])/550))

BASE='https://pkpk26261.github.io'
AUTHOR={'@type':'Person','name':'永成 Yong Cheng','url':'https://github.com/pkpk26261'}
def jsonld(data):
    return '<script type="application/ld+json">'+json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')+'</script>'
def structured(url,section,title,desc,article):
    if url=='/':
        return jsonld({'@context':'https://schema.org','@type':'WebSite','name':'永成的學習手札','alternateName':'Yong Cheng Notebook','url':BASE+'/','inLanguage':'zh-Hant','author':AUTHOR,'potentialAction':{'@type':'SearchAction','target':{'@type':'EntryPoint','urlTemplate':BASE+'/articles/?q={search_term_string}'},'query-input':'required name=search_term_string'}})
    if section=='post' and article:
        canonical=BASE+article['url']
        post={'@context':'https://schema.org','@type':'BlogPosting','headline':title,'description':desc,'datePublished':article['date'],'dateModified':article['updated'] or article['date'],'url':canonical,'mainEntityOfPage':canonical,'inLanguage':'zh-Hant','author':AUTHOR,'publisher':AUTHOR,'image':BASE+'/assets/illustrations/usagi-study-v2.png','articleSection':article['label'],'keywords':[v['name'] for v in article['tags']]}
        crumbs={'@context':'https://schema.org','@type':'BreadcrumbList','itemListElement':[{'@type':'ListItem','position':1,'name':'首頁','item':BASE+'/'},{'@type':'ListItem','position':2,'name':'文章','item':BASE+'/articles/'},{'@type':'ListItem','position':3,'name':article['label'],'item':BASE+'/articles/?topic='+article['group']},{'@type':'ListItem','position':4,'name':title,'item':canonical}]}
        return jsonld([post,crumbs])
    return ''

def shell(title, body, url='/', section='home', description=None, article=None):
    desc=description or '永成的個人網站與學習紀錄。分享 Python、AI 影像辨識、系統工具與程式設計的探索。'
    nav=[('home','/','首頁'),('articles','/articles/','文章'),('categories','/categories/','主題分類'),('pen','/pen/','筆札'),('about','/about/','關於我')]
    navhtml=''.join(f'<a href="{u}"'+(' aria-current="page"' if s==section else '')+f'>{n}</a>' for s,u,n in nav)
    artmeta=(f'<meta property="article:published_time" content="{article["date"]}"><meta property="article:modified_time" content="{article["updated"] or article["date"]}"><link rel="preconnect" href="https://i.imgur.com" crossorigin><link rel="dns-prefetch" href="https://i.imgur.com">' if section=='post' and article else '')
    return f'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#f8f7f3"><meta name="description" content="{E(desc)}"><meta name="google-site-verification" content="3Gg_DXWjGio6FEIB-ARrwaMbjfLismSD4kKjoMkCVV8">
<title>{E(title)}{' | 永成 · Yong Cheng' if url!='/' else ''}</title><link rel="canonical" href="https://pkpk26261.github.io{E(url)}">
<meta property="og:type" content="{'article' if section=='post' else 'website'}"><meta property="og:title" content="{E(title)}"><meta property="og:description" content="{E(desc)}"><meta property="og:url" content="https://pkpk26261.github.io{E(url)}"><meta property="og:image" content="https://pkpk26261.github.io/assets/illustrations/usagi-study-v2.png"><meta property="og:locale" content="zh_TW"><meta name="twitter:card" content="summary">
<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg"><link rel="stylesheet" href="/assets/site.css"><script defer src="/assets/site.js"></script>{artmeta}{structured(url,section,title,desc,article)}</head>
<body data-article-count="{len(POSTS)}"><a class="skip-link" href="#main">跳至主要內容</a>
<header class="site-header"><div class="header-inner wrap"><a class="identity" href="/" aria-label="永成 Yong Cheng 首頁"><span class="monogram">yc<span>.</span></span><span class="identity-name">永成 <span>/ Yong Cheng</span></span></a>
<nav class="desktop-nav" aria-label="主要導覽">{navhtml}</nav><div class="header-actions"><button class="icon-button" type="button" data-search aria-label="搜尋文章">{icon('search')}</button><a class="github-link" href="https://github.com/pkpk26261" target="_blank" rel="noopener noreferrer">GitHub {ARROW}</a><button class="icon-button menu-toggle" aria-label="開啟導覽選單" aria-expanded="false" aria-controls="mobile-nav" type="button">{icon('menu')}</button></div></div><nav id="mobile-nav" class="mobile-nav" aria-label="行動版導覽" hidden>{navhtml}<a href="/archives/">文章歸檔</a><a href="/tags/">所有標籤</a></nav></header>
<main id="main" tabindex="-1">{body}</main>
<footer class="site-footer"><div class="wrap footer-top"><a class="identity" href="/"><span class="monogram">yc<span>.</span></span><span class="identity-name">永成 <span>/ Yong Cheng</span></span></a><p>保持好奇，持續學習。<br><span>Usagi 主題 · 永成的個人網站</span></p><nav aria-label="頁尾導覽"><a href="/archives/">文章歸檔</a><a href="/tags/">所有標籤</a><a href="/about/#聯絡我">聯絡我 {ARROW}</a><a href="/editor/">文章工作室</a></nav></div><div class="wrap footer-bottom"><span>© 2019–2026 Yong Cheng</span><span>以文字記錄探索，讓知識持續累積。</span><a href="https://github.com/pkpk26261" target="_blank" rel="noopener noreferrer">Built on GitHub Pages {ARROW}</a></div></footer>
<dialog class="search-dialog" aria-labelledby="search-title"><div class="search-dialog-head"><div><span class="eyebrow">SEARCH THE JOURNAL</span><h2 id="search-title">尋找一筆學習紀錄</h2></div><button type="button" class="icon-button close-search" aria-label="關閉搜尋">×</button></div><form method="get" action="/articles/" class="search-form"><label class="search-field">{icon('search')}<input type="search" name="q" placeholder="搜尋文章、關鍵字或主題…" aria-label="搜尋關鍵字" autocomplete="off"></label></form><p id="search-status" class="search-status" role="status">輸入關鍵字，搜尋所有 {len(POSTS)} 篇文章。</p><div class="search-results"></div><div class="search-help"><span>標題、分類與全文搜尋</span><span><kbd>Esc</kbd> 關閉</span></div></dialog>
<dialog class="image-dialog" aria-label="文章圖片預覽"><button class="icon-button close-image" type="button" aria-label="關閉圖片預覽">×</button><img alt=""><p></p></dialog><button class="back-top icon-button" type="button" aria-label="回到頁面頂端" hidden>↑</button><div class="sr-only" id="site-status" role="status"></div></body></html>'''

def write(url, html):
    WRITTEN.add(url)
    p=ROOT/unquote(url).lstrip('/')/'index.html'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(html)

def card(a, index=0, hidden=False):
    return f'''<article class="article-card" data-group="{a['group']}" data-topics="{' '.join(a['topics'])}" data-search-text="{E(a['title']+' '+a['summary']+' '+a['label']+' '+' '.join(v['name'] for v in a['tags']))}"{' hidden' if hidden else ''}>
<a class="card-art art-{a['group']}" href="{E(a['url'])}" tabindex="-1" aria-hidden="true"><img src="/assets/illustrations/{a['group']}.svg" alt="" loading="lazy" width="600" height="340"><span class="art-label">{E(a['label'])}</span><span class="art-number">{index+1:02d}</span></a><div class="card-meta"><span>{E(a['label'])}</span><time datetime="{a['date']}">{a['date'].replace('-','.')} </time></div><h3><a href="{E(a['url'])}">{E(CARD_TITLES.get(a['title'],a['title']))}</a></h3><p>{E(a['summary'])}</p><a class="card-read" href="{E(a['url'])}" aria-label="閱讀：{E(a['title'])}">閱讀文章 <span aria-hidden="true">↗</span></a></article>'''

def filters(): return '<div class="filter-bar" role="group" aria-label="篩選文章主題"><button type="button" class="active" data-filter="all" aria-pressed="true">全部文章</button>'+''.join(f'<button type="button" data-filter="{k}" aria-pressed="false">{v}</button>' for k,v in LABELS.items())+'</div>'
def pagehead(eyebrow,title,desc): return f'<header class="page-heading wrap"><span class="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{desc}</p></header>'

def topics():
    items=[('python','code','Python 開發','環境設定、GUI 介面與應用程式實作','/tags/Python/',sum(any(v['name']=='Python' for v in a['tags']) for a in POSTS)),('ai','spark','AI 與電腦視覺','YOLO 物件偵測、影像處理與 ChatGPT','/articles/?topic=ai',sum(a['group']=='ai' for a in POSTS)),('systems','terminal','系統與工具','Ubuntu、Linux 與開發環境的整理','/categories/Ubuntu/',sum(a['group']=='systems' for a in POSTS)),('notes','book','課程筆記','程式設計與專業課程的學習紀錄','/articles/?topic=notes',sum(a['group']=='notes' for a in POSTS))]
    return '<div class="topic-grid">'+''.join(f'<a class="topic-card" href="{u}"><span class="topic-icon">{icon(i)}</span><span class="topic-count">{n:02d} 篇文章</span><h3>{t}</h3><p>{d}</p><span class="topic-arrow" aria-hidden="true">↗</span></a>' for g,i,t,d,u,n in items)+'</div>'

def homepage():
    body=f'''<section class="hero wrap"><div class="hero-copy"><div class="eyebrow"><span class="tiny-dot"></span> YONG CHENG’S NOTEBOOK · SINCE 2019</div><h1>保持好奇，<br>把學習變成<span>小小冒險。</span></h1><p class="hero-description">你好，我是永成。<br>在程式、科技與生活之間探索，把每一次實作與發現，<br class="desktop-break">整理成值得留下的學習紀錄。</p><div class="hero-actions"><a class="button button-primary" href="#journal">探索我的文章 <span aria-hidden="true">↗</span></a><a class="text-link" href="/about/">認識我 <span aria-hidden="true">→</span></a></div><div class="hero-footnote"><span>PYTHON</span><span>COMPUTER VISION</span><span>LEARNING BY DOING</span></div></div><div class="hero-art"><img src="/assets/illustrations/usagi-study-v2.png" width="550" height="480" alt="Usagi 兔兔與筆記本、電腦一起學習的插畫"><div class="hero-art-caption"><span>USAGI & MY LITTLE NOTEBOOK</span><span>保持好奇，持續前進 ↗</span></div></div></section>
<div class="intro-strip wrap"><p><span class="tiny-dot"></span> 一個持續累積的知識角落</p><div><span><strong>{len(POSTS)}</strong> 篇學習紀錄</span><span><strong>2019</strong> 開始記錄</span><a href="/archives/">走過的學習路徑 <span aria-hidden="true">→</span></a></div></div>
<section class="journal-section wrap" id="journal" data-list data-limit="6"><div class="section-heading"><div><span class="eyebrow">THE JOURNAL</span><h2>最近的學習手札<span class="heading-dot">.</span></h2></div><a class="text-link" href="/articles/">查看全部文章 <span class="count-pill">{len(POSTS)}</span> <span aria-hidden="true">↗</span></a></div>{filters()}<p class="filter-status sr-only" role="status"></p><div class="article-grid">{''.join(card(a,i,i>=6) for i,a in enumerate(POSTS))}</div><p class="empty-state" hidden>目前沒有這個主題的文章。</p></section>
<section class="topics-section"><div class="wrap"><div class="section-heading"><div><span class="eyebrow">EXPLORE BY TOPIC</span><h2>從感興趣的主題開始<span class="heading-dot">.</span></h2></div><a class="text-link" href="/categories/">所有分類 <span aria-hidden="true">↗</span></a></div>{topics()}</div></section>
<section class="about-strip wrap"><div class="about-index"><span class="eyebrow">A LITTLE ABOUT ME</span><img class="usagi-sticker" src="/assets/illustrations/usagi-sticker.png" width="160" height="160" loading="lazy" alt="Usagi 兔兔拿著筆和筆記本的原創貼圖"></div><div><h2>我一旦決定，<br>就為了夢前進。</h2><p>學習不只是找到答案，也是把問題想得更清楚。<br>這裡收藏我的實作過程、課堂筆記，以及沿途的發現。</p><a class="text-link" href="/about/">更多關於永成 <span aria-hidden="true">↗</span></a></div><a class="about-github" href="https://github.com/pkpk26261" target="_blank" rel="noopener noreferrer">{icon('github')}<span>也可以在 GitHub<br>找到我的探索紀錄</span><span aria-hidden="true">↗</span></a></section>'''
    write('/',shell('永成的學習手札 · Usagi 主題個人網站',body))

def hub():
    body=pagehead('ALL WRITING','文章與學習紀錄<span class="heading-dot">.</span>','把實作中的問題與發現，整理成可以再次翻閱的知識。')
    body+=f'<section class="wrap listing-section" data-list><div class="listing-toolbar"><p>共 <strong>{len(POSTS)}</strong> 篇文章</p><label class="inline-search">{icon("search")}<input type="search" placeholder="搜尋文章…" aria-label="篩選文章關鍵字"></label><a class="text-link" href="/archives/">依年份瀏覽 →</a></div>{filters()}<p class="filter-status" role="status">顯示全部 {len(POSTS)} 篇文章</p><div class="article-grid">'+''.join(card(a,i) for i,a in enumerate(POSTS))+'</div><p class="empty-state" hidden>沒有符合的文章，試試其他關鍵字或主題。</p></section>'
    write('/articles/',shell('文章與學習紀錄',body,'/articles/','articles'))

def categoryindex(kind):
    by_url={}
    for a in POSTS:
        for v in a[kind]:
            item=by_url.setdefault(v['url'],{'names':[],'posts':[]})
            if v['name'] not in item['names']: item['names'].append(v['name'])
            if a not in item['posts']: item['posts'].append(a)
    prefix='/categories/' if kind=='categories' else '/tags/'
    title='主題分類' if kind=='categories' else '所有標籤'
    body=pagehead('TOPICS & COLLECTIONS' if kind=='categories' else 'THE INDEX',title+'<span class="heading-dot">.</span>','依照主題探索，找到下一篇想讀的文章。' if kind=='categories' else '用關鍵字串起相關的實作、工具與課程筆記。')
    body+='<section class="wrap collection-section"><div class="collection-grid">'+''.join(f'<a class="collection-card" href="{E(u)}"><span class="eyebrow">COLLECTION {i+1:02d}</span><h2>{E(" / ".join(x["names"]))}</h2><p>{len(x["posts"])} 篇學習紀錄</p><span aria-hidden="true">↗</span></a>' for i,(u,x) in enumerate(sorted(by_url.items(),key=lambda v:-len(v[1]['posts']))))+'</div></section>'
    write(prefix,shell(title,body,prefix,kind))
    return by_url

def row(a): return f'<a class="archive-row" href="{E(a["url"])}"><time datetime="{a["date"]}">{a["date"][5:].replace("-",".")}</time><span>{E(a["title"])}</span><span class="row-label">{E(a["label"])}</span><span aria-hidden="true">↗</span></a>'
def archivepage(url, posts, heading='文章歸檔'):
    body=pagehead('THE LEARNING TIMELINE',E(heading)+'<span class="heading-dot">.</span>',f'沿著時間，重新翻閱 {len(posts)} 篇學習紀錄。')
    body+='<section class="wrap archive-section"><nav class="year-links" aria-label="依年份瀏覽"><a href="/archives/"'+(' aria-current="page"' if url=='/archives/' else '')+'>所有年份</a>'+''.join(f'<a href="/archives/{y}/"'+(' aria-current="page"' if url==f'/archives/{y}/' else '')+f'>{y}</a>' for y in sorted({a['date'][:4] for a in POSTS},reverse=True))+'</nav>'
    for year in sorted({a['date'][:4] for a in posts},reverse=True):
        selected=[a for a in posts if a['date'].startswith(year)]
        body+=f'<div class="archive-year"><div><h2>{year}</h2><span>{len(selected)} 篇紀錄</span></div><div>{"".join(row(a) for a in selected)}</div></div>'
    body+='</section>'
    write(url,shell(heading,body,url,'articles'))

def legacylist(page, cats, tags):
    url=page['url']
    if url in ('/','/categories/','/tags/','/about/','/pen/'): return
    if url.startswith('/archives/'):
        tail=url[len('/archives/'):].strip('/'); match=re.match(r'^(\d{4})(?:/(\d{2}))?',tail)
        if match:
            prefix=match[1]+('-'+match[2] if match[2] else '')
            posts=[a for a in POSTS if a['date'].startswith(prefix)]
            archivepage(url,posts,prefix.replace('-',' 年 ')+' 的文章'); return
        if url=='/archives/': archivepage(url,POSTS); return
        posts=[a for a in POSTS if a['url'] in page['articles']]; archivepage(url,posts); return
    kind='categories' if url.startswith('/categories/') else 'tags' if url.startswith('/tags/') else None
    lookup=cats if kind=='categories' else tags
    if kind:
        base=re.sub(r'page/\d+/$','',url)
        item=lookup.get(base)
        posts=item['posts'] if item else [a for a in POSTS if a['url'] in page['articles']]
        title=' / '.join(item['names']) if item else page['title']
        back=f'<a class="text-link back-link" href="/{kind}/">← 返回{"主題分類" if kind=="categories" else "所有標籤"}</a>'
    else:
        posts=[a for a in POSTS if a['url'] in page['articles']]; title='文章列表'; back='<a class="text-link back-link" href="/articles/">← 查看全部文章</a>'
    body=pagehead('COLLECTED NOTES',E(title)+'<span class="heading-dot">.</span>',f'關於這個主題的 {len(posts)} 篇學習紀錄。')
    body+=f'<section class="wrap listing-section">{back}<div class="article-grid">'+''.join(card(a,i) for i,a in enumerate(posts))+'</div></section>'
    write(url,shell(title,body,url,kind or 'articles'))

def prose(source):
    # Preserve stored sources byte for byte; normalize heading hierarchy only in rendered output.
    html=(ROOT/source).read_text()
    html=re.sub(r'<(/?)h([1-5])\b',lambda m:'<'+m[1]+'h'+str(int(m[2])+1),html)
    html=re.sub(r'<img\b', '<img loading="lazy"',html)
    html=re.sub(r'<iframe\b', '<iframe loading="lazy" title="原文嵌入內容"',html)
    html=re.sub(r'class="headerlink" title="([^"]*)">', r'class="headerlink" title="\1" aria-label="章節連結：\1">', html)
    return html

def toc(source):
    body=(ROOT/source).read_text(); items=[]
    for m in re.finditer(r'<h([1-3])\b[^>]*id="([^"]+)"[^>]*>(.*?)</h\1>',body,re.S):
        t=re.sub(r'<[^>]+>','',m[3]); items.append(f'<li class="toc-level-{m[1]}"><a href="#{E(m[2])}">{E(unescape(t))}</a></li>')
    return '<details class="article-toc" open><summary>本篇目錄 <span aria-hidden="true">⌄</span></summary><nav aria-label="文章目錄"><ol>'+''.join(items)+'</ol></nav></details>' if items else ''

def article(a,i):
    crumbs=f'<nav class="breadcrumbs" aria-label="麵包屑"><a href="/">首頁</a><span>/</span><a href="/articles/">文章</a><span>/</span><a href="/articles/?topic={a["group"]}">{E(a["label"])}</a></nav>'
    meta=f'<time datetime="{a["date"]}">{a["date"].replace("-",".")}</time><span>永成 / Yong Cheng</span><span>約 {a["minutes"]} 分鐘閱讀</span>'
    if a['updated']: meta+=f'<span>更新於 {a["updated"].replace("-",".")}</span>'
    body=f'<div class="wrap article-wrap">{crumbs}<header class="article-heading"><span class="eyebrow">{E(a["label"])}</span><h1>{E(a["title"])}</h1><div class="article-meta">{meta}</div><p>{E(a["summary"])}</p></header><div class="reading-layout"><article class="prose">{prose(a["body"])}<footer class="article-end"><p>這是永成的學習紀錄，內容保留發表當時的實作經驗。</p><div class="tag-links">'+''.join(f'<a href="{E(v["url"])}"># {E(v["name"])}</a>' for v in a['tags'])+'</div></footer></article><aside class="reading-aside">'+toc(a['body'])+'<div class="reader-card"><img class="reader-sticker" src="/assets/illustrations/usagi-sticker.png" width="95" height="95" loading="lazy" alt=""><span class="eyebrow">KEEP EXPLORING</span><p>每一筆紀錄，<br>都是下一步的起點。</p><a href="/articles/">探索更多文章 →</a></div></aside></div><nav class="post-pagination" aria-label="上一篇與下一篇">'
    for label,p in [('較新的文章',POSTS[i-1] if i>0 else None),('較早的文章',POSTS[i+1] if i+1<len(POSTS) else None)]:
        if p: body+=f'<a href="{E(p["url"])}"><span>{label}</span><strong>{E(p["title"])}</strong><span aria-hidden="true">↗</span></a>'
    body+='</nav></div>'
    write(a['url'],shell(a['title'],body,a['url'],'post',a['summary'],a))

def about():
    body=pagehead('THE PERSON BEHIND THE NOTES','你好，我是永成<span class="heading-dot">.</span>','把好奇心放進實作，把學習過程留在文字裡。')
    body+='''<section class="wrap about-intro"><div class="about-visual"><span class="eyebrow">YONG CHENG</span><img class="usagi-about-image" src="/assets/illustrations/usagi-sticker.png" alt="Usagi 兔兔的學習日常"><div>EXPLORING & LEARNING<br>SINCE 2019</div></div><div class="about-intro-copy"><h2>我一旦決定，<br>就為了夢前進。</h2><p>我喜歡把學到的知識付諸實作，也喜歡將過程整理成紀錄。從 Python、影像辨識到課程筆記，這個網站是我收藏問題、嘗試與發現的地方。</p><p>2019 年 10 月 17 日，我開始建立這個部落格。希望在未來的路上，仍能翻閱這些紀錄，看見自己一步步走來的軌跡。</p><div class="hero-actions"><a class="button button-primary" href="https://github.com/pkpk26261" target="_blank" rel="noopener noreferrer">前往我的 GitHub ↗</a><a class="text-link" href="#聯絡我">與我聯絡 →</a></div></div></section>'''
    body+='<section class="wrap about-details"><div class="section-heading"><div><span class="eyebrow">MY STORY & WORK</span><h2>關於這個網站與我的作品<span class="heading-dot">.</span></h2></div></div><div class="reading-layout"><div class="prose">'+prose('content/about.html')+'</div><aside class="reading-aside">'+toc('content/about.html')+'</aside></div></section>'
    body+='''<section class="wrap contact-section" id="聯絡我"><div><span class="eyebrow">LET’S CONNECT</span><h2>有想法，歡迎聊聊。</h2><p>如果文章幫上了忙，或有問題想交流，歡迎與我聯繫。</p></div><div><a href="mailto:a0979488285@gmail.com">E-Mail <span>a0979488285@gmail.com ↗</span></a><a href="https://line.me/ti/p/XGjZN3WZhs" target="_blank" rel="noopener noreferrer">LINE <span>pkpk26261 ↗</span></a><a href="https://github.com/pkpk26261" target="_blank" rel="noopener noreferrer">GitHub <span>@pkpk26261 ↗</span></a></div></section>'''
    write('/about/',shell('關於永成',body,'/about/','about'))

def pen():
    body=pagehead('HANDWRITTEN NOTES','筆札，留住思考的痕跡<span class="heading-dot">.</span>','收藏課堂上的手寫筆記，也留下學習過程中的片刻。')
    body+='<section class="wrap pen-section"><div class="reading-layout"><div class="prose note-prose">'+prose('content/pen.html')+'</div><aside class="reading-aside"><div class="reader-card"><img class="reader-sticker" src="/assets/illustrations/usagi-sticker.png" width="95" height="95" loading="lazy" alt=""><span class="eyebrow">CLASSROOM NOTES</span><p>從紙上的推演，<br>到腦中的理解。</p><span class="muted">筆記保留原有雲端連結；存取權限依原平台設定。</span></div></aside></div></section>'
    write('/pen/',shell('手寫筆札',body,'/pen/','pen'))

def main():
    homepage(); hub(); cats=categoryindex('categories'); tags=categoryindex('tags')
    for p in DATA['pages']: legacylist(p,cats,tags)
    for kind,lookup in [('categories',cats),('tags',tags)]:
        for url in lookup: legacylist({'url':url,'title':'','articles':[]},cats,tags)
    for year in sorted({a['date'][:4] for a in POSTS}): archivepage(f'/archives/{year}/',[a for a in POSTS if a['date'].startswith(year)],year+' 的文章')
    for month in sorted({a['date'][:7] for a in POSTS}): archivepage('/archives/'+month.replace('-','/')+'/',[a for a in POSTS if a['date'].startswith(month)],month+' 的文章')
    for i,a in enumerate(POSTS): article(a,i)
    about(); pen()
    search=[{k:a[k] for k in ('title','url','date','summary','text','label')} for a in POSTS]
    (ROOT/'assets/search-index.json').write_text(json.dumps(search,ensure_ascii=False,separators=(',',':'))+'\n')
    write('/404/',shell('找不到這個頁面',pagehead('404 · PAGE NOT FOUND','這一頁，還沒有筆記。','你可以回到首頁，或搜尋想找的學習紀錄。')+'<div class="wrap error-actions"><a class="button button-primary" href="/">回到首頁 ↗</a><button class="button button-outline" type="button" data-search>搜尋文章</button></div>','/404/'))
    (ROOT/'404.html').write_text((ROOT/'404/index.html').read_text())
    (ROOT/'.nojekyll').touch()
    urls=sorted(WRITTEN - {'/404/'})
    postmod={a['url']:(a['updated'] or a['date']) for a in POSTS}
    latest=max(postmod.values())
    def loc(u):
        lm=postmod.get(u,latest)
        freq='weekly' if u in ('/','/articles/') else ('monthly' if u in postmod else 'yearly')
        prio='1.0' if u=='/' else ('0.8' if u in postmod else '0.5')
        return f'<url><loc>{BASE}{E(u)}</loc><lastmod>{lm}</lastmod><changefreq>{freq}</changefreq><priority>{prio}</priority></url>'
    (ROOT/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join(loc(u) for u in urls)+'</urlset>\n')
    print(f'Built {len(urls)} pages + 404; rendered {len(POSTS)} articles.')
if __name__=='__main__': main()
