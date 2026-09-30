# 永成的學習手札 · Usagi 主題個人網站

奶油黃、米白與鼠尾草綠的個人學習網站，包含 Usagi 學習插畫與新製作的透明貼圖。原有 21 篇文章、圖片、程式碼、原文連結與 71 個頁面網址均已保留，統一首頁、文章、分類、標籤、歸檔、筆札和關於頁的設計。

網站仍是適合 GitHub Pages 的靜態檔案。日後寫作使用附帶的繁體中文「文章工作室」，在自己的電腦儲存草稿與產生網站；不需要 npm、雲端帳號或付費服務。

## 開始編輯文章

在 Mac 上雙擊 `啟動文章工作室.command`。它會開啟終端機、啟動本機儲存服務，並開啟文章工作室。若本機服務已啟動，直接開啟 http://127.0.0.1:4174/editor/ 即可。

也可以在此資料夾執行：

```sh
python3 scripts/serve_editor.py
```

- 網站預覽：http://127.0.0.1:4174/
- 文章工作室：http://127.0.0.1:4174/editor/

編輯期間保持該終端機開啟；結束時按 Control-C 停止服務。Python 需為 3.9 或更新版本，僅使用標準函式庫。若 4174 埠已被占用，可執行 `python3 scripts/serve_editor.py --port 4175`，再開啟輸出網址。

## 新增文章

1. 按「＋ 新增文章」，輸入標題、發表日期、分類、標籤與摘要。網址代稱會先提供預設值，也可改成容易辨識的英文名稱。
2. 在「視覺編輯」寫下內容；工具列可加入標題、段落、粗體、清單、連結、圖片及多行程式碼。需要精確控制格式時，切換「HTML 原始碼」。
3. 「上傳圖片」會將 PNG、JPEG、WebP 或 GIF 存入專案的 `images/uploads/`；單張限制 5 MB。也可以插入既有 `/images/` 路徑或 HTTPS 圖片網址。
4. 按「儲存草稿」存回專案。草稿不會出現在網站首頁或公開文章列表。瀏覽器另有自動暫存，用於恢復未存回專案的內容。
5. 按「預覽文章」檢查排版；確認後按「發布到本機網站」。這會重新產生文章、首頁、分類、標籤、日期歸檔、全文搜尋索引與 sitemap。
6. 上傳更新後的網站檔案到 GitHub，線上網站才會更新。

## 修改原有文章

從左側文章列表選取文章即可載入完整原文。修改時可儲存未發布草稿，公開版本繼續保留。按「發布到本機網站」才會更新公開版本。

已發布文章的日期與網址會固定，避免原有網址失效。每次發布前，本機會在 `content/revisions/` 保存當時的文章與網站資料。原版網站備份的位置則記錄於 `content/site.json` 的 `backup` 欄位，備份檔在本機工作區之外。

## 上傳 GitHub Pages

本次未推送或發布；這份工作區沒有 `.git` 資料夾。

首次換版請一併更新根目錄首頁、原有年份文章資料夾、`about/`、`pen/`、`archives/`、`categories/`、`tags/`、`page/`、新增的 `articles/`、`editor/`、`assets/`、`404.html`、`sitemap.xml` 和 `.nojekyll`。具體清單在 `design/upload-manifest.txt`；只上傳首頁會讓其他頁面保持舊版。

日後發布新文章後，同樣上傳產生的文章資料夾與更新後的列表、分類、標籤、歸檔、`assets/search-index.json`、首頁及 `sitemap.xml`。新增圖片時也要更新 `images/uploads/`。保留原有 `images/`、`lib/`、`py/`、`search.xml` 與 Google 驗證檔。

建議一併保存 `content/site.json`、`content/articles/`、`content/about.html`、`content/pen.html`、`scripts/`、啟動檔與本 README，供下次繼續編輯。`.gitignore` 排除了未公開草稿、修改前的本機版本、測試報告與暫存檔；這些資料留在自己的電腦。

## 直接調整版型與內容

- `scripts/build_site.py`：全站樣板、首頁文案、文章摘要與主題設定。
- `assets/site.css`：網站的色彩、字級、卡片、桌面與手機排版。
- `assets/site.js`：搜尋、文章篩選、手機選單、目錄、圖片放大與程式碼複製。
- `editor/`：文章工作室介面。
- `scripts/serve_editor.py`：只綁定 127.0.0.1 的本機文章儲存服務。
- `content/site.json`：文章資料；`content/articles/*.html` 是完整文章來源。
- `assets/illustrations/usagi-study-v2.png`：首頁 Usagi 學習插畫。
- `assets/illustrations/usagi-sticker.png`：新設計的透明 Usagi 學習貼圖，加入首頁、關於頁及閱讀側欄。

修改版型後執行：

```sh
python3 scripts/build_site.py
node --check assets/site.js
node --check editor/editor.js
python3 scripts/verify_site.py
```

原文雜湊保留在 `legacy_body_hashes`；文章工作室確認發布後更新 `current_body_hashes`，供驗證目前版本。若直接手動修改文章來源，需先確認實際差異再更新預期雜湊。不要為了讓未確認的修改通過而改寫雜湊。

這份工作區只有已產生的 Hexo 網頁，沒有原始 Hexo 專案；新版使用 `build_site.py` 重新產生，另用舊 Hexo 主題重建會覆蓋新版 HTML。

## 搜尋、圖像與驗證

導覽列的搜尋支援文章標題、分類與全文。全部文章頁可依主題與關鍵字篩選，Python 與 AI 主題允許重疊。原本 C#、C++ 共用分類／標籤網址，新版在該網址合併呈現兩者。

Usagi 外觀參考 [Chiikawa 官方角色頁](https://www.chiikawaofficial.com/characters/)，由內建 image_gen 製作新的主視覺與學習貼圖。這是永成的個人網站主題設計。完整提示詞與參考來源在 `design/image-prompts.json`。文章卡片小插畫為 SVG，可用 `python3 scripts/make_illustrations.py` 重新產生。

`design/static-verification.json` 記錄所有頁面的標題、H1、文章內容雜湊、原網址、站內資源與錨點檢查。`design/browser-verification.json` 記錄本機桌面、平板與手機尺寸的互動；`design/editor-verification.json` 記錄新增、草稿、修改、本機發布與讀回、還原測試。

以上是本機證據，並非 GitHub Pages 正式環境或實體手機驗收。原文的外站圖片、影片與雲端筆記權限依其原服務狀態，未逐一驗證外站存取。
