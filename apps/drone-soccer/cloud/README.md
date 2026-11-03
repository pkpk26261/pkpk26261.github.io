# FLIGHT CLUB 雲端多人版

部署完成後，主持人與玩家都只需要瀏覽器。雲端主機執行原本的飛控、碰撞、AI、計分及正式賽引擎；每個房間保留獨立比賽狀態。玩家不需下載遊戲、安裝啟動器或信任私人憑證。搖桿仍需由各裝置的作業系統辨識，網頁再透過 Gamepad API 讀取。

## 網站的日常使用

1. 主持人開啟部署後的 HTTPS 網址，加上 `/admin`，在「主持人登入」輸入管理金鑰。
2. 按「建立比賽」，選擇賽場、紅藍編成與 AI 難度，再進入主持人房間。
3. 分享房間的玩家網址或 QR。玩家開啟 `/play?code=房碼`，輸入暱稱、選擇席位，確認搖桿後準備。
4. 主持人確認真人玩家全部就緒，再開始比賽。觀眾可由公開大廳進入獨立轉播。
5. 結束本場後釋放賽場，可繼續建立下一場；關閉主持人的電腦不會停止雲端服務。

只有需要管理所有賽場的人持有 `ADMIN_TOKEN`。玩家與觀眾只需房間邀請，不要分享管理金鑰。此版本採主持人管理賽場，沒有開放匿名訪客任意建立伺服器資源。

## 使用 Render 部署

根目錄的 `Dockerfile` 與 `render.yaml` 已提供雲端部署設定。GitHub 保存程式，Render 執行 Node 與 Chromium，對外提供固定 HTTPS／WSS 網址；不使用 GitHub Pages 執行多人伺服器。

**建立前需確認費用。** 模板選用 `1c-2g`（1 CPU、2 GB RAM），並掛載 1 GB 持久磁碟。依 2026-11-03 查閱的官方價格，運算費為 US$25／月，磁碟與可能的額外用量另計；最終以 Render 建立服務時顯示的費用為準。這是部署起點，尚未證明雲端可同時穩定運行兩場比賽。[Render 價格](https://render.com/pricing)

1. 雲端發佈使用 [pkpk26261/pkpk26261.github.io 的 flight-club-cloud 分支](https://github.com/pkpk26261/pkpk26261.github.io/tree/flight-club-cloud)。這個分支只放乾淨的遊戲與伺服器原始碼，保留根目錄 `Dockerfile`、`render.yaml` 與它們引用的檔案；原有 `master` 分支的 GitHub Pages 網站不受影響。本機 runtime、`node_modules`、憑證、瀏覽器設定檔與比賽資料不需上傳。
2. 登入 [Render Dashboard](https://dashboard.render.com/)，選擇 **New → Blueprint**，連接 `pkpk26261/pkpk26261.github.io`，並選擇 `flight-club-cloud` 分支。
3. 檢查區域為 Singapore、實例數為 1、方案與磁碟費用。`ADMIN_TOKEN` 欄位填入至少 32 字元的隨機管理金鑰。金鑰只保存在 Render 的秘密環境變數，禁止寫入 GitHub。
4. 確認建立，等待 Docker 建置與 `/api/health` 健康檢查通過。Render 會設定 `RENDER_EXTERNAL_URL`，伺服器使用它產生同來源的 HTTPS／WSS 邀請，不需要手填或猜測網站網域。
5. 開啟 Render 回傳的實際網址，測試主持人登入、建房、兩台裝置加入、搖桿準備、開賽及重連。完成實測前，不把本機自動化結果視為雲端或實體設備驗收。

自動部署預設關閉，避免推送程式時中斷正在進行的比賽。更新前先結束活動，再在 Render 手動部署已檢查的版本。官方欄位定義見 [Blueprint 規格](https://render.com/docs/blueprint-spec)。

## 設定與資料保存

| 設定 | 用途 |
| --- | --- |
| `PORT` | 容器內 HTTP 埠；模板為 `10000`。 |
| `PUBLIC_ORIGIN` | 最終公開 HTTPS 來源，不含路徑或房碼。Render 可省略，改用自動提供的 `RENDER_EXTERNAL_URL`。 |
| `ADMIN_TOKEN` | 至少 32 字元的管理金鑰；只放服務的秘密環境變數。 |
| `DATA_DIR` | 比賽與服務資料目錄；模板為持久磁碟上的 `/data`。 |
| `MAX_ARENAS` | 賽場上限；模板預設 `2`，實際可用量需以雲端測試確認。 |

伺服器使用 `--cloud` 模式，於容器內接收 HTTP；平台代理負責對外 HTTPS 與 WSS。其他容器主機也必須提供固定公開 HTTPS、WebSocket 升級及持久資料目錄，不可把容器的 HTTP 埠直接當作玩家入口。

此架構只部署 **一個服務實例**。房間即時狀態與控制器輸入在該實例記憶體中，不能直接增加水平實例數。持久磁碟保存房間與檢查點，重新部署或重啟仍會中斷連線；已開始的房間以暫停狀態復原，原玩家需重連、回中並重新準備。磁碟只有掛載路徑下的檔案能跨重啟保留。[Render 持久磁碟說明](https://render.com/docs/disks)

## 容器建置

從專案根目錄建置：

```powershell
docker build -t flight-club-cloud .
```

映像固定使用 `mcr.microsoft.com/playwright:v1.56.1-noble`，與 `lan/package.json` 的 Playwright `1.56.1` 一致。之後升級 Playwright 時必須一起更新映像版本。建置安裝 production 套件，遊戲 runtime、Windows 啟動器、Cloudflare Tunnel 與本機憑證都不在映像內。

`Dockerfile` 的健康檢查只確認服務與引擎啟動狀態；它不代表實際 Wi-Fi 延遲、控制器相容性、長時間運行或雲端效能已通過驗收。

## Linux 容器驗證

雲端分支的 GitHub Actions `Cloud game verification` 會建置 Linux Docker 映像，先檢查雲端啟動、授權與資料保存，再透過隔離的測試 HTTPS 代理，驗證一位主持人與兩位玩家的瀏覽器流程。測試的搖桿訊號為合成輸入，測試憑證不會用於正式服務。驗證截圖與報告保存在該次 workflow 的 `cloud-browser-verification` artifact。

Linux 映像建置與伺服器啟動、授權、資料保存檢查已通過。容器中的模擬玩家各使用獨立 Chromium 程序，採 SwANGLE 軟體繪圖、800×600 觸控配置及 4 FPS，以限制沒有顯示卡的 CI 主機繪圖負載。這是功能流程驗證，不代表預設畫質與更新率的效能驗收。伺服器的權威引擎、約 20 Hz 狀態同步與每 32 ms 的搖桿輸入流程維持原設定；測試配置不套用到正式玩家頁面。

瀏覽器流程仍以 [Actions 中該次提交的實際結果](https://github.com/pkpk26261/pkpk26261.github.io/actions) 為準，尚未將修正後的完整 workflow 宣稱為通過。正式雲端服務尚未上線；即使 Linux 檢查全部通過，公開 HTTPS／WSS、多裝置實體搖桿、網路延遲及長時間運行仍需在實際服務確認。
