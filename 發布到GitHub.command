#!/bin/zsh
# 一鍵發布：重建網站 → 檢查 → 驗證 → 提交 → 推送到 GitHub Pages
cd "$(dirname "$0")" || exit 1

pause() { print -n "\n按 Enter 關閉視窗…"; read _; }
fail() { print "\n✗ $1"; pause; exit 1; }

print "▶ 1/4 重建網站…"
python3 scripts/build_site.py || fail "建置失敗，已中止。"

print "\n▶ 2/4 檢查前端腳本…"
if command -v node >/dev/null 2>&1; then
  node --check assets/site.js || fail "site.js 語法錯誤，已中止。"
  node --check editor/editor.js || fail "editor.js 語法錯誤，已中止。"
else
  print "（略過：未安裝 node，將不檢查 JS 語法）"
fi

print "\n▶ 3/4 驗證網站完整性…"
python3 scripts/verify_site.py || fail "驗證未通過，已中止。請先修正問題再發布。"

print "\n▶ 4/4 提交並推送到 GitHub…"
[ -d .git ] || fail "尚未初始化 Git。請先在終端機執行：git init"

if ! git remote get-url origin >/dev/null 2>&1; then
  print "尚未設定遠端儲存庫，請先在終端機執行（擇一，依你的驗證方式）："
  print "    git remote add origin https://github.com/pkpk26261/pkpk26261.github.io.git"
  print "    # 或使用 SSH："
  print "    git remote add origin git@github.com:pkpk26261/pkpk26261.github.io.git"
  fail "設定好遠端後再重新執行本腳本。"
fi

git add -A
if git diff --cached --quiet; then
  print "（沒有變更需要提交，直接嘗試推送最新狀態）"
else
  git commit -m "更新網站內容 $(date '+%Y-%m-%d %H:%M')" || fail "提交失敗。"
fi

branch=$(git symbolic-ref --short HEAD 2>/dev/null || print main)
git push -u origin "$branch" || fail "推送失敗，請檢查網路連線或 GitHub 登入權限。"

print "\n✓ 發布完成！GitHub Pages 通常會在數分鐘內更新線上網站。"
pause
