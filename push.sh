#!/bin/bash
# Деплой на GitHub + HuggingFace Spaces одной командой
cd "$(dirname "$0")"

MSG="${1:-обновление}"

echo "📦 Пушу в GitHub..."
git add .
git commit -m "$MSG" 2>/dev/null || echo "Нет изменений для коммита"
git push

echo "🚀 Загружаю на HuggingFace Spaces..."
hf upload lleeoon3/bg-remover . --type space \
  --exclude "venv/**" \
  --exclude "backgrounds/**" \
  --exclude "__pycache__/**" \
  --exclude "*.pyc" \
  --commit-message "$MSG"

echo ""
echo "✅ Готово!"
echo "🔗 https://lleeoon3-bg-remover.hf.space"
