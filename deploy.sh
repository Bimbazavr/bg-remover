#!/bin/bash
SERVER="212.57.118.111"
USER="root"

echo "=== Деплой bg-remover на $SERVER ==="

ssh $USER@$SERVER << 'EOF'
  set -e
  cd /root

  if [ ! -d "bg-remover" ]; then
    echo "Клонирую репозиторий..."
    git clone https://github.com/Bimbazavr/bg-remover.git
  else
    echo "Обновляю код..."
    cd bg-remover && git pull && cd ..
  fi

  cd bg-remover
  mkdir -p backgrounds

  echo "Собираю Docker-образ (первый раз ~10 мин, скачивается модель)..."
  docker compose build

  echo "Запускаю контейнер..."
  docker compose up -d

  echo ""
  echo "=== Готово! ==="
  echo "Сервис доступен: http://$HOSTNAME:8001"
  docker compose ps
EOF
