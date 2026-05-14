#!/bin/bash
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "Создаю виртуальное окружение..."
  python3 -m venv venv
  venv/bin/pip install -r requirements.txt
fi

echo "Запускаю сервер на http://localhost:8001"
echo "Открой браузер: http://localhost:8001"
venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload
