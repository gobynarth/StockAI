@echo off
REM Daily StockAI signal runner
REM Set your Discord webhook here (or leave blank):
set DISCORD_WEBHOOK=

cd /d C:\Users\Dream\StockAI
python live_signal.py >> C:\Users\Dream\StockAI\signal_log.txt 2>&1
