@echo off
REM Daily StockAI signal runner
REM Set your Discord webhook here (or leave blank):
set DISCORD_WEBHOOK=

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "KRONOS_PATH=C:\Users\Dream\Kronos"
set "IB_HOST=127.0.0.1"
set "IB_PORT=4002"
set "IBC_STARTER=C:\IBC\StartGateway.bat"
set "IB_WAIT_SECONDS=120"
set "IB_AUTO_START_ALLOWED="

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "C:\Users\Dream\AppData\Local\Programs\Python\Python314\python.exe" (
  set "PYTHON_EXE=C:\Users\Dream\AppData\Local\Programs\Python\Python314\python.exe"
) else if exist "C:\Users\Dream\AppData\Local\Programs\Python\Python38\python.exe" (
  set "PYTHON_EXE=C:\Users\Dream\AppData\Local\Programs\Python\Python38\python.exe"
) else (
  set "PYTHON_EXE=py -3"
)

echo.>> "%SCRIPT_DIR%signal_log.txt"
echo ================================================================>> "%SCRIPT_DIR%signal_log.txt"
echo [%date% %time%] StockAI daily launcher starting>> "%SCRIPT_DIR%signal_log.txt"

for /f %%A in ('powershell -NoProfile -Command "$now = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), 'Eastern Standard Time'); $weekday = $now.DayOfWeek -ge [System.DayOfWeek]::Monday -and $now.DayOfWeek -le [System.DayOfWeek]::Friday; $minutes = ($now.Hour * 60) + $now.Minute; if ($weekday -and $minutes -ge 510 -and $minutes -le 990) { '1' }"' ) do set "IB_AUTO_START_ALLOWED=%%A"

set "IB_READY="
for /f %%A in ('powershell -NoProfile -Command "$ok = Test-NetConnection -ComputerName '%IB_HOST%' -Port %IB_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue; if ($ok) { '1' }"' ) do set "IB_READY=%%A"

if "%IB_READY%"=="1" (
  echo [%date% %time%] IBKR already listening on %IB_HOST%:%IB_PORT%>> "%SCRIPT_DIR%signal_log.txt"
) else if not "%IB_AUTO_START_ALLOWED%"=="1" (
  echo [%date% %time%] Outside weekday market window; skipping IBKR auto-start>> "%SCRIPT_DIR%signal_log.txt"
) else (
  echo [%date% %time%] IBKR not listening on %IB_HOST%:%IB_PORT%; attempting startup via "%IBC_STARTER%">> "%SCRIPT_DIR%signal_log.txt"
  if exist "%IBC_STARTER%" (
    call "%IBC_STARTER%" /INLINE /NOICON
  ) else (
    echo [%date% %time%] WARNING: IBKR starter not found at "%IBC_STARTER%">> "%SCRIPT_DIR%signal_log.txt"
  )

  powershell -NoProfile -Command ^
    "$deadline = (Get-Date).AddSeconds(%IB_WAIT_SECONDS%);" ^
    "while ((Get-Date) -lt $deadline) {" ^
    "  if (Test-NetConnection -ComputerName '%IB_HOST%' -Port %IB_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue) { exit 0 }" ^
    "  Start-Sleep -Seconds 2" ^
    "}" ^
    "exit 1"

  if errorlevel 1 (
    echo [%date% %time%] WARNING: IBKR did not open %IB_HOST%:%IB_PORT% within %IB_WAIT_SECONDS%s; continuing anyway>> "%SCRIPT_DIR%signal_log.txt"
  ) else (
    echo [%date% %time%] IBKR is now listening on %IB_HOST%:%IB_PORT%>> "%SCRIPT_DIR%signal_log.txt"
  )
)

%PYTHON_EXE% live_signal.py >> "%SCRIPT_DIR%signal_log.txt" 2>&1
