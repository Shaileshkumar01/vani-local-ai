@echo off
REM Wrapper around uv that keeps the (multi-GB) virtualenv OUTSIDE OneDrive,
REM so OneDrive doesn't try to sync/lock thousands of library files.
REM Usage:  run.bat sync          (install everything)
REM         run.bat run v2v       (start the voice agent)
set UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\voice-to-voice

set ARGS=%*
if "%~1"=="" set ARGS=run --no-sync v2v
if "%~1"=="app" set ARGS=run --no-sync v2v --app %2 %3 %4
if "%~1"=="mini" set ARGS=run --no-sync v2v --app --mini
if "%~1"=="run" set ARGS=run --no-sync %2 %3 %4 %5 %6 %7 %8 %9

where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    uv %ARGS%
    exit /b %ERRORLEVEL%
)

if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" (
    "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" %ARGS%
    exit /b %ERRORLEVEL%
)

for /d %%i in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv*") do (
    if exist "%%i\uv.exe" (
        "%%i\uv.exe" %ARGS%
        exit /b %ERRORLEVEL%
    )
)

echo [ERROR] uv.exe not found!
exit /b 1
