@echo off
REM Serve the viewer over HTTP and open it in the default browser.
REM
REM A browser refuses to fetch viewer_data.json from a file:// page, so the
REM viewer must be served. This starts a tiny HTTP server (port 8766) and opens
REM the page; close this window to stop the server.

setlocal
set ROOT=%~dp0
set PYTHON_BASE=D:\SoftWare\Anaconda

cd /d "%ROOT%out\viewer"

if not exist viewer.html (
    echo Viewer not built yet. Run run_demo.bat first.
    pause
    exit /b 1
)

echo Starting local server on http://127.0.0.1:8766
echo Opening http://127.0.0.1:8766/viewer.html
echo Close this window to stop the server.
echo.

start "" "http://127.0.0.1:8766/viewer.html"
"%PYTHON_BASE%\python.exe" -m http.server 8766

endlocal
