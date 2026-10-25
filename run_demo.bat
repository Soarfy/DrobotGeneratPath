@echo off
REM ============================================================
REM  DobotPathGeneration - one-click demo
REM
REM    run_demo.bat             plan + render video + open viewer
REM    run_demo.bat raster      same, for the single-tooth raster task
REM    run_demo.bat isaac       also try to launch Isaac Sim
REM
REM  Adjust PYTHON_BASE below if your Anaconda lives elsewhere.
REM ============================================================

setlocal enabledelayedexpansion
set ROOT=%~dp0
set PYTHON_BASE=D:\SoftWare\Anaconda
set PY=%PYTHON_BASE%\python.exe
set PYRENDER=%PYTHON_BASE%\envs\Uclean\python.exe
set PYISAAC=%PYTHON_BASE%\envs\isaacsim\python.exe

set TASK=config\tasks\arch_walk.yaml
set NAME=arch_lower_jaw
set MODE=%1

if /I "%MODE%"=="raster" (
    set TASK=config\tasks\demo_occlusal_patch.yaml
    set NAME=trajectory
    set MODE=%2
)

cd /d "%ROOT%"

echo.
echo ============================================================
echo  [1/4] Planning trajectory
echo ============================================================
"%PY%" -m dobot_pathgen.cli plan --task "%TASK%" --out "out\!NAME!.json"
if errorlevel 1 (
    echo.
    echo Planning failed. Nothing else will run.
    goto :end
)

echo.
echo ============================================================
echo  [2/4] Preflight checks (limits, FK replay, tool clearance)
echo ============================================================
"%PY%" scripts\preflight.py --trajectory "out\!NAME!.json" --skip-network

echo.
echo ============================================================
echo  [3/4] Rendering offline video (PyVista)
echo ============================================================
"%PYRENDER%" scripts\render_vtk.py --trajectory "out\!NAME!.json" --out out\vtk_preview --frames 80 --speed 3
if errorlevel 1 echo   (video render failed - the 3D viewer below still works)

echo.
echo ============================================================
echo  [4/4] Building the interactive 3D viewer
echo ============================================================
"%PY%" scripts\export_viewer.py --trajectory "out\!NAME!.json" --out out\viewer --title "!NAME!"

echo.
echo Opening the viewer (local server on port 8766)...
start "" "%ROOT%open_viewer.bat"

if /I "%MODE%"=="isaac" (
    echo.
    echo ============================================================
    echo  Optional: Isaac Sim replay
    echo ============================================================
    set OMNI_KIT_ACCEPT_EULA=YES
    "%PYISAAC%" launch_isaac_sim.py --config config\robot.yaml --trajectory "out\!NAME!.json"
)

echo.
echo ============================================================
echo  Done.
echo    trajectory : out\!NAME!.json
echo    video      : out\vtk_preview\robot_path.mp4
echo    viewer     : out\viewer\viewer.html
echo ============================================================

:end
endlocal
