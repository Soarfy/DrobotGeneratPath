@echo off
setlocal
set OMNI_KIT_ACCEPT_EULA=YES
set PYTHON=D:\SoftWare\Anaconda\envs\isaacsim\python.exe
cd /d D:\CompanyProject\DrobotGeneratPath
echo Launching Isaac Sim (headless smoke test)...
"%PYTHON%" -c "import os; os.environ['OMNI_KIT_ACCEPT_EULA']='YES'; from isaacsim import SimulationApp; app=SimulationApp({'headless': True}); import omni.usd; print('STAGE:', omni.usd.get_context().get_stage()); print('ISAAC_READY'); app.close()" > out\isaac_launch.log 2>&1
echo Exit code: %errorlevel%
more out\isaac_launch.log | findstr /C:"ISAAC_READY" /C:"STAGE:" /C:"Error" /C:"Traceback"
