@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Verifying EdgeFall-ST installation...
python tools\verify_install.py --device auto
if errorlevel 1 (
    echo Installation verification failed.
    exit /b 1
)

echo [2/2] Running the bundled demo videos...
python -u infer.py --input demo\source --device auto
if errorlevel 1 (
    echo Demo inference failed.
    exit /b 1
)

echo Demo completed. Results are under the results directory.
exit /b 0
