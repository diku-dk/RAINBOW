@echo off
setlocal

rem set SDF_COUNTS=100 200 400 600 800
set SDF_COUNTS=100 400 800

for %%a in (%SDF_COUNTS%) do (
    echo Starting experiment %%a
    start cmd /c "python backlash_experiment.py %%a --headless --steps 200 1> backlash_%%a.txt 2>&1"
)
