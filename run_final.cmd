@echo off
REM Final Davis benchmark, five canonical folds at H_DIM=256.
REM
REM Launched by Task Scheduler as well as by hand. Everything is written to
REM final_log.txt, so a scheduled run with no console still leaves a full
REM record. cd /d is required: a scheduled task starts in system32.
cd /d "%~dp0"
call conda activate ibdta 2>nul
python run_final.py --no-live --log final_log.txt >> final_console.txt 2>&1
