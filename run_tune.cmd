@echo off
REM Finish the hyperparameter sweep, under Task Scheduler or by hand.
REM
REM Capacity first: hdim384 and hdim512 are what decide whether 256 is an
REM optimum or merely the largest setting anyone got round to testing,
REM which is the difference between a measured bound and a loose one. The
REM remaining four factors follow once that question is closed.
REM
REM cd /d is required: a scheduled task starts in system32.
cd /d "%~dp0"
python run_tune.py --no-live --only hdim384 hdim512 >> tune_console.txt 2>&1
python run_tune.py --no-live --only lr1e-3 lr2e-4 plen800 sparse1e-4 >> tune_console.txt 2>&1
python run_tune.py --report >> tune_console.txt 2>&1
