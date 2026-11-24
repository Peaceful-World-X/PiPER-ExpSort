@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0" || exit /b 1
if exist "renders\." goto ready
mkdir "renders"
if errorlevel 1 goto done
:ready
if not "%~1"=="" goto render_only
set "fixed_native=renders\.comparison\fixed_native.mp4"
call "source\run_isaac.bat" "%~dp0source\render_isaac.py" --path-tracing --movie "%fixed_native%"
if errorlevel 1 goto done
".venv\Scripts\python.exe" -X utf8 "source\finish.py" --exposure "%fixed_native%"
if errorlevel 1 goto done
call "source\run_isaac.bat" "%~dp0source\render_isaac.py" --path-tracing --comparison-ending --movie "renders\.comparison\ending_native.mp4"
if errorlevel 1 goto done
".venv\Scripts\python.exe" -X utf8 "source\finish.py" --grade-ending "renders\.comparison\ending_native.mp4" --compare
if errorlevel 1 goto done
".venv\Scripts\python.exe" -X utf8 "source\grasp_review.py"
goto done
:render_only
call "source\run_isaac.bat" "%~dp0source\render_isaac.py" --path-tracing %*
:done
set "run_status=%errorlevel%"
popd
exit /b %run_status%
