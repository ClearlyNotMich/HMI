@echo off
python "%~dp0vision_train.py" %*
exit /b %ERRORLEVEL%
