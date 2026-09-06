@echo off
setlocal enabledelayedexpansion
title SOUL-LLM: Train with Qwen LLM Teacher
cls

echo =====================================================================
echo    SOUL-LLM: Knowledge Distillation and Training with Qwen LLM
echo =====================================================================
echo.
echo  This tool uses the local Qwen LLM (qwen2.5:7b via Ollama) to generate
echo  deep educational and reasoning knowledge, and then trains the
echo  independent SOUL-LLM Transformer model on that knowledge.
echo.
echo  Options:
echo   [1] Quick Training   (5 Qwen prompt samples, 10 training epochs)
echo   [2] Full Training    (All 28 Qwen educational prompts, 25 epochs)
echo   [3] Custom Training  (Specify prompt count and epochs)
echo   [4] Generate Only    (Generate Qwen knowledge without training)
echo   [5] Verify Status    (Run SOUL-LLM Independence Audit)
echo   [6] Exit
echo.
set /p choice="Select an option (1-6) [default: 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="1" goto QUICK
if "%choice%"=="2" goto FULL
if "%choice%"=="3" goto CUSTOM
if "%choice%"=="4" goto GEN_ONLY
if "%choice%"=="5" goto VERIFY
if "%choice%"=="6" goto END

:QUICK
echo.
echo [1/3] Starting Quick Distillation (5 Qwen Prompts, 10 Epochs)...
python distill.py --samples 5 --epochs 10
goto FINISHED

:FULL
echo.
echo [1/3] Starting Full Distillation (All 28 Qwen Prompts, 25 Epochs)...
python distill.py --epochs 25
goto FINISHED

:CUSTOM
set /p custom_samples="How many Qwen prompt samples (e.g. 5, 10, 28): "
set /p custom_epochs="How many training epochs (e.g. 10, 25, 50): "
echo.
echo Starting Custom Distillation (%custom_samples% samples, %custom_epochs% epochs)...
python distill.py --samples %custom_samples% --epochs %custom_epochs%
goto FINISHED

:GEN_ONLY
set /p gen_samples="How many Qwen prompt samples to generate: "
python distill.py --generate-only --samples %gen_samples%
goto FINISHED

:VERIFY
echo.
echo Running Independence Verification Audit...
python verify_independence.py
pause
goto END

:FINISHED
echo.
echo =====================================================================
echo  Distillation Complete! Running SOUL-LLM Independence Audit...
echo =====================================================================
python verify_independence.py
echo.
echo You can now run the web interface with: python app.py
pause

:END
