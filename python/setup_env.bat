@echo off
REM AI Music Detector - Windows Environment Setup
REM This script creates a conda environment with all dependencies

echo ============================================
echo AI Music Detector - Environment Setup
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Miniconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Found conda installation
echo.

REM Check if environment already exists
conda env list | findstr /C:"ai-music-detector" >nul
if %ERRORLEVEL% EQU 0 (
    echo Environment 'ai-music-detector' already exists.
    set /p OVERWRITE="Do you want to remove and recreate it? (y/N): "
    if /i "%OVERWRITE%"=="y" (
        echo Removing existing environment...
        conda env remove -n ai-music-detector -y
    ) else (
        echo Skipping environment creation.
        goto :activate
    )
)

echo Creating conda environment...
echo This may take several minutes...
echo.

conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to create environment.
    echo Trying alternative setup with pip only...
    
    conda create -n ai-music-detector python=3.11 -y
    conda activate ai-music-detector
    pip install -r requirements.txt
)

:activate
echo.
echo ============================================
echo Environment created successfully!
echo ============================================
echo.
echo To activate the environment, run:
echo   conda activate ai-music-detector
echo.
echo Then you can run the training pipeline:
echo   python download_data.py --dataset all
echo   python extract_fakeprints.py --input ... --output ... --label ...
echo   python train_model.py --real ... --fake ...
echo   python export_onnx.py
echo.

pause
