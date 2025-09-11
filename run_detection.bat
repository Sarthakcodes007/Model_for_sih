@echo off
REM Simple batch script for Yellow Rust Detection
REM This script provides an easy way to run rust detection on Windows

echo ========================================
echo    Yellow Rust Detection System
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "run_inference.py" (
    echo ERROR: run_inference.py not found
    echo Please run this script from the yellow_rust_segmentation directory
    pause
    exit /b 1
)

REM Check if model exists
if not exist "models\checkpoints\best.pth" (
    echo ERROR: Model file not found at models\checkpoints\best.pth
    echo Please ensure the model is trained and saved.
    pause
    exit /b 1
)

echo Model found. Ready to detect rust!
echo.

REM Show menu
:menu
echo Choose an option:
echo 1. Analyze single image
echo 2. Analyze folder of images
echo 3. Run demo
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto single_image
if "%choice%"=="2" goto batch_images
if "%choice%"=="3" goto run_demo
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
echo.
goto menu

:single_image
echo.
set /p image_path="Enter path to image file: "
if not exist "%image_path%" (
    echo ERROR: Image file not found: %image_path%
    echo.
    goto menu
)

echo.
echo Choose detection sensitivity:
echo 1. High sensitivity (threshold 0.2) - detects more rust
echo 2. Balanced (threshold 0.3) - recommended
echo 3. Conservative (threshold 0.5) - detects only obvious rust
set /p sens="Enter choice (1-3, default 2): "

if "%sens%"=="1" set threshold=0.2
if "%sens%"=="3" set threshold=0.5
if "%sens%"=="" set threshold=0.3
if not defined threshold set threshold=0.3

echo.
echo Analyzing image with threshold %threshold%...
python run_inference.py --image "%image_path%" --threshold %threshold%

echo.
echo Analysis complete! Check the 'inference_results' folder for output images.
echo.
pause
goto menu

:batch_images
echo.
set /p folder_path="Enter path to folder containing images: "
if not exist "%folder_path%" (
    echo ERROR: Folder not found: %folder_path%
    echo.
    goto menu
)

echo.
echo Choose detection sensitivity:
echo 1. High sensitivity (threshold 0.2)
echo 2. Balanced (threshold 0.3) - recommended
echo 3. Conservative (threshold 0.5)
set /p sens="Enter choice (1-3, default 2): "

if "%sens%"=="1" set threshold=0.2
if "%sens%"=="3" set threshold=0.5
if "%sens%"=="" set threshold=0.3
if not defined threshold set threshold=0.3

echo.
echo Processing all images in folder with threshold %threshold%...
echo This may take a while depending on the number of images.
python run_inference.py --batch "%folder_path%" --threshold %threshold%

echo.
echo Batch processing complete! Check the 'inference_results' folder for outputs.
echo.
pause
goto menu

:run_demo
echo.
echo Running demonstration...
echo This will show different features of the detection system.
echo.
python quick_demo.py

echo.
pause
goto menu

:exit
echo.
echo Thank you for using Yellow Rust Detection System!
echo.
pause
exit /b 0