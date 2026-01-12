@echo off
setlocal

set GCLOUD=C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd
set REGION=us-central1
set PROJECT=gen-lang-client-0213220308
set IMAGE=%REGION%-docker.pkg.dev/%PROJECT%/chess-images/chess-migration:latest

echo === Rebuilding Docker Image with Cloud Build ===

REM Copy source files to cloud directory
echo Copying source files...
copy /Y "..\config.py" "." >nul
copy /Y "..\utils.py" "." >nul
copy /Y "..\data_utils.py" "." >nul
copy /Y "..\generate_data.py" "." >nul
copy /Y "..\cloud_generate.py" "." >nul
copy /Y "..\migrate_mpv_moves.py" "." >nul

echo Building with Cloud Build (this may take 2-3 minutes)...
call "%GCLOUD%" builds submit --tag %IMAGE% .

if %ERRORLEVEL% EQU 0 (
    echo [OK] Image built and pushed successfully!
) else (
    echo [FAIL] Build failed
)

REM Cleanup
echo Cleaning up...
del config.py utils.py data_utils.py generate_data.py cloud_generate.py migrate_mpv_moves.py 2>nul

echo Done!
