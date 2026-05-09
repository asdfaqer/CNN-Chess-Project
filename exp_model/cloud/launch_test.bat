@echo off
setlocal

set GCLOUD=C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd
set GSUTIL=C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd
set BUCKET=chess-migration-gen-lang-c
set ZONE=us-central1-a
set REGION=us-central1
set IMAGE=us-central1-docker.pkg.dev/gen-lang-client-0213220308/chess-images/chess-migration:latest

echo === Launching Test Generation VM ===

REM Create startup script
echo #!/bin/bash > startup.sh
echo set -ex >> startup.sh
echo curl -fsSL https://get.docker.com ^| sh >> startup.sh
echo gcloud auth configure-docker %REGION%-docker.pkg.dev --quiet >> startup.sh
echo mkdir -p /data/output >> startup.sh
echo docker run --rm -v /data/output:/output %IMAGE% python cloud_generate.py --count 1000 --batch-name batch_mpv_test.zst --output-dir /output >> startup.sh
echo gsutil cp /data/output/batch_mpv_test.zst gs://%BUCKET%/input/ >> startup.sh
echo shutdown -h now >> startup.sh

echo Launching VM...
call "%GCLOUD%" compute instances create gen-test-vm --zone=%ZONE% --machine-type=e2-standard-4 --provisioning-model=SPOT --instance-termination-action=DELETE --scopes=cloud-platform --metadata-from-file=startup-script=startup.sh --boot-disk-size=30GB --image-family=debian-12 --image-project=debian-cloud

if %ERRORLEVEL% EQU 0 (
    echo [OK] VM launched successfully!
    echo Monitor with: "%GSUTIL%" ls gs://%BUCKET%/input/
) else (
    echo [FAIL] VM launch failed
)

del startup.sh 2>nul
