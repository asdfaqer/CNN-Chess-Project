"""
Full GCP Migration Test - Tests the complete workflow with a small sample.
"""
import os
import subprocess
import sys
import time

# Configuration
PROJECT_ID = "gen-lang-client-0213220308"
REGION = "us-central1"
ZONE = f"{REGION}-a"
BUCKET_NAME = "chess-migration-gen-lang-c"
IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/chess-images/chess-migration:latest"

# Paths
GCLOUD = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
GSUTIL = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd"
CLOUD_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CLOUD_DIR)

def run(cmd, check=True):
    """Run command via cmd.exe and return output."""
    full_cmd = f'cmd /c "{cmd}"'
    print(f"$ {cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0, result.stdout

def create_artifact_registry():
    """Create Artifact Registry repository for Docker images."""
    print("\n=== Creating Artifact Registry ===")
    # Check if repo exists
    ok, out = run(f'"{GCLOUD}" artifacts repositories describe chess-images --location={REGION}', check=False)
    if ok:
        print("[OK] Repository already exists")
        return True
    
    print("Creating repository...")
    ok, _ = run(f'"{GCLOUD}" artifacts repositories create chess-images --repository-format=docker --location={REGION} --description="Chess migration images"')
    if ok:
        print("[OK] Repository created")
        return True
    print("[FAIL] Failed to create repository")
    return False

def copy_source_files():
    """Copy source files to cloud directory for Docker build."""
    import shutil
    required_files = ['config.py', 'utils.py', 'data_utils.py', 'migrate_mpv_moves.py']
    print("\n=== Copying source files ===")
    for f in required_files:
        src = os.path.join(PARENT_DIR, f)
        dst = os.path.join(CLOUD_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {f}")
        else:
            print(f"  [WARN] {f} not found")
    return True

def cleanup_source_files():
    """Remove copied source files."""
    required_files = ['config.py', 'utils.py', 'data_utils.py', 'migrate_mpv_moves.py']
    for f in required_files:
        path = os.path.join(CLOUD_DIR, f)
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

def build_docker_image():
    """Build Docker image using Cloud Build."""
    print("\n=== Building Docker Image with Cloud Build ===")
    print("(This builds in the cloud - no local Docker needed)")
    
    copy_source_files()
    
    # Change to cloud directory
    original_dir = os.getcwd()
    os.chdir(CLOUD_DIR)
    
    try:
        ok, _ = run(f'"{GCLOUD}" builds submit --tag {IMAGE_NAME} .')
        if ok:
            print("[OK] Image built and pushed successfully!")
            return True
        else:
            print("[FAIL] Build failed")
            return False
    finally:
        os.chdir(original_dir)
        cleanup_source_files()

def upload_test_batch():
    """Upload the test sample batch to GCS."""
    print("\n=== Uploading Test Batch ===")
    
    # First create a small test sample if it doesn't exist
    test_path = os.path.join(PARENT_DIR, "generated_data", "test_verbose_sample.zst")
    if not os.path.exists(test_path):
        print("Creating test sample first...")
        os.chdir(PARENT_DIR)
        run(f'python migrate_mpv_moves.py --test')
    
    if not os.path.exists(test_path):
        print("[FAIL] Test sample not found")
        return False
    
    ok, _ = run(f'"{GSUTIL}" cp "{test_path}" gs://{BUCKET_NAME}/input/')
    if ok:
        print("[OK] Test batch uploaded")
        return True
    print("[FAIL] Upload failed")
    return False

def launch_test_vm():
    """Launch a small Spot VM to test the migration."""
    print("\n=== Launching Test VM ===")
    
    batch_name = "test_verbose_sample.zst"
    vm_name = "test-migration-vm"
    
    # Create startup script
    startup_script = f'''#!/bin/bash
set -ex

echo "=== Chess Data Migration Test VM ==="
echo "Batch: {batch_name}"

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with Artifact Registry
gcloud auth configure-docker {REGION}-docker.pkg.dev --quiet

# Download input batch
mkdir -p /data/input /data/output
gsutil cp gs://{BUCKET_NAME}/input/{batch_name} /data/input/

# Run migration
docker run --rm \\
    -v /data/input:/input \\
    -v /data/output:/output \\
    {IMAGE_NAME} \\
    python migrate_mpv_moves.py --batch /input/{batch_name} --output /output/{batch_name}

# Upload result
gsutil cp /data/output/{batch_name} gs://{BUCKET_NAME}/output/

# Signal completion
echo "MIGRATION_COMPLETE" > /tmp/done.txt
gsutil cp /tmp/done.txt gs://{BUCKET_NAME}/status/{batch_name}.done

echo "=== Migration Complete ==="

# Self-terminate
shutdown -h now
'''
    
    script_path = os.path.join(CLOUD_DIR, "startup.sh")
    with open(script_path, "w", newline='\n') as f:
        f.write(startup_script)
    
    # Delete existing VM if any
    run(f'"{GCLOUD}" compute instances delete {vm_name} --zone={ZONE} --quiet', check=False)
    
    # Launch VM
    cmd = f'"{GCLOUD}" compute instances create {vm_name} --zone={ZONE} --machine-type=e2-standard-4 --provisioning-model=SPOT --instance-termination-action=DELETE --scopes=cloud-platform --metadata-from-file=startup-script={script_path} --boot-disk-size=30GB --image-family=debian-12 --image-project=debian-cloud'
    
    ok, _ = run(cmd)
    if ok:
        print("[OK] VM launched!")
        print(f"    Monitor: gsutil ls gs://{BUCKET_NAME}/status/")
        print(f"    SSH: gcloud compute ssh {vm_name} --zone={ZONE}")
        return True
    print("[FAIL] Failed to launch VM")
    return False

def check_status():
    """Check if migration is complete."""
    print("\n=== Checking Migration Status ===")
    
    batch_name = "test_verbose_sample.zst"
    ok, out = run(f'"{GSUTIL}" ls gs://{BUCKET_NAME}/status/{batch_name}.done', check=False)
    
    if ok and batch_name in out:
        print("[OK] Migration complete!")
        return True
    else:
        print("[...] Migration still in progress")
        # Check VM status
        run(f'"{GCLOUD}" compute instances describe test-migration-vm --zone={ZONE} --format="get(status)"', check=False)
        return False

def download_result():
    """Download the migrated test batch."""
    print("\n=== Downloading Result ===")
    
    batch_name = "test_verbose_sample.zst"
    output_path = os.path.join(PARENT_DIR, "generated_data", "gcp_migrated_test.zst")
    
    ok, _ = run(f'"{GSUTIL}" cp gs://{BUCKET_NAME}/output/{batch_name} "{output_path}"')
    if ok:
        print(f"[OK] Downloaded to {output_path}")
        return True
    print("[FAIL] Download failed")
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GCP Migration Test")
    parser.add_argument("--setup", action="store_true", help="Create Artifact Registry")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    parser.add_argument("--upload", action="store_true", help="Upload test batch")
    parser.add_argument("--launch", action="store_true", help="Launch test VM")
    parser.add_argument("--status", action="store_true", help="Check migration status")
    parser.add_argument("--download", action="store_true", help="Download result")
    parser.add_argument("--full-test", action="store_true", help="Run full test workflow")
    args = parser.parse_args()
    
    if args.setup:
        create_artifact_registry()
    elif args.build:
        create_artifact_registry()
        build_docker_image()
    elif args.upload:
        upload_test_batch()
    elif args.launch:
        launch_test_vm()
    elif args.status:
        check_status()
    elif args.download:
        download_result()
    elif args.full_test:
        print("=" * 60)
        print("FULL GCP MIGRATION TEST")
        print("=" * 60)
        
        steps = [
            ("Create Artifact Registry", create_artifact_registry),
            ("Build Docker Image", build_docker_image),
            ("Upload Test Batch", upload_test_batch),
            ("Launch Test VM", launch_test_vm),
        ]
        
        for name, func in steps:
            print(f"\n>>> {name}")
            if not func():
                print(f"\n[FAIL] Failed at: {name}")
                return
        
        print("\n" + "=" * 60)
        print("Test VM launched! It will:")
        print("  1. Download the test batch from GCS")
        print("  2. Run migration in Docker container")
        print("  3. Upload result to GCS")
        print("  4. Self-terminate")
        print("\nEstimated time: 5-10 minutes")
        print(f"\nCheck status: python run_gcp_test.py --status")
        print(f"Download result: python run_gcp_test.py --download")
        print("=" * 60)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
