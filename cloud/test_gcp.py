"""
Simple GCP Test Script - Uses cmd.exe to avoid PowerShell execution policy issues.
Tests the full GCP workflow with a small sample.
"""
import os
import subprocess
import sys

# Configuration from .gcp_config.json
PROJECT_ID = "gen-lang-client-0213220308"
REGION = "us-central1"
ZONE = f"{REGION}-a"
BUCKET_NAME = "chess-migration-gen-lang-c"

# Paths
GCLOUD = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
GSUTIL = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd"

def run(cmd):
    """Run command via cmd.exe and return output."""
    full_cmd = f'cmd /c "{cmd}"'
    print(f"$ {cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0, result.stdout

def test_gcloud():
    """Test that gcloud works and is authenticated."""
    print("\n=== Step 1: Testing gcloud ===")
    ok, out = run(f'"{GCLOUD}" config get-value project')
    if ok and PROJECT_ID in out:
        print(f"[OK] gcloud works, project: {PROJECT_ID}")
        return True
    else:
        print("[FAIL] gcloud not working properly")
        return False

def test_bucket():
    """Create or verify bucket exists."""
    print("\n=== Step 2: Testing GCS Bucket ===")
    ok, out = run(f'"{GSUTIL}" ls -b gs://{BUCKET_NAME}')
    if ok:
        print(f"[OK] Bucket gs://{BUCKET_NAME} exists")
        return True
    else:
        print(f"Creating bucket gs://{BUCKET_NAME}...")
        ok, _ = run(f'"{GSUTIL}" mb -l {REGION} gs://{BUCKET_NAME}')
        if ok:
            print(f"[OK] Bucket created")
            return True
        else:
            print("[FAIL] Failed to create bucket")
            return False

def upload_test_file():
    """Upload a small test file to GCS."""
    print("\n=== Step 3: Uploading Test File ===")
    
    # Create a simple test file
    test_path = "generated_data/gcp_test.txt"
    with open(test_path, "w") as f:
        f.write("GCP upload test successful!")
    
    ok, _ = run(f'"{GSUTIL}" cp "{test_path}" gs://{BUCKET_NAME}/test/')
    if ok:
        print(f"[OK] Upload successful")
        os.remove(test_path)
        return True
    else:
        print("[FAIL] Upload failed")
        return False

def download_test_file():
    """Download the test file back."""
    print("\n=== Step 4: Downloading Test File ===")
    
    download_path = "generated_data/gcp_test_download.txt"
    ok, _ = run(f'"{GSUTIL}" cp gs://{BUCKET_NAME}/test/gcp_test.txt "{download_path}"')
    
    if ok and os.path.exists(download_path):
        with open(download_path) as f:
            content = f.read()
        os.remove(download_path)
        if "successful" in content:
            print(f"[OK] Download successful, content verified")
            return True
    
    print("[FAIL] Download failed")
    return False

def cleanup():
    """Clean up test files."""
    print("\n=== Cleanup ===")
    run(f'"{GSUTIL}" rm gs://{BUCKET_NAME}/test/gcp_test.txt')
    print("[OK] Cleanup done")

def main():
    print("=" * 50)
    print("GCP Connection Test")
    print("=" * 50)
    
    steps = [
        ("gcloud", test_gcloud),
        ("bucket", test_bucket),
        ("upload", upload_test_file),
        ("download", download_test_file),
    ]
    
    for name, func in steps:
        if not func():
            print(f"\n[FAIL] Test failed at step: {name}")
            return False
    
    cleanup()
    
    print("\n" + "=" * 50)
    print("[OK] ALL TESTS PASSED - GCP is working!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Upload test batch: python test_gcp.py --upload-batch")
    print("  2. Build Docker image: python test_gcp.py --build-image")
    print("  3. Run migration: python test_gcp.py --run-migration")
    return True

if __name__ == "__main__":
    main()
