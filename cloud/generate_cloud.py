"""
Launch GCP VMs for data generation.
Uses the existing Docker image to generate new chess training data.
"""
import os
import subprocess
import sys
import argparse

# Configuration
PROJECT_ID = "gen-lang-client-0213220308"
REGION = "us-central1"
ZONE = f"{REGION}-a"
BUCKET_NAME = "chess-migration-gen-lang-c"
IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/chess-images/chess-migration:latest"

# Paths
GCLOUD = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
GSUTIL = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd"

MACHINE_TYPES = {
    "test": "n2-standard-2",        # 2 vCPUs, 8GB RAM
    "small": "n2-standard-4",       # 4 vCPUs, 16GB RAM
    "medium": "n2-standard-8",      # 8 vCPUs, 32GB RAM
    "large": "n2-standard-16",      # 16 vCPUs, 64GB RAM
    "xlarge": "n2-standard-32",     # 32 vCPUs, 128GB RAM
}

def run(cmd, check=True):
    """Run command via cmd.exe."""
    full_cmd = f'cmd /c "{cmd}"'
    print(f"$ {cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0, result.stdout

def get_next_batch_id():
    """Find the next available batch ID from GCS."""
    ok, out = run(f'"{GSUTIL}" ls gs://{BUCKET_NAME}/input/batch_mpv_*.zst', check=False)
    if not ok:
        return 100  # Start cloud batches at 100
    
    ids = []
    for line in out.strip().split('\n'):
        if 'batch_mpv_' in line:
            try:
                part = line.split('batch_mpv_')[1].split('.zst')[0]
                ids.append(int(part))
            except:
                pass
    return max(ids) + 1 if ids else 100

def create_generation_startup_script(batch_name: str, games_count: int) -> str:
    """Generate startup script for a data generation VM."""
    return f'''#!/bin/bash
set -ex

echo "=== Chess Data Generation VM ==="
echo "Batch: {batch_name}"
echo "Games: {games_count}"
date

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with Artifact Registry
gcloud auth configure-docker {REGION}-docker.pkg.dev --quiet

# Run generation
mkdir -p /data/output
docker run --rm \\
    -v /data/output:/output \\
    {IMAGE_NAME} \\
    python cloud_generate.py --count {games_count} --batch-name {batch_name} --output-dir /output

# Upload result
gsutil cp /data/output/{batch_name} gs://{BUCKET_NAME}/input/

# Signal completion
echo "GEN_COMPLETE at $(date)" | gsutil cp - gs://{BUCKET_NAME}/status/{batch_name}.gen.done

echo "=== Generation Complete ==="
date

# Self-terminate to save costs
shutdown -h now
'''

def launch_generation_vm(batch_id: int, games_count: int, machine_type: str):
    """Launch a single generation VM."""
    batch_name = f"batch_mpv_{batch_id}.zst"
    vm_name = f"gen-batch-{batch_id}"
    
    script_content = create_generation_startup_script(batch_name, games_count)
    script_path = os.path.join(os.path.dirname(__file__), f"{vm_name}-startup.sh")
    
    with open(script_path, "w", newline='\n') as f:
        f.write(script_content)
    
    machine = MACHINE_TYPES.get(machine_type, machine_type)
    
    print(f"\nLaunching {vm_name} ({machine}) to generate {games_count} games...")
    
    cmd = f'"{GCLOUD}" compute instances create {vm_name} --zone={ZONE} --machine-type={machine} --provisioning-model=SPOT --instance-termination-action=DELETE --scopes=cloud-platform --metadata-from-file=startup-script="{script_path}" --boot-disk-size=30GB --image-family=debian-12 --image-project=debian-cloud --quiet'
    
    ok, _ = run(cmd)
    
    # Cleanup script file
    try:
        os.remove(script_path)
    except:
        pass
    
    if ok:
        print(f"[OK] VM {vm_name} launched!")
        return True
    else:
        print(f"[FAIL] Failed to launch {vm_name}")
        return False

def check_status():
    """Check generation status."""
    print("\n=== Generation Status ===")
    ok, out = run(f'"{GSUTIL}" ls gs://{BUCKET_NAME}/status/*.gen.done', check=False)
    
    if ok and out.strip():
        done_files = [l for l in out.strip().split('\n') if '.gen.done' in l]
        print(f"Completed: {len(done_files)} batches")
        for f in done_files:
            batch = os.path.basename(f).replace('.gen.done', '')
            print(f"  [OK] {batch}")
    else:
        print("No completed generation batches found.")
    
    # Check running VMs
    print("\nRunning VMs:")
    run(f'"{GCLOUD}" compute instances list --filter="name~gen-batch" --format="table(name,zone,status,machineType)"', check=False)

def download_generated():
    """Download newly generated batches."""
    print("\n=== Downloading Generated Batches ===")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_data")
    ok, _ = run(f'"{GSUTIL}" -m cp "gs://{BUCKET_NAME}/input/batch_mpv_*.zst" "{output_dir}/"')
    if ok:
        print(f"[OK] Downloaded to {output_dir}")
    else:
        print("[FAIL] Download failed")

def main():
    parser = argparse.ArgumentParser(description="Launch GCP VMs for data generation")
    parser.add_argument("--launch", type=int, metavar="N", help="Launch N generation VMs")
    parser.add_argument("--games", type=int, default=20000, help="Games per VM (default: 20000)")
    parser.add_argument("--machine", type=str, default="medium", choices=list(MACHINE_TYPES.keys()), help="Machine type")
    parser.add_argument("--status", action="store_true", help="Check generation status")
    parser.add_argument("--download", action="store_true", help="Download generated batches")
    parser.add_argument("--test", action="store_true", help="Launch one small test VM")
    parser.add_argument("--batch-id", type=int, help="Specify starting batch ID (optional)")
    args = parser.parse_args()
    
    if args.test:
        batch_id = args.batch_id if args.batch_id is not None else get_next_batch_id()
        print(f"Launching TEST VM with batch ID {batch_id}")
        launch_generation_vm(batch_id, 1000, "test")
        print(f"\nMonitor: python cloud\\generate_cloud.py --status")
        
    elif args.launch:
        start_id = args.batch_id if args.batch_id is not None else get_next_batch_id()
        print(f"Starting from batch ID: {start_id}")
        print(f"Launching {args.launch} VMs, {args.games} games each, machine: {args.machine}")
        
        for i in range(args.launch):
            batch_id = start_id + i
            launch_generation_vm(batch_id, args.games, args.machine)
        
        print(f"\n=== Launched {args.launch} VMs ===")
        print(f"Monitor: python cloud\\generate_cloud.py --status")
        print(f"Download: python cloud\\generate_cloud.py --download")
        
    elif args.status:
        check_status()
        
    elif args.download:
        download_generated()
        
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python cloud\\generate_cloud.py --test                    # Test with 1 small VM")
        print("  python cloud\\generate_cloud.py --launch 5 --games 20000  # Launch 5 VMs")
        print("  python cloud\\generate_cloud.py --status                  # Check progress")
        print("  python cloud\\generate_cloud.py --download                # Download results")

if __name__ == "__main__":
    main()
