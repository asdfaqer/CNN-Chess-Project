"""
GCP Cloud Migration Orchestrator
Handles uploading data, launching Spot VMs, and downloading results.
"""
import os
import subprocess
import argparse
import time
import json

# Configuration - loaded from environment, config file, or user input
CONFIG_FILE = os.path.join(os.path.dirname(__file__), ".gcp_config.json")

def load_config():
    """Load configuration from file, environment, or prompt user."""
    config = {
        "project_id": None,
        "region": "us-central1",
        "bucket_name": None
    }
    
    # Try loading from config file first
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
                config.update(saved)
        except:
            pass
    
    # Override with environment variables if set
    if os.environ.get("GCP_PROJECT"):
        config["project_id"] = os.environ.get("GCP_PROJECT")
    if os.environ.get("GCP_REGION"):
        config["region"] = os.environ.get("GCP_REGION")
    if os.environ.get("GCS_BUCKET"):
        config["bucket_name"] = os.environ.get("GCS_BUCKET")
    
    return config

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {CONFIG_FILE}")

def ensure_config(project_id=None, bucket_name=None):
    """Ensure all required config values are set, prompting if needed."""
    global PROJECT_ID, REGION, ZONE, BUCKET_NAME, IMAGE_NAME
    
    config = load_config()
    changed = False
    
    # Use provided values or prompt for missing ones
    if project_id:
        config["project_id"] = project_id
        changed = True
    elif not config["project_id"] or config["project_id"] == "your-project-id":
        print("\n=== GCP Configuration ===")
        print("To find your project ID, run: gcloud projects list")
        config["project_id"] = input("Enter your GCP Project ID: ").strip()
        changed = True
    
    if bucket_name:
        config["bucket_name"] = bucket_name
        changed = True
    elif not config["bucket_name"]:
        default_bucket = f"chess-migration-{config['project_id'][:10]}"
        bucket_input = input(f"Enter GCS bucket name [{default_bucket}]: ").strip()
        config["bucket_name"] = bucket_input if bucket_input else default_bucket
        changed = True
    
    if changed:
        save_config(config)
    
    # Set global variables
    PROJECT_ID = config["project_id"]
    REGION = config["region"]
    ZONE = f"{REGION}-a"
    BUCKET_NAME = config["bucket_name"]
    IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/chess-images/chess-migration:latest"
    
    print(f"\nUsing configuration:")
    print(f"  Project:  {PROJECT_ID}")
    print(f"  Region:   {REGION}")
    print(f"  Bucket:   {BUCKET_NAME}")
    print()

# Initialize with defaults (will be updated by ensure_config)
PROJECT_ID = "your-project-id"
REGION = "us-central1"
ZONE = f"{REGION}-a"
BUCKET_NAME = "chess-data-migration"
IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/chess-images/chess-migration:latest"

# Machine types for different workloads
MACHINE_TYPES = {
    "test": "e2-standard-4",      # 4 vCPUs, cheap for testing
    "small": "c3-highcpu-22",     # 22 vCPUs
    "medium": "c3-highcpu-44",    # 44 vCPUs
    "large": "c3-highcpu-88",     # 88 vCPUs
    "xlarge": "c3-highcpu-176",   # 176 vCPUs (max)
}

def run_cmd(cmd, check=True):
    """Run a shell command and return output."""
    import platform
    if platform.system() == "Windows":
        cmd = cmd.replace("gcloud ", "gcloud.cmd ")
        cmd = cmd.replace("gsutil ", "gsutil.cmd ")
    
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise Exception(f"Command failed: {cmd}")
    return result.stdout.strip()

def check_gcloud(project_id=None, bucket_name=None):
    """Verify gcloud is installed and authenticated, then ensure config is set."""
    try:
        run_cmd("gcloud --version", check=False)
        project = run_cmd("gcloud config get-value project")
        print(f"GCloud project from CLI: {project}")
        
        # Ensure our config is loaded/prompted
        ensure_config(project_id, bucket_name)
        return True
    except Exception as e:
        print(f"ERROR: gcloud CLI issue: {e}")
        print("Install gcloud from: https://cloud.google.com/sdk/docs/install")
        return False

def create_bucket():
    """Create GCS bucket if it doesn't exist."""
    result = subprocess.run(
        f"gsutil ls -b gs://{BUCKET_NAME}", 
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0 and BUCKET_NAME in result.stdout:
        print(f"Bucket gs://{BUCKET_NAME} already exists")
    else:
        print(f"Creating bucket gs://{BUCKET_NAME}...")
        run_cmd(f"gsutil mb -l {REGION} gs://{BUCKET_NAME}")
        print(f"Bucket created successfully!")

def upload_batch(local_path: str, remote_name: str = None):
    """Upload a batch file to GCS."""
    if remote_name is None:
        remote_name = os.path.basename(local_path)
    remote_path = f"gs://{BUCKET_NAME}/input/{remote_name}"
    print(f"Uploading {local_path} to {remote_path}...")
    run_cmd(f'gsutil cp "{local_path}" {remote_path}')
    return remote_path

def download_results(remote_name: str, local_dir: str):
    """Download migrated batch from GCS."""
    remote_path = f"gs://{BUCKET_NAME}/output/{remote_name}"
    local_path = os.path.join(local_dir, remote_name)
    print(f"Downloading {remote_path} to {local_path}...")
    run_cmd(f'gsutil cp {remote_path} "{local_path}"')
    return local_path

def build_and_push_image():
    """Build Docker image using Cloud Build and push to GCR."""
    import shutil
    
    # Get paths
    cloud_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cloud_dir)
    
    # Files needed for Docker build (relative to parent)
    required_files = ['config.py', 'utils.py', 'data_utils.py', 'migrate_mpv_moves.py', 'generate_data.py', 'cloud_generate.py']
    
    # Copy files to cloud directory for Docker build
    print("Copying source files for Cloud Build...")
    for f in required_files:
        src = os.path.join(parent_dir, f)
        dst = os.path.join(cloud_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {f}")
        else:
            print(f"  WARNING: {f} not found!")
    
    try:
        print("Building Docker image using Google Cloud Build...")
        print("(This builds in the cloud, no local Docker required)")
        # Run from the cloud directory where the Dockerfile is
        current_dir = os.getcwd()
        os.chdir(cloud_dir)
        try:
            run_cmd(f"gcloud builds submit --tag {IMAGE_NAME} .")
        finally:
            os.chdir(current_dir)
            
        print("Image built and pushed successfully!")
    finally:
        # Clean up copied files
        print("Cleaning up temporary files...")
        for f in required_files:
            dst = os.path.join(cloud_dir, f)
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except:
                    pass

def create_generation_startup_script(batch_name: str, games_count: int) -> str:
    """Generate startup script for a data generation VM."""
    return f'''#!/bin/bash
set -e

echo "=== Chess Data Generation VM ==="
echo "Batch: {batch_name}"
echo "Games: {games_count}"

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with GCR
gcloud auth configure-docker --quiet

# Run generation
mkdir -p /data/output
docker run --rm \\
    -v /data/output:/output \\
    {IMAGE_NAME} \\
    python cloud_generate.py --count {games_count} --batch-name {batch_name} --output-dir /output

# Upload result
gsutil cp /data/output/{batch_name} gs://{BUCKET_NAME}/input/

# Signal completion
echo "GEN_COMPLETE" > /tmp/done.txt
gsutil cp /tmp/done.txt gs://{BUCKET_NAME}/status/{batch_name}.gen.done

# Self-terminate
echo "Shutting down..."
for i in {{1..10}}; do
    shutdown -h now && break
    sleep 5
done
'''

def create_startup_script(batch_name: str, num_workers: int = None) -> str:
    """Generate startup script for a single batch."""
    if num_workers is None:
        num_workers = "$(nproc)"
    
    return f'''#!/bin/bash
set -e

echo "=== Chess Data Migration VM ==="
echo "Batch: {batch_name}"
echo "Workers: {num_workers}"

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with GCR
gcloud auth configure-docker --quiet

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

# Self-terminate to save costs
echo "Shutting down..."
shutdown -h now
'''

def create_startup_script_all_batches() -> str:
    """Generate startup script for processing ALL batches on a single VM."""
    return f'''#!/bin/bash
set -e

echo "=== Chess Data Migration - ALL BATCHES ==="
echo "Started at: $(date)"

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with GCR
gcloud auth configure-docker --quiet

# Download all input batches
mkdir -p /data/input /data/output
echo "Downloading all batches from GCS..."
gsutil -m cp gs://{BUCKET_NAME}/input/batch_mpv_*.zst /data/input/

# Pull Docker image
echo "Pulling Docker image..."
docker pull {IMAGE_NAME}

# Process each batch
TOTAL=$(ls /data/input/batch_mpv_*.zst | wc -l)
COUNT=0

for batch in /data/input/batch_mpv_*.zst; do
    name=$(basename $batch)
    COUNT=$((COUNT + 1))
    
    # Check if already processed
    if gsutil -q stat gs://{BUCKET_NAME}/output/$name 2>/dev/null; then
        echo "[$COUNT/$TOTAL] Skipping $name (already migrated)"
        continue
    fi
    
    echo "[$COUNT/$TOTAL] Processing: $name"
    START=$(date +%s)
    
    docker run --rm \\
        -v /data/input:/input \\
        -v /data/output:/output \\
        {IMAGE_NAME} \\
        python migrate_mpv_moves.py --batch /input/$name --output /output/$name
    
    # Upload result immediately
    gsutil cp /data/output/$name gs://{BUCKET_NAME}/output/
    
    # Mark as done
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "DONE in ${{ELAPSED}}s" | gsutil cp - gs://{BUCKET_NAME}/status/${{name}}.done
    
    echo "[$COUNT/$TOTAL] $name completed in ${{ELAPSED}}s"
done

# Final status
echo "ALL_COMPLETE at $(date)" | gsutil cp - gs://{BUCKET_NAME}/status/ALL_DONE.txt
echo "=== All batches processed! ==="

# Self-terminate to save costs
echo "Shutting down..."
shutdown -h now
'''

def launch_spot_vm(batch_name: str, machine_type: str = "test"):
    """Launch a Spot VM to process a batch."""
    vm_name = f"migrate-{batch_name.replace('.', '-').replace('_', '-')[:20]}"
    
    startup_script = create_startup_script(batch_name)
    import tempfile
    script_path = os.path.join(tempfile.gettempdir(), f"{vm_name}-startup.sh")
    with open(script_path, "w") as f:
        f.write(startup_script)
    
    machine = MACHINE_TYPES.get(machine_type, machine_type)
    
    print(f"Launching Spot VM: {vm_name} ({machine})...")
    cmd = f'''gcloud compute instances create {vm_name} \\
        --zone={ZONE} \\
        --machine-type={machine} \\
        --provisioning-model=SPOT \\
        --instance-termination-action=DELETE \\
        --scopes=cloud-platform \\
        --metadata-from-file=startup-script={script_path} \\
        --boot-disk-size=50GB \\
        --image-family=debian-12 \\
        --image-project=debian-cloud'''
    
    run_cmd(cmd)
    print(f"VM {vm_name} launched successfully!")
    return vm_name

def launch_spot_vm_all(machine_type: str = "large"):
    """Launch a Spot VM to process ALL batches sequentially."""
    vm_name = "migrate-all-batches"
    
    startup_script = create_startup_script_all_batches()
    import tempfile
    script_path = os.path.join(tempfile.gettempdir(), "migrate-all-startup.sh")
    with open(script_path, "w") as f:
        f.write(startup_script)
    
    machine = MACHINE_TYPES.get(machine_type, machine_type)
    
    print(f"Launching Spot VM for ALL batches: {vm_name} ({machine})...")
    cmd = f'''gcloud compute instances create {vm_name} \\
        --zone={ZONE} \\
        --machine-type={machine} \\
        --provisioning-model=SPOT \\
        --instance-termination-action=DELETE \\
        --scopes=cloud-platform \\
        --metadata-from-file=startup-script={script_path} \\
        --boot-disk-size=100GB \\
        --image-family=debian-12 \\
        --image-project=debian-cloud'''
    
    run_cmd(cmd)
    print(f"VM {vm_name} launched successfully!")
    print(f"\nMonitor progress with:")
    print(f"  gsutil ls gs://{BUCKET_NAME}/status/")
    print(f"\nOr SSH into the VM:")
    print(f"  gcloud compute ssh {vm_name} --zone={ZONE}")
    return vm_name

def wait_for_all_completion(timeout_minutes: int = 180):
    """Wait for all migrations to complete."""
    status_path = f"gs://{BUCKET_NAME}/status/ALL_DONE.txt"
    start = time.time()
    
    print(f"Waiting for all migrations to complete (timeout: {timeout_minutes}min)...")
    while time.time() - start < timeout_minutes * 60:
        try:
            result = run_cmd(f"gsutil cat {status_path}", check=False)
            if "ALL_COMPLETE" in result:
                print("All migrations complete!")
                return True
        except:
            pass
        
        # Show progress
        try:
            done_list = run_cmd(f"gsutil ls gs://{BUCKET_NAME}/status/*.done", check=False)
            done_count = len([l for l in done_list.split('\n') if l.strip()])
            print(f"  Progress: {done_count}/26 batches complete...")
        except:
            pass
        
        time.sleep(60)
        elapsed = int((time.time() - start) / 60)
        print(f"  Running for {elapsed}min...")
    
    print("Timeout reached!")
    return False

def wait_for_completion(batch_name: str, timeout_minutes: int = 60):
    """Wait for migration to complete."""
    status_path = f"gs://{BUCKET_NAME}/status/{batch_name}.done"
    start = time.time()
    
    print(f"Waiting for migration to complete (timeout: {timeout_minutes}min)...")
    while time.time() - start < timeout_minutes * 60:
        try:
            run_cmd(f"gsutil ls {status_path}", check=False)
            print("Migration complete!")
            return True
        except:
            pass
        time.sleep(30)
        elapsed = int((time.time() - start) / 60)
        print(f"  Still running... ({elapsed}min)")
    
    print("Timeout reached!")
    return False

def upload_all_batches(data_dir: str):
    """Upload all batch files to GCS."""
    import glob
    batch_files = glob.glob(os.path.join(data_dir, "batch_mpv_*.zst"))
    print(f"Found {len(batch_files)} batch files to upload")
    
    # Use gsutil -m for parallel uploads
    run_cmd(f'gsutil -m cp "{data_dir}/batch_mpv_*.zst" gs://{BUCKET_NAME}/input/')
    print(f"Uploaded {len(batch_files)} files to gs://{BUCKET_NAME}/input/")

def download_all_results(output_dir: str):
    """Download all migrated batches from GCS."""
    print(f"Downloading all results to {output_dir}...")
    run_cmd(f'gsutil -m cp "gs://{BUCKET_NAME}/output/*" "{output_dir}"')
    print("Download complete!")

def test_local_docker():
    """Test the Docker container locally before deploying to GCP."""
    print("\n=== Testing Docker Locally ===")
    
    # Build image
    print("Building Docker image...")
    run_cmd("docker build -t chess-migration-test .")
    
    # Run with test data
    print("Running migration on test data...")
    run_cmd('''docker run --rm ^
        -v "%cd%\\..\\generated_data:/data" ^
        chess-migration-test ^
        python migrate_mpv_moves.py --test''')
    
    print("Local Docker test passed!")

def main():
    parser = argparse.ArgumentParser(description="GCP Cloud Migration Orchestrator")
    parser.add_argument("--test-local", action="store_true", help="Test Docker locally")
    parser.add_argument("--test-gcp", type=str, help="Test GCP with a specific batch file")
    parser.add_argument("--migrate", type=str, help="Migrate a single batch file")
    parser.add_argument("--migrate-all", action="store_true", help="Migrate ALL batch files on a single VM")
    parser.add_argument("--upload-all", type=str, help="Upload all batches from a directory")
    parser.add_argument("--download-all", type=str, help="Download all results to a directory")
    parser.add_argument("--generate", type=int, help="Number of VMs to launch for generation")
    parser.add_argument("--games-per-vm", type=int, default=20000, help="Games per VM")
    parser.add_argument("--machine", type=str, default="test", choices=list(MACHINE_TYPES.keys()))
    parser.add_argument("--download", type=str, help="Download a single completed batch")
    parser.add_argument("--status", action="store_true", help="Check migration status")
    parser.add_argument("--build-push", action="store_true", help="Build and push Docker image only")
    args = parser.parse_args()
    
    if args.test_local:
        test_local_docker()
    elif args.build_push:
        if not check_gcloud():
            return
        build_and_push_image()
        print("Docker image built and pushed successfully!")
    elif args.generate:
        if not check_gcloud():
            return
        create_bucket()
        build_and_push_image()
        
        # Determine starting batch ID
        print("Checking for existing batches in GCS...")
        try:
            ls_result = run_cmd(f"gsutil ls gs://{BUCKET_NAME}/input/batch_mpv_*.zst", check=False)
            ids = []
            for line in ls_result.split('\n'):
                if 'batch_mpv_' in line:
                    try:
                        part = line.split('batch_mpv_')[1].split('.zst')[0]
                        ids.append(int(part))
                    except: pass
            start_id = max(ids) + 1 if ids else 100 # Start at 100 for cloud batches to avoid local overlap
        except:
            start_id = 100
            
        print(f"Starting generation from Batch ID: {start_id}")
        
        for i in range(args.generate):
            batch_id = start_id + i
            batch_name = f"batch_mpv_{batch_id}.zst"
            
            vm_name = f"gen-batch-{batch_id}"
            startup_script = create_generation_startup_script(batch_name, args.games_per_vm)
            import tempfile
            script_path = os.path.join(tempfile.gettempdir(), f"{vm_name}-startup.sh")
            with open(script_path, "w") as f:
                f.write(startup_script)
            
            machine = MACHINE_TYPES.get(args.machine, args.machine)
            print(f" [{i+1}/{args.generate}] Launching {vm_name} ({machine})...")
            
            cmd = f'''gcloud compute instances create {vm_name} \\
                --zone={ZONE} \\
                --machine-type={machine} \\
                --provisioning-model=SPOT \\
                --instance-termination-action=DELETE \\
                --scopes=cloud-platform \\
                --metadata-from-file=startup-script={script_path} \\
                --boot-disk-size=30GB \\
                --image-family=debian-12 \\
                --image-project=debian-cloud \\
                --quiet'''
            run_cmd(cmd)
            
        print(f"\nSuccessfully launched {args.generate} generation VMs!")
        print(f"Monitor: gsutil ls gs://{BUCKET_NAME}/status/*.gen.done")
        print(f"Download: python launch_gcp.py --download-all ../generated_data")
        
    elif args.upload_all:
        if not check_gcloud():
            return
        create_bucket()
        upload_all_batches(args.upload_all)
    elif args.migrate_all:
        if not check_gcloud():
            return
        create_bucket()
        build_and_push_image()
        launch_spot_vm_all(args.machine)
        print("\n=== VM Launched ===")
        print("The VM will process all batches and self-terminate when done.")
        print(f"Estimated time: 2-3 hours with machine type '{args.machine}'")
        print(f"\nMonitor: gsutil ls gs://{BUCKET_NAME}/status/")
        print(f"Download when done: python launch_gcp.py --download-all ../generated_data")
    elif args.test_gcp:
        if not check_gcloud():
            return
        create_bucket()
        upload_batch(args.test_gcp)
        build_and_push_image()
        batch_name = os.path.basename(args.test_gcp)
        launch_spot_vm(batch_name, args.machine)
        if wait_for_completion(batch_name, timeout_minutes=30):
            download_results(batch_name, "../generated_data")
    elif args.migrate:
        if not check_gcloud():
            return
        batch_name = os.path.basename(args.migrate)
        upload_batch(args.migrate)
        launch_spot_vm(batch_name, args.machine)
    elif args.download_all:
        if not check_gcloud():
            return
        download_all_results(args.download_all)
    elif args.download:
        download_results(args.download, "../generated_data")
    elif args.status:
        if not check_gcloud():
            return
        print(f"\n=== Migration Status ===")
        done_count = 0
        try:
            done_list = run_cmd(f"gsutil ls gs://{BUCKET_NAME}/status/*.done", check=False)
            done_files = [l for l in done_list.split('\n') if l.strip() and '.done' in l]
            done_count = len(done_files)
            print(f"Completed: {done_count}/26 batches")
            for f in done_files:
                print(f"  ✓ {os.path.basename(f).replace('.done', '')}")
        except:
            print("No completed batches found (bucket may not exist yet).")
        
        try:
            result = run_cmd(f"gsutil cat gs://{BUCKET_NAME}/status/ALL_DONE.txt", check=False)
            if "ALL_COMPLETE" in result:
                print("\n✓ ALL BATCHES COMPLETE!")
            elif done_count == 0:
                print("\n⋯ Migration not started yet. Run --migrate-all to begin.")
            else:
                print("\n⋯ Migration still in progress...")
        except:
            if done_count == 0:
                print("\n⋯ Migration not started yet. Run --migrate-all to begin.")
            else:
                print("\n⋯ Migration still in progress...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

