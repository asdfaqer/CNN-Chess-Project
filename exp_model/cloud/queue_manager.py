"""
Queue Manager for GCP Data Generation.
Monitor's the 32-vCPU quota and launches next batches as soon as slots open.
Target: 15 batches (ID 27 to 41) of 20,000 games each.
"""
import time
import subprocess
import os

# Configuration
QUOTA_VCPUS = 32
CPU_PER_VM = 8
TARGET_BATCHES = range(27, 42)
PROJECT_ID = "gen-lang-client-0213220308"
BUCKET_NAME = "chess-migration-gen-lang-c"

GCLOUD = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
GSUTIL = r"C:\Users\ccbdc\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd"

def run(cmd):
    result = subprocess.run(f'cmd /c "{cmd}"', shell=True, capture_output=True, text=True)
    return result.stdout

def get_running_vms():
    out = run(f'"{GCLOUD}" compute instances list --filter="name~gen-batch" --format="value(name)"')
    return [line.strip() for line in out.splitlines() if line.strip()]

def get_completed_batches():
    out = run(f'"{GSUTIL}" ls gs://{BUCKET_NAME}/status/*.gen.done')
    completed = []
    for line in out.splitlines():
        if '.gen.done' in line:
            try:
                batch_id = int(line.split('batch_mpv_')[1].split('.zst')[0])
                completed.append(batch_id)
            except:
                pass
    return completed

def main():
    print("=== GCP Queue Manager Starting ===")
    
    while True:
        running = get_running_vms()
        completed = get_completed_batches()
        
        # Determine what's left to do
        in_progress_ids = []
        for vm in running:
            try:
                in_progress_ids.append(int(vm.split('gen-batch-')[1]))
            except:
                pass
        
        to_launch = []
        for b_id in TARGET_BATCHES:
            if b_id not in completed and b_id not in in_progress_ids:
                to_launch.append(b_id)
        
        print(f"\nStatus at {time.ctime()}:")
        print(f"  Running: {len(running)} VMs ({len(running)*CPU_PER_VM}/{QUOTA_VCPUS} vCPUs)")
        print(f"  Completed: {len(completed)} / {len(TARGET_BATCHES)} target batches")
        print(f"  Remaining to launch: {len(to_launch)}")
        
        if not to_launch and not running:
            print("All 15 batches are completed! Task finished.")
            break
            
        # Slots available?
        current_vcpus = len(running) * CPU_PER_VM
        available_vcpus = QUOTA_VCPUS - current_vcpus
        slots = available_vcpus // CPU_PER_VM
        
        if slots > 0 and to_launch:
            num_to_start = min(slots, len(to_launch))
            print(f"[*] Found {slots} free slots. Launching {num_to_start} new batches...")
            
            for i in range(num_to_start):
                next_id = to_launch[i]
                # We use the existing generate_cloud.py to launch specifically
                cmd = f'python cloud/generate_cloud.py --launch 1 --games 20000 --machine medium --batch-id {next_id}'
                print(f"  Executing: {cmd}")
                subprocess.run(f'cmd /c "{cmd}"', shell=True)
        
        # Wait before next poll
        print("Sleeping 5 minutes...")
        time.sleep(300)

if __name__ == "__main__":
    main()
