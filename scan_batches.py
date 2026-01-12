import os
import json
from data_utils import load_batch
from config import DATA_CACHE_DIR

def scan_batches():
    print("=== Scanning batches for Verbose Data Structure (Run 3) ===")
    
    all_files = sorted([f for f in os.listdir(DATA_CACHE_DIR) 
                if f.startswith("batch_mpv_") and f.endswith(".zst")])
    
    results = {
        "verbose_run3": [],
        "legacy_partial": [],
        "broken_or_basic": []
    }
    
    required_run3_keys = {'mpv_cp', 'mpv_mate', 'mpv_depth', 'mpv_moves'}
    
    for filename in all_files:
        path = os.path.join(DATA_CACHE_DIR, filename)
        try:
            d = load_batch(path)
            keys = set(d.keys())
            
            if required_run3_keys.issubset(keys):
                results["verbose_run3"].append(filename)
            elif "mpv_scores" in keys:
                results["legacy_partial"].append(filename)
            else:
                results["broken_or_basic"].append(filename)
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            results["broken_or_basic"].append(filename)

    # Save to JSON
    output_file = "batch_status.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nScan complete!")
    print(f"  Run 3 Verbose: {len(results['verbose_run3'])}")
    print(f"  Legacy Partial: {len(results['legacy_partial'])}")
    print(f"  Basic/Broken: {len(results['broken_or_basic'])}")
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    scan_batches()
