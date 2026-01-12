import os
import argparse
import time
import torch
import config
from generate_data import generate_data_batch_parallel
from data_utils import compress_data_mpv, save_batch
from utils import get_dynamic_resource_info

def main():
    parser = argparse.ArgumentParser(description="Cloud Data Generation Worker")
    parser.add_argument("--count", type=int, default=config.DATA_BATCH_SIZE, help="Number of games to generate")
    parser.add_argument("--batch-name", type=str, required=True, help="Filename for the generated batch")
    parser.add_argument("--output-dir", type=str, default="/output", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Use 90% of CPUs on cloud VMs for maximum throughput
    res = get_dynamic_resource_info()
    num_workers = max(1, int(res["cpu_count"] * 0.9))
    
    # Temporarily override config size
    original_size = config.DATA_BATCH_SIZE
    config.DATA_BATCH_SIZE = args.count
    
    print(f"[*] Starting cloud generation: {args.count} games on {num_workers} workers")
    raw_data = generate_data_batch_parallel(0, num_workers=num_workers)
    
    if raw_data:
        print(f"[*] Compressing {len(raw_data)} games...")
        packed_data = compress_data_mpv(raw_data)
        final_path = os.path.join(args.output_dir, args.batch_name)
        save_batch(packed_data, final_path)
        print(f"[*] Batch saved to {final_path}")
    else:
        print("[!] No games generated.")
        exit(1)

    # Restore config just in case
    config.DATA_BATCH_SIZE = original_size

if __name__ == "__main__":
    main()
