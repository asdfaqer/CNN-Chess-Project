# Cloud Migration Implementation Plan

## Overview
Run the verbose data migration (`migrate_mpv_moves.py`) on Google Cloud Platform using Spot VMs to process 26 batch files (~920 MB total) with Stockfish analysis.

---

## Data Summary
| Metric | Value |
|--------|-------|
| Total Batches | 26 files (batch_mpv_1.zst to batch_mpv_26.zst) |
| Total Data Size | ~920 MB compressed |
| Avg Batch Size | ~37 MB each |
| Positions per Batch | ~50,000 (estimated from 20,000 games × 2.5 positions avg) |
| Analysis Time per Position | 20ms |

---

## Phase 1: Local Setup & Testing ✅ (Partially Complete)

### 1.1 Prerequisites
- [x] Docker installed
- [x] Google Cloud CLI installed
- [ ] Authenticate with GCP: `gcloud auth login`
- [ ] Set project: `gcloud config set project YOUR_PROJECT_ID`
- [ ] Enable required APIs:
  ```bash
  gcloud services enable compute.googleapis.com
  gcloud services enable containerregistry.googleapis.com
  gcloud services enable storage.googleapis.com
  ```

### 1.2 Test Docker Locally
```powershell
cd d:\Chess CNN\exp_model\cloud
docker build -t chess-migration-test .
docker run --rm -v "d:/Chess CNN/exp_model/generated_data:/data" chess-migration-test python migrate_mpv_moves.py --test
```

---

## Phase 2: GCP Infrastructure Setup

### 2.1 Environment Variables
Set these before running commands:
```powershell
$env:GCP_PROJECT = "your-project-id"
$env:GCP_REGION = "us-central1"
$env:GCS_BUCKET = "chess-data-migration"
```

### 2.2 Create GCS Bucket
```bash
gsutil mb -l us-central1 gs://chess-data-migration
```

### 2.3 Build & Push Docker Image
```powershell
cd d:\Chess CNN\exp_model\cloud

# Configure Docker for GCR
gcloud auth configure-docker

# Build image
docker build -t gcr.io/$env:GCP_PROJECT/chess-migration:latest .

# Push to Google Container Registry
docker push gcr.io/$env:GCP_PROJECT/chess-migration:latest
```

---

## Phase 3: Upload Data to GCS

### 3.1 Upload All Batches
```powershell
cd d:\Chess CNN\exp_model\generated_data
gsutil -m cp batch_mpv_*.zst gs://chess-data-migration/input/
```

**Estimated upload time:** ~5-10 minutes (depends on upload speed)

---

## Phase 4: Choose VM Strategy

### Option A: Single Large VM (Recommended for simplicity)
- **Machine Type:** `c3-highcpu-88` (88 vCPUs)
- **Spot Price:** ~$0.50/hr
- **Estimated Time:** ~2-3 hours for all 26 batches
- **Estimated Cost:** ~$1.50

### Option B: Multiple Small VMs (Parallel processing)
- **Machine Type:** `c3-highcpu-22` per batch (22 vCPUs each)
- **Spot Price:** ~$0.12/hr each
- **VMs Needed:** 5-10 in parallel
- **Estimated Time:** ~30 minutes
- **Estimated Cost:** ~$0.50-$1.00

### Option C: Cloud Run Jobs (Serverless - Most cost-effective)
- **vCPUs:** 8 per job
- **Price:** ~$0.005/min per job
- **Parallel Jobs:** 26 (one per batch)
- **Estimated Time:** ~20 minutes
- **Estimated Cost:** ~$2.60

**Recommended:** Option A (Single Large Spot VM) for simplicity and reliability.

---

## Phase 5: Run Migration

### 5.1 Launch Single Large Spot VM
```powershell
python launch_gcp.py --migrate-all --machine large
```

Or manually with the orchestrator:
```powershell
# Upload data first
cd d:\Chess CNN\exp_model\generated_data
gsutil -m cp batch_mpv_*.zst gs://chess-data-migration/input/

# Launch VM with all-batches script (see Section 6)
```

### 5.2 Monitor Progress
```bash
# Check VM status
gcloud compute instances list

# View VM logs
gcloud compute ssh migrate-all-batches --zone=us-central1-a --command="tail -f /var/log/syslog"

# Check completion status
gsutil ls gs://chess-data-migration/status/
```

---

## Phase 6: Updated VM Startup Script (Multi-Batch)

Create a new script for processing all batches sequentially:

```bash
#!/bin/bash
set -e

echo "=== Chess Data Migration - All Batches ==="

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with GCR
gcloud auth configure-docker --quiet

# Download all input batches
mkdir -p /data/input /data/output
gsutil -m cp gs://chess-data-migration/input/* /data/input/

# Pull Docker image
docker pull gcr.io/PROJECT_ID/chess-migration:latest

# Process each batch
for batch in /data/input/batch_mpv_*.zst; do
    name=$(basename $batch)
    echo "Processing: $name"
    
    docker run --rm \
        -v /data/input:/input \
        -v /data/output:/output \
        gcr.io/PROJECT_ID/chess-migration:latest \
        python migrate_mpv_moves.py --batch /input/$name --output /output/$name
    
    # Upload result immediately
    gsutil cp /data/output/$name gs://chess-data-migration/output/
    
    # Mark as done
    echo "DONE" | gsutil cp - gs://chess-data-migration/status/${name}.done
done

# Final status
echo "ALL_COMPLETE" | gsutil cp - gs://chess-data-migration/status/ALL_DONE.txt

# Self-terminate
shutdown -h now
```

---

## Phase 7: Download Results

### 7.1 Check Completion
```bash
gsutil ls gs://chess-data-migration/status/ALL_DONE.txt
```

### 7.2 Download Migrated Batches
```powershell
cd d:\Chess CNN\exp_model\generated_data

# Backup original data first
mkdir -p backup
move batch_mpv_*.zst backup/

# Download migrated data
gsutil -m cp gs://chess-data-migration/output/* .
```

### 7.3 Verify Migration
```python
# Quick verification script
import torch
import zstandard as zstd
import io

def verify_batch(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        raw = dctx.decompress(f.read())
    data = torch.load(io.BytesIO(raw), weights_only=False)
    required_keys = ['mpv_cp', 'mpv_mate', 'mpv_depth', 'mpv_moves']
    return all(k in data for k in required_keys)

# Run on all migrated batches
```

---

## Phase 8: Cleanup

### 8.1 Delete Cloud Resources
```bash
# Delete VM (if not auto-terminated)
gcloud compute instances delete migrate-all-batches --zone=us-central1-a

# Optional: Delete bucket after confirming local backups
gsutil rm -r gs://chess-data-migration
```

### 8.2 Remove Local Backups (after verification)
```powershell
Remove-Item -Recurse d:\Chess CNN\exp_model\generated_data\backup
```

---

## Cost Estimation Summary

| Resource | Estimate |
|----------|----------|
| Spot VM (c3-highcpu-88 × 3 hours) | ~$1.50 |
| Cloud Storage (1 GB × 1 day) | ~$0.02 |
| Network Egress (920 MB download) | ~$0.10 |
| **Total** | **~$1.62** |

---

## Quick Start Commands

```powershell
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth configure-docker

# 2. Build and push Docker image
cd d:\Chess CNN\exp_model\cloud
docker build -t gcr.io/YOUR_PROJECT_ID/chess-migration:latest .
docker push gcr.io/YOUR_PROJECT_ID/chess-migration:latest

# 3. Upload data
cd d:\Chess CNN\exp_model\generated_data
gsutil -m cp batch_mpv_*.zst gs://chess-data-migration/input/

# 4. Run migration (use updated launch_gcp.py)
cd d:\Chess CNN\exp_model\cloud
python launch_gcp.py --migrate-all --machine large

# 5. Wait and download
gsutil -m cp gs://chess-data-migration/output/* d:\Chess CNN\exp_model\generated_data\
```

---

## Troubleshooting

### VM Preempted (Spot Instance Terminated)
- Re-run the launch command - completed batches are preserved in GCS
- The script checks for existing output files before re-processing

### Out of Quota
- Request quota increase: Compute Engine → Quotas → C3 CPUs
- Or use smaller machine type (c3-highcpu-44)

### Docker Build Fails
- Ensure Stockfish download URL is correct
- Check Docker is running locally

### Migration Takes Too Long
- Increase `ANALYSIS_TIME` in migrate_mpv_moves.py for accuracy
- Or decrease for speed (minimum 10ms recommended)
