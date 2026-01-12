#!/bin/bash
set -e

echo "=== Chess Data Migration - ALL BATCHES ==="
echo "Started at: $(date)"

# Install Docker
curl -fsSL https://get.docker.com | sh

# Authenticate with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Download all input batches
mkdir -p /data/input /data/output
echo "Downloading all batches from GCS..."
gsutil -m cp gs://chess-migration-gen-lang-c/input/batch_mpv_*.zst /data/input/

# Pull Docker image
echo "Pulling Docker image..."
docker pull us-central1-docker.pkg.dev/gen-lang-client-0213220308/chess-images/chess-migration:latest

# Process each batch
TOTAL=$(ls /data/input/batch_mpv_*.zst | wc -l)
COUNT=0

for batch in /data/input/batch_mpv_*.zst; do
    name=$(basename $batch)
    COUNT=$((COUNT + 1))
    
    # Check if already processed
    if gsutil -q stat gs://chess-migration-gen-lang-c/output/$name 2>/dev/null; then
        echo "[$COUNT/$TOTAL] Skipping $name (already migrated)"
        continue
    fi
    
    echo "[$COUNT/$TOTAL] Processing: $name"
    START=$(date +%s)
    
    docker run --rm \
        -v /data/input:/input \
        -v /data/output:/output \
        us-central1-docker.pkg.dev/gen-lang-client-0213220308/chess-images/chess-migration:latest \
        python migrate_mpv_moves.py --batch /input/$name --output /output/$name
    
    # Upload result immediately
    gsutil cp /data/output/$name gs://chess-migration-gen-lang-c/output/
    
    # Mark as done
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "DONE in ${ELAPSED}s" | gsutil cp - gs://chess-migration-gen-lang-c/status/${name}.done
    
    echo "[$COUNT/$TOTAL] $name completed in ${ELAPSED}s"
done

# Final status
echo "ALL_COMPLETE at $(date)" | gsutil cp - gs://chess-migration-gen-lang-c/status/ALL_DONE.txt
echo "=== All batches processed! ==="

# Self-terminate to save costs
echo "Shutting down..."
shutdown -h now
