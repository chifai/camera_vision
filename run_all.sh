#!/bin/bash

echo "Starting batch processing of all PNG images using sobel_finder.py..."

for f in *.png; do
  # Skip already processed or debug images
  if [[ "$f" == *"_debug"* ]] || [[ "$f" == *"_edges"* ]]; then
    continue
  fi
  
  echo "Processing $f..."
  python sobel_finder.py "$f"
done

echo "Batch processing complete!"
echo "Results can be found in:"
echo "  - Debug Masks: processed/debug_sobel/"
echo "  - Edge Overlays: processed/edges_sobel/"
