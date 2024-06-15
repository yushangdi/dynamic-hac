#!/bin/bash

# Directory containing subdirectories
main_dir="/home/ubuntu/ILSVRC/Data/CLS-LOC/train/"

# Destination directory for sampled files (make sure it exists or create it)
destination_dir="/home/ubuntu/imagenet/images/"

# Iterate through each subdirectory in the main directory
for dir in "$main_dir"*/; do
    echo "Processing $dir"
    # Ensure it's a directory
    if [ -d "$dir" ]; then
        # Sample 50 files and copy them to the destination directory
        find "$dir" -type f -print0 | shuf -zn50 | xargs -0 -I {} cp {} "$destination_dir"
    fi
done
