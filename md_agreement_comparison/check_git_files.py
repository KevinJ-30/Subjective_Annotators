from pathlib import Path
import os

def get_file_size_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)

# Get all files being tracked by git
tracked_files = os.popen('git ls-files').read().splitlines()

large_files = []
for file in tracked_files:
    size_mb = get_file_size_mb(file)
    if size_mb > 0.1:  # Show files larger than 100KB
        large_files.append((file, size_mb))

print("\nLarge tracked files:")
for file, size in sorted(large_files, key=lambda x: x[1], reverse=True):
    print(f"{file}: {size:.2f}MB")