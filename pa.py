from pathlib import Path
import tarfile
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_path = Path("data")

# Find all CSV files directly in data directory
direct_csv_files = sorted(data_path.glob("*.csv"))
print(f"Direct CSV files found: {[f.name for f in direct_csv_files]}")

# Find and extract CSVs from ZIP files
csv_from_zips = {}
zip_files = sorted(data_path.glob("*.zip"))
print(f"ZIP files found: {[f.name for f in zip_files]}")

for zip_file in zip_files:
    try:
        with zipfile.ZipFile(zip_file, allowZip64=True) as zf:
            csv_members = [
                name for name in zf.namelist() if name.lower().endswith(".csv")
            ]
            if csv_members:
                print(f"  CSV files in {zip_file.name}: {csv_members}")
                for csv_member in csv_members:
                    csv_from_zips[csv_member] = (zip_file, csv_member)
            else:
                print(f"  No CSV files found in {zip_file.name}")
    except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
        print(f"  Warning: Could not read {zip_file.name}: {type(e).__name__}: {e}")

# Combine all CSV sources
all_csv_sources = [(f, None) for f in direct_csv_files] + list(csv_from_zips.values())
print(f"\nTotal CSV files to process: {len(all_csv_sources)}")
print(f"  Direct files: {len(direct_csv_files)}")
print(f"  Files from ZIPs: {len(csv_from_zips)}")

# for file_info in all_csv_sources:


# csv_member = csv_members[0]
# csv_uri = f"zip://{zip_path.as_posix()}!{csv_member}"
# in_out_time = pd.read_csv(csv_uri)

# Read the ZIP directly in pandas (no extraction on disk).

## Exploration
### Type

### Nombre

### Maximum

### Minimum

### Moyenne

### Distribution des données

## Valeur
### Faire les médianes
