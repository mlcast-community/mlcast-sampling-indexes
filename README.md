# MLCast sampling indexes

Generate and validate Parquet sampling indexes for radar datasets listed in [`mlcast-datasets`](https://github.com/mlcast-community/mlcast-datasets). The script uses MLCast's `build-sampling-index` and `validate-sampling-index` commands, with additional preprocessing for Germany and the UK.

## Requirements

- Python with `mlcast`, `numpy`, `pyarrow`, `xarray`, `zarr` and `pyyaml`
- Git and a local clone of `mlcast-datasets`
- Local copies of the radar Zarr datasets (the script does not download data from S3)

## Usage

```bash
git clone https://github.com/mlcast-community/mlcast-datasets.git

export MLCAST_SAMPLING_ROOT=/path/to/output
CATALOG_REF=2f45743fac43a1723b877c39839cc77d76af13b5

# Inspect the datasets in this catalog revision.
python build_sampling_indexes.py sources all \
  --catalog-repo ./mlcast-datasets \
  --catalog-ref "$CATALOG_REF" \
  --data-root /path/to/observations

# Build indexes for all supported countries on a compute node.
python build_sampling_indexes.py build all \
  --catalog-repo ./mlcast-datasets \
  --catalog-ref "$CATALOG_REF" \
  --data-root /path/to/observations

# Or build an index for a single country.
python build_sampling_indexes.py build FR \
  --catalog-repo ./mlcast-datasets \
  --catalog-ref "$CATALOG_REF" \
  --data-root /path/to/observations

# Validate existing indexes.
MLCAST_INDEX_DIR=/path/to/output/indexes \
  python build_sampling_indexes.py validate all
```

Use `all` for every supported country (BE, FR, IT, DE, DK and UK), or a country code for a single index. Germany and UK preprocessing is applied automatically. Use `--source-override COUNTRY=/path/to/dataset.zarr` if a local dataset path differs from the catalog layout (as with the Belgian archive on VUB-HPC). The example commit contains all six entries; it is a pinned Git commit, **not** a catalog release tag.

## Sampling settings

- Datacube size: 24 frames × 256 × 256 pixels
- Strides: 3 frames in time, 16 pixels in each spatial direction
- Missing-data limit: 157,286 pixels per datacube (about 10%)
- Wet threshold: 0.1 mm/h for BE, FR, IT and UK; 7 dBZ for DK; equivalent to 0.1 mm/h for DE

Each output is named `<COUNTRY>_sampling_index.parquet` and contains `t`, `x`, `y`, `nan_count`, `sum`, `mean` and `frac_wet`. New builds also write a `.source.json` provenance file. Validation of DE and UK needs the corresponding QC reports and UK exclusion manifest.

For **Germany**, values of `999` are treated as missing and five-minute accumulations are converted to mm/h for index statistics. For the **UK**, frames with negative rainfall values, and all sampling windows overlapping those frames, are excluded. The UK filter does not detect artifacts containing only nonnegative values.

