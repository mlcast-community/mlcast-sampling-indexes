#!/usr/bin/env python3
"""Build and validate MLCast radar sampling indexes from a pinned dataset catalog.

The source catalog is read at a specified mlcast-datasets Git tag or commit.
Radar data can be read from local mirrors, with explicit path overrides where needed.
UK windows overlapping a negative-rainfall frame are excluded; German 999 fill
values are masked and five-minute rainfall amounts converted to mm/h for statistics.

Requires mlcast, numpy, pyarrow, xarray, zarr, and PyYAML.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
import zarr
import yaml

TIME_DEPTH = 24
HEIGHT = WIDTH = 256
STEP_T = 3
STEP_X = STEP_Y = 16
MAX_NAN = 157286
WET_THRESHOLD_MM_H = 0.1
DE_AMOUNT_TO_RATE = 12.0  # kg/m2 in five minutes -> mm/h
DE_MISSING_VALUE = np.float32(999.0)
QC_VERSION = "v2"

# Each country describes how to find a source in the official Intake catalog.
# Dataset locations and versions are read from the selected catalog revision.
# Tuple fields: unique catalog URL marker, Zarr variable, frame cadence (minutes).
COUNTRY_CONFIG = {
    "BE": ("/BE-RMI-radclim-mfb/", "rain_rate", 5),
    "FR": ("/FR-MF-prate/", "prate", 5),
    "IT": ("/IT-DPC-SRI/", "RR", 5),
    "DE": ("/radklim/", "rainfall_amount", 5),
    "DK": ("/dmi/", "dbz", 10),
    "UK": ("/UK-METOFFICE-RADAR/", "RR", 5),
}
CATALOG_FILE = "src/mlcast_datasets/catalog/precipitation/catalog.yml"
CATALOG_REPOSITORY = "https://github.com/mlcast-community/mlcast-datasets"

# Populated from a selected catalog revision for `build` and `sources`.
SOURCES: dict[str, tuple[str, str, int]] = {}
CATALOG_PROVENANCE: dict[str, object] = {}


def read_catalog(repo: Path, ref: str) -> tuple[dict, str]:
    """Read the Intake precipitation catalog at one immutable Git commit."""
    if not (repo / ".git").exists() and not (repo / "HEAD").is_file():
        raise FileNotFoundError(
            f"Not a Git checkout: {repo}. Clone {CATALOG_REPOSITORY} first."
        )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        text=True,
    ).strip()
    content = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{CATALOG_FILE}"],
        text=True,
    )
    catalog = yaml.safe_load(content)
    entries = catalog.get("sources", {}) if isinstance(catalog, dict) else {}
    if not isinstance(entries, dict):
        raise ValueError(f"Unexpected dataset catalog structure at {commit}")
    return entries, commit


def parse_source_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    for item in values:
        country, sep, pathname = item.partition("=")
        if not sep or country not in COUNTRY_CONFIG or not pathname.strip():
            raise ValueError(f"Expected COUNTRY=/path/to/source.zarr, got {item!r}")
        if country in overrides:
            raise ValueError(f"Duplicate source override: {country}")
        overrides[country] = Path(pathname).expanduser().resolve()
    return overrides


def configure_sources(args: argparse.Namespace) -> None:
    """Resolve the versioned catalog paths to user-supplied local Zarr mirrors."""
    if not args.catalog_repo or not args.catalog_ref:
        raise ValueError("Specify --catalog-repo and --catalog-ref")
    entries, commit = read_catalog(Path(args.catalog_repo).expanduser().resolve(), args.catalog_ref)
    overrides = parse_source_overrides(args.source_override)
    root_arg = args.data_root or os.environ.get("MLCAST_OBS_ROOT")
    root = Path(root_arg).expanduser().resolve() if root_arg else None

    resolved: dict[str, tuple[str, str, int]] = {}
    details: dict[str, dict[str, str]] = {}
    requested = selected_countries(args.country)
    for country in requested:
        marker, variable, cadence = COUNTRY_CONFIG[country]
        matches = []
        for entry_name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            url = (entry.get("args") or {}).get("urlpath", "")
            if isinstance(url, str) and marker in url:
                matches.append((entry_name, url))
        if country == "DE":
            matches = [
                (name, url) for name, url in matches
                if url.rstrip("/").endswith("/5_minutes.zarr")
            ]
        if len(matches) != 1:
            raise ValueError(
                f"{country}: expected exactly one source containing {marker!r} "
                f"in catalog {commit}; found {len(matches)}. "
                "Choose a catalog release containing every supported country."
            )
        name, url = matches[0]
        parsed = urlparse(url)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"{country}: unexpected catalog source URL {url!r}")
        if country in overrides:
            local = overrides[country]
        elif root is not None:
            local = root / parsed.path.lstrip("/")
        else:
            local = None
        if local is not None:
            resolved[country] = (str(local), variable, cadence)
        details[country] = {
            "catalog_entry": name,
            "catalog_url": url,
            "local_path": str(local) if local is not None else "not mapped",
            "variable": variable,
            "cadence_minutes": str(cadence),
        }

    SOURCES.clear()
    SOURCES.update(resolved)
    CATALOG_PROVENANCE.clear()
    CATALOG_PROVENANCE.update({
        "catalog_repository": CATALOG_REPOSITORY,
        "catalog_ref": args.catalog_ref,
        "catalog_commit": commit,
        "sources": details,
    })
    if args.action == "build":
        missing = [country for country in requested if country not in resolved]
        if missing:
            raise ValueError(
                f"No local mirror configured for {missing}. "
                "Set --data-root and/or --source-override COUNTRY=/path/to.zarr"
            )


def show_sources() -> None:
    """Print catalog-provided source URLs and their local mirror locations."""
    print(f"mlcast-datasets commit: {CATALOG_PROVENANCE['catalog_commit']}")
    for country, info in CATALOG_PROVENANCE["sources"].items():
        print(f"{country}: {info['catalog_entry']}")
        print(f"  catalog: {info['catalog_url']}")
        print(f"  local:   {info['local_path']}")


def output_root() -> Path:
    """Return the output directory for indexes and QC files."""
    root = os.environ.get("MLCAST_SAMPLING_ROOT")
    if root is None:
        scratch = os.environ.get("VSC_SCRATCH")
        if scratch is None:
            raise EnvironmentError(
                "Set MLCAST_SAMPLING_ROOT (or VSC_SCRATCH) before building indexes"
            )
        root = str(Path(scratch) / "mlcast")
    return Path(root).expanduser().resolve()


def paths(country: str) -> tuple[Path, Path, Path]:
    zarr_path = Path(SOURCES[country][0])
    indexes = output_root() / "indexes"
    qc = output_root() / "qc"
    indexes.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    return zarr_path, indexes, qc


def mlcast_executable() -> Path:
    """Find the MLCast CLI, preferring an explicitly configured installation."""
    configured = os.environ.get("MLCAST_CLI")
    if configured:
        executable = Path(configured).expanduser()
        if executable.is_file():
            return executable.resolve()
        resolved = shutil.which(configured)
        if resolved:
            return Path(resolved)
        raise FileNotFoundError(f"MLCast executable not found: {configured}")

    vsc_data = os.environ.get("VSC_DATA")
    if vsc_data:
        executable = Path(vsc_data) / "mlcast/.venv/bin/mlcast"
        if executable.is_file():
            return executable

    resolved = shutil.which("mlcast")
    if resolved:
        return Path(resolved)
    raise FileNotFoundError("MLCast CLI not found; set MLCAST_CLI")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.write_text(content)
    os.replace(temp, path)


class RadarSource:
    """Read a Zarr radar variable by named dimensions, without loading all frames."""

    def __init__(self, path: Path, variable: str, time_name: str = "time") -> None:
        self.path = path.resolve()
        self.variable = variable
        self.time_name = time_name
        self.dataset = xr.open_zarr(
            str(self.path), consolidated=False, chunks=None, decode_times=True
        )
        try:
            data = self.dataset[variable]
            self.dims = tuple(data.dims)
            if len(self.dims) != 3 or set(self.dims) != {time_name, "y", "x"}:
                raise ValueError(
                    f"{variable}: unsupported dimension names/order {self.dims}"
                )
            self.times = np.asarray(self.dataset[time_name].values).astype(
                "datetime64[ns]"
            )
            if self.times.ndim != 1 or len(self.times) != data.sizes[time_name]:
                raise ValueError("Invalid time coordinate")
            if np.isnat(self.times).any() or np.any(self.times[1:] <= self.times[:-1]):
                raise ValueError("Time coordinate must be strictly increasing")
            self.shape = (len(self.times), data.sizes["y"], data.sizes["x"])
            self.attributes = dict(data.attrs)
            self.global_attributes = dict(self.dataset.attrs)
            self.array = zarr.open_group(str(self.path), mode="r")[variable]
            if tuple(self.array.shape) != tuple(data.shape):
                raise ValueError("xarray and raw Zarr array shapes disagree")
        except Exception:
            self.dataset.close()
            raise

    def frame(self, t: int) -> np.ndarray:
        selectors = tuple(
            t if dim == self.time_name else slice(None) for dim in self.dims
        )
        image = np.asarray(self.array[selectors])
        spatial_dims = [dim for dim in self.dims if dim != self.time_name]
        return np.transpose(image, [spatial_dims.index("y"), spatial_dims.index("x")])

    def require_canonical(self) -> None:
        # The existing MLCast CLI has not been checked for noncanonical axes.
        if self.dims != (self.time_name, "y", "x"):
            raise ValueError(
                f"MLCast CLI requires a reviewed adapter for dims {self.dims}; "
                "do not reinterpret x/y silently"
            )

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "RadarSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def source_info(country: str, canonical: bool = True) -> tuple[int, int]:
    zarr_path, _, _ = paths(country)
    with RadarSource(zarr_path, SOURCES[country][1]) as source:
        if canonical:
            source.require_canonical()
        print(
            f"{country}: {source.path}; dims={source.dims}; shape={source.shape}",
            flush=True,
        )
        first_year = int(source.times[0].astype("datetime64[Y]").astype(int) + 1970)
        last_year = int(source.times[-1].astype("datetime64[Y]").astype(int) + 1970)
        return first_year, last_year


def mlcast_builder_command(country: str, output: Path) -> list[str]:
    zarr_path, _, _ = paths(country)
    _, variable, cadence = SOURCES[country]
    return [
        str(mlcast_executable()), "build-sampling-index", str(zarr_path),
        "--data-var", variable, "--time-var", "time",
        "--time-depth", str(TIME_DEPTH), "--width", str(WIDTH),
        "--height", str(HEIGHT), "--step-t", str(STEP_T),
        "--step-x", str(STEP_X), "--step-y", str(STEP_Y),
        "--max-nan", str(MAX_NAN), "--time-step-minutes", str(cadence),
        "--device", "cpu", "--workers", "4", "--output", str(output),
    ]


def build_standard(country: str) -> None:
    source_info(country)
    _, indexes, _ = paths(country)
    dest = indexes / f"{country}_sampling_index.parquet"
    if dest.exists():
        print(f"Keeping existing index: {dest}", flush=True)
        return
    temp = indexes / f".{country}.{os.getpid()}.part.parquet"
    if temp.exists():
        raise FileExistsError(temp)
    subprocess.run(mlcast_builder_command(country, temp), check=True)
    if dest.exists():
        raise FileExistsError(dest)
    os.replace(temp, dest)
    print(f"Built {dest}", flush=True)


# ---------------------------------------------------------------------------
# UK: scan every source frame, collect a manifest, and exclude whole windows.
# ---------------------------------------------------------------------------

def overlaps_bad_frame(
    starts: np.ndarray, bad_frames: np.ndarray, depth: int
) -> np.ndarray:
    """Return a mask for windows containing at least one flagged time index.

    bad_frames must be sorted; checking them is O(log N) per candidate.
    """
    if depth <= 0:
        raise ValueError("Time depth must be positive")
    starts = np.asarray(starts, dtype=np.int64)
    bad_frames = np.asarray(bad_frames, dtype=np.int64)
    positions = np.searchsorted(bad_frames, starts)
    valid = positions < bad_frames.size
    overlap = np.zeros(starts.shape, dtype=bool)
    overlap[valid] = bad_frames[positions[valid]] < starts[valid] + depth
    return overlap


def scan_uk_year(year: int) -> None:
    zarr_path, _, qc = paths("UK")
    folder = qc / f"uk_scan_{QC_VERSION}" / f"year_{year}"
    folder.mkdir(parents=True, exist_ok=True)
    with RadarSource(zarr_path, "RR") as source:
        source.require_canonical()
        start = int(np.searchsorted(source.times, np.datetime64(f"{year}-01-01")))
        stop = int(np.searchsorted(source.times, np.datetime64(f"{year + 1}-01-01")))
        if stop <= start:
            raise ValueError(f"UK Zarr contains no timestamps for {year}")
        settings = {
            "source": str(source.path), "year": year, "start": start, "stop": stop,
            "block_frames": 256, "shape": list(source.shape), "dims": list(source.dims),
        }
        config = folder / "config.json"
        if config.exists() and json.loads(config.read_text()) != settings:
            raise RuntimeError(f"Stale UK scan configuration in {folder}")
        if not config.exists():
            atomic_text(config, json.dumps(settings, indent=2) + "\n")

        negative_pixels = negative_frames = 0
        for first in range(start, stop, 256):
            last = min(stop, first + 256)
            stem = f"{first:08d}_{last:08d}"
            frame_file = folder / f"{stem}.frames.csv"
            done_file = folder / f"{stem}.done.json"
            if frame_file.exists() and done_file.exists():
                done = json.loads(done_file.read_text())
                with frame_file.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if (
                    done.get("start") == first
                    and done.get("stop") == last
                    and len(rows) == done.get("negative_frames")
                    and sum(int(row["negative_pixels"]) for row in rows)
                    == done.get("negative_pixels")
                ):
                    negative_pixels += done["negative_pixels"]
                    negative_frames += done["negative_frames"]
                    continue
            frame_file.unlink(missing_ok=True)
            done_file.unlink(missing_ok=True)
            rows = ["t,timestamp_utc,negative_pixels"]
            block_pixels = block_frames = 0
            for t in range(first, last):
                frame = source.frame(t)
                count = int(np.count_nonzero(np.isfinite(frame) & (frame < 0)))
                if count:
                    timestamp = np.datetime_as_string(
                        source.times[t], unit="s", timezone="UTC"
                    )
                    rows.append(f"{t},{timestamp},{count}")
                    block_frames += 1
                    block_pixels += count
            atomic_text(frame_file, "\n".join(rows) + "\n")
            atomic_text(
                done_file,
                json.dumps(
                    {"start": first, "stop": last, "negative_frames": block_frames,
                     "negative_pixels": block_pixels},
                    indent=2,
                ) + "\n",
            )
            negative_frames += block_frames
            negative_pixels += block_pixels
        atomic_text(
            folder / "complete.json",
            json.dumps(
                {**settings, "negative_frames": negative_frames,
                 "negative_pixels": negative_pixels,
                 "blocks": (stop - start + 255) // 256},
                indent=2,
            ) + "\n",
        )
        print(
            f"UK {year}: {stop - start:,} frames; "
            f"{negative_frames:,} bad frames; {negative_pixels:,} negative pixels",
            flush=True,
        )


def collect_uk_manifest() -> None:
    zarr_path, _, qc = paths("UK")
    destination = qc / f"UK_bad_frames_negative_{QC_VERSION}.csv"
    summary = destination.with_suffix(".json")
    if destination.exists() or summary.exists():
        raise FileExistsError("UK manifest already exists; use a fresh QC version")
    data = []
    years = []
    cursor = 0
    with RadarSource(zarr_path, "RR") as source:
        source.require_canonical()
        first_year = int(source.times[0].astype("datetime64[Y]").astype(int) + 1970)
        last_year = int(source.times[-1].astype("datetime64[Y]").astype(int) + 1970)
        for year in range(first_year, last_year + 1):
            folder = qc / f"uk_scan_{QC_VERSION}" / f"year_{year}"
            info = json.loads((folder / "complete.json").read_text())
            if (info["source"] != str(zarr_path.resolve()) or info["start"] != cursor):
                raise ValueError(f"Gap, overlap or wrong UK source in {year}")
            parts = sorted(folder.glob("*.done.json"))
            if len(parts) != info["blocks"]:
                raise ValueError(f"Incomplete UK scan year {year}")
            neg_frames = neg_pixels = 0
            for done_file in parts:
                done = json.loads(done_file.read_text())
                if done["start"] != cursor or done["stop"] <= cursor:
                    raise ValueError(f"Gap or overlap at {done_file}")
                block_name = done_file.name.removesuffix(".done.json")
                frame_file = folder / f"{block_name}.frames.csv"
                with frame_file.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != done["negative_frames"]:
                    raise ValueError(f"Corrupt UK scan block {frame_file}")
                if (
                    sum(int(row["negative_pixels"]) for row in rows)
                    != done["negative_pixels"]
                ):
                    raise ValueError(f"Inconsistent UK scan block {frame_file}")
                prev = cursor - 1
                for row in rows:
                    t = int(row["t"])
                    if not (prev < t < done["stop"]):
                        raise ValueError(
                            f"Unsorted negative-frame list in {frame_file}"
                        )
                    data.append((t, row["timestamp_utc"], int(row["negative_pixels"])))
                    prev = t
                cursor = done["stop"]
                neg_frames += len(rows)
                neg_pixels += done["negative_pixels"]
            if cursor != info["stop"] or neg_frames != info["negative_frames"]:
                raise ValueError(f"Incomplete UK year {year}")
            if neg_pixels != info["negative_pixels"]:
                raise ValueError(f"Inconsistent UK negative-pixel count for {year}")
            years.append({"year": year, "frames": info["stop"] - info["start"],
                          "negative_frames": neg_frames, "negative_pixels": neg_pixels})
        if cursor != source.shape[0]:
            raise ValueError(
                f"UK scan only covers {cursor} of {source.shape[0]} frames"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t", "timestamp_utc", "negative_pixels"])
        writer.writerows(data)
    os.replace(temp, destination)
    report = {
        "rule": (
            "reject every window containing a frame with any "
            "negative finite RR pixel"
        ),
        "source": str(zarr_path.resolve()), "frames_scanned": cursor,
        "negative_frames": len(data),
        "negative_pixels": sum(row[2] for row in data),
        "manifest_sha256": sha256_file(destination), "by_year": years,
        "limitation": "positive-only radar artifacts are not identified",
    }
    atomic_text(summary, json.dumps(report, indent=2) + "\n")
    print(f"UK manifest: {len(data):,} excluded frames", flush=True)


def read_bad_frames(path: Path) -> np.ndarray:
    with path.open(newline="") as source:
        values = sorted({int(row["t"]) for row in csv.DictReader(source)})
    return np.asarray(values, dtype=np.int64)


def filter_uk_index(raw: Path, output: Path, manifest: Path) -> None:
    """Stream through a raw MLCast Parquet without loading it all into RAM."""
    if output.exists():
        raise FileExistsError(output)
    bad_frames = read_bad_frames(manifest)
    parquet = pq.ParquetFile(raw)
    schema = parquet.schema_arrow
    if schema.get_field_index("t") < 0:
        raise ValueError("Sampling index lacks the global time-index column 't'")
    temp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    original = rejected = 0
    try:
        with pq.ParquetWriter(str(temp), schema=schema, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=250_000):
                starts = batch.column(batch.schema.get_field_index("t")).to_numpy(
                    zero_copy_only=False
                )
                keep = ~overlaps_bad_frame(starts, bad_frames, TIME_DEPTH)
                original += len(starts)
                rejected += int(np.count_nonzero(~keep))
                if np.any(keep):
                    writer.write_batch(batch.filter(pa.array(keep)))
                if original % 25_000_000 < len(starts):
                    print(f"UK candidates scanned: {original:,}", flush=True)
        remaining = original - rejected
        if remaining <= 0 or pq.read_metadata(temp).num_rows != remaining:
            raise ValueError("Unexpected UK index row count")
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    report = {
        "country": "UK", "qc_version": QC_VERSION, "input_index": str(raw),
        "output_index": str(output), "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest), "bad_frames": len(bad_frames),
        "time_depth": TIME_DEPTH, "rows_original": original,
        "rows_rejected": rejected, "rows_remaining": remaining,
        "note": (
            "negative-associated frames removed; "
            "positive-only artifacts not detected"
        ),
    }
    atomic_text(output.with_suffix(".qc.json"), json.dumps(report, indent=2) + "\n")


def build_uk() -> None:
    source_info("UK")
    _, indexes, qc = paths("UK")
    manifest = qc / f"UK_bad_frames_negative_{QC_VERSION}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"UK scan/collection missing: {manifest}")
    output = indexes / "UK_sampling_index.parquet"
    if output.exists():
        raise FileExistsError(output)
    raw = indexes / f".UK_raw_{os.getpid()}.parquet"
    if raw.exists():
        raise FileExistsError(raw)
    subprocess.run(mlcast_builder_command("UK", raw), check=True)
    filter_uk_index(raw, output, manifest)
    raw.unlink()
    print(f"Built {output}", flush=True)


# ---------------------------------------------------------------------------
# DE: treat Zarr fill code 999 as missing, express mean and sum in mm/h.
# ---------------------------------------------------------------------------

def de_spatial_sums(image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Compute all 256x256 crop sums using a summed-area table."""
    integral = np.pad(
        np.cumsum(np.cumsum(image, axis=0, dtype=np.float64), axis=1, dtype=np.float64),
        ((1, 0), (1, 0)),
    )
    y0, x0 = ys[:, None], xs[None, :]
    y1, x1 = y0 + HEIGHT, x0 + WIDTH
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def de_spatial_features(frame: np.ndarray, ys: np.ndarray, xs: np.ndarray):
    """Return sums in mm/h plus counts of valid and wet pixels."""
    frame = np.asarray(frame, dtype=np.float32)
    valid = np.isfinite(frame) & (frame != DE_MISSING_VALUE)
    if np.any(valid & (frame < 0)):
        raise ValueError("German radar contains unexpected negative rainfall amounts")
    rate = np.where(valid, frame.astype(np.float64) * DE_AMOUNT_TO_RATE, 0.0)
    wet = valid & (rate >= WET_THRESHOLD_MM_H)
    return (
        de_spatial_sums(rate, ys, xs),
        de_spatial_sums(valid.astype(np.uint8), ys, xs).astype(np.int64),
        de_spatial_sums(wet.astype(np.uint8), ys, xs).astype(np.int64),
    )


def create_de_schema(zarr_path: Path, destination: Path) -> None:
    """Build a schema template with the installed MLCast CLI.

    The synthetic Zarr is used only to obtain a compatible Parquet schema.
    No synthetic rows are included in the German index.
    """
    with RadarSource(zarr_path, "rainfall_amount") as source:
        attrs = {k: v for k, v in source.attributes.items() if k != "_FillValue"}
        attrs["standard_name"] = "rainfall_flux"
        attrs["units"] = "kg m-2 h-1"
        global_attrs = source.global_attributes

    times = (
        np.datetime64("2001-01-01T00:00:00")
        + np.arange(27) * np.timedelta64(5, "m")
    )
    seed_dataset = xr.Dataset(
        {"rainfall_amount": (("time", "y", "x"),
                             np.zeros((27, HEIGHT, WIDTH), dtype=np.float32), attrs)},
        coords={"time": times, "y": np.arange(HEIGHT), "x": np.arange(WIDTH)},
        attrs=global_attrs,
    )
    with tempfile.TemporaryDirectory(prefix="mlcast_de_schema_") as work:
        folder = Path(work)
        seed = folder / "seed.zarr"
        donor = folder / "index.parquet"
        seed_dataset.to_zarr(str(seed), mode="w", consolidated=False, zarr_format=2)
        cmd = [
            str(mlcast_executable()), "build-sampling-index", str(seed),
            "--data-var", "rainfall_amount", "--time-var", "time",
            "--time-depth", str(TIME_DEPTH), "--width", str(WIDTH),
            "--height", str(HEIGHT), "--step-t", str(STEP_T),
            "--step-x", str(STEP_X), "--step-y", str(STEP_Y),
            "--max-nan", str(MAX_NAN), "--time-step-minutes", "5",
            "--device", "cpu", "--workers", "1", "--data-kind", "rainrate",
            "--wet-threshold", str(WET_THRESHOLD_MM_H), "--output", str(donor),
        ]
        subprocess.run(cmd, check=True)
        schema = pq.read_schema(donor)
        expected = ["t", "x", "y", "nan_count", "sum", "mean", "frac_wet"]
        if schema.names != expected:
            raise ValueError(f"Unsupported installed MLCast schema: {schema.names}")
        metadata = {
            key: val.replace(str(seed).encode(), str(zarr_path.resolve()).encode())
            for key, val in (schema.metadata or {}).items()
        }
        metadata[b"mlcast_qc_version"] = QC_VERSION.encode()
        metadata[b"mlcast_qc_source_data_units"] = b"kg m-2 per five minutes"
        metadata[b"mlcast_qc_index_statistics_units"] = b"mm h-1"
        metadata[b"mlcast_qc_missing_value"] = b"999.0"
        metadata[b"mlcast_qc_amount_to_rate_factor"] = b"12"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_batches([], schema=schema.with_metadata(metadata)),
            destination,
        )


def build_de() -> None:
    zarr_path, indexes, qc = paths("DE")
    source_info("DE", canonical=False)
    output = indexes / "DE_sampling_index.parquet"
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="de_schema_", dir=qc) as work:
        schema_file = Path(work) / "schema.parquet"
        create_de_schema(zarr_path, schema_file)
        schema = pq.ParquetFile(schema_file).schema_arrow
        expected = ["t", "x", "y", "nan_count", "sum", "mean", "frac_wet"]
        if schema.names != expected:
            raise ValueError(f"Unexpected MLCast index columns: {schema.names}")
        with RadarSource(zarr_path, "rainfall_amount") as source:
            times = source.times
            frames, ny, nx = source.shape
            xs = np.arange(0, nx - WIDTH + 1, STEP_X, dtype=np.int64)
            ys = np.arange(0, ny - HEIGHT + 1, STEP_Y, dtype=np.int64)
            if not xs.size or not ys.size or frames < TIME_DEPTH:
                raise ValueError("German Zarr does not fit the configured datacube")
            # Reject windows spanning any non-five-minute timestamp interval.
            is_gap = np.diff(times) != np.timedelta64(5, "m")
            gap_prefix = np.r_[0, np.cumsum(is_gap, dtype=np.int64)]
            coords_y, coords_x = np.meshgrid(ys, xs, indexing="ij")
            base_x = coords_x.ravel()
            base_y = coords_y.ravel()
            pixels_per_cube = TIME_DEPTH * HEIGHT * WIDTH

            rolling: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque()
            cumulative_sum = np.zeros((len(ys), len(xs)), dtype=np.float64)
            cumulative_valid = np.zeros_like(cumulative_sum, dtype=np.int64)
            cumulative_wet = np.zeros_like(cumulative_valid)
            pending: dict[str, list[np.ndarray]] = {name: [] for name in expected}
            pending_rows = emitted = rejected = count_999 = 0
            windows = 0
            temp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
            writer = None

            def flush() -> None:
                nonlocal pending_rows, emitted
                if not pending_rows:
                    return
                arrays = [
                    pa.array(np.concatenate(pending[field.name]), type=field.type)
                    for field in schema
                ]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
                for key in pending:
                    pending[key].clear()
                emitted += pending_rows
                pending_rows = 0

            try:
                writer = pq.ParquetWriter(temp, schema=schema, compression="zstd")
                for frame_index in range(frames):
                    frame = source.frame(frame_index)
                    count_999 += int(np.count_nonzero(frame == DE_MISSING_VALUE))
                    features = de_spatial_features(frame, ys, xs)
                    rolling.append(features)
                    for target, feature in zip(
                        (cumulative_sum, cumulative_valid, cumulative_wet), features
                    ):
                        target += feature
                    if len(rolling) > TIME_DEPTH:
                        previous = rolling.popleft()
                        for target, feature in zip(
                            (cumulative_sum, cumulative_valid, cumulative_wet), previous
                        ):
                            target -= feature
                    if len(rolling) < TIME_DEPTH:
                        continue
                    start = frame_index - TIME_DEPTH + 1
                    if (
                        start % STEP_T
                        or gap_prefix[start + TIME_DEPTH - 1] != gap_prefix[start]
                    ):
                        continue
                    windows += 1
                    nan_count = pixels_per_cube - cumulative_valid
                    keep = (nan_count <= MAX_NAN) & (cumulative_valid > 0)
                    rejected += int(keep.size - np.count_nonzero(keep))
                    if not np.any(keep):
                        continue
                    selected = keep.ravel()
                    count = int(selected.sum())
                    totals = np.maximum(cumulative_sum.ravel()[selected], 0.0)
                    valid_counts = cumulative_valid.ravel()[selected]
                    wet_counts = cumulative_wet.ravel()[selected]
                    values = {
                        "t": np.full(count, start, dtype=np.int64),
                        "x": base_x[selected], "y": base_y[selected],
                        "nan_count": nan_count.ravel()[selected],
                        "sum": totals, "mean": totals / valid_counts,
                        "frac_wet": wet_counts / valid_counts,
                    }
                    for key in pending:
                        pending[key].append(values[key])
                    pending_rows += count
                    if pending_rows >= 300_000:
                        flush()
                    if start % 9000 == 0:
                        print(
                            f"DE t={start:,}; rows={emitted + pending_rows:,}",
                            flush=True,
                        )
                flush()
                writer.close()
                writer = None
                if pq.read_metadata(temp).num_rows != emitted:
                    raise AssertionError("German Parquet row-count mismatch")
                os.replace(temp, output)
            finally:
                if writer is not None:
                    writer.close()
                temp.unlink(missing_ok=True)
        report = {
            "country": "DE", "qc_version": QC_VERSION,
            "source_zarr": str(zarr_path), "source_variable": "rainfall_amount",
            "missing_value": 999.0, "source_units": "kg m-2 per five minutes",
            "index_statistics_units": "mm h-1", "rate_multiplier": DE_AMOUNT_TO_RATE,
            "wet_threshold_mm_h": WET_THRESHOLD_MM_H,
            "time_depth": TIME_DEPTH, "height": HEIGHT, "width": WIDTH,
            "step_t": STEP_T, "step_x": STEP_X, "step_y": STEP_Y,
            "max_nan": MAX_NAN, "candidate_time_windows": windows,
            "candidate_rows_rejected_nan": rejected, "candidate_rows_written": emitted,
            "raw_999_pixels": count_999,
            "note": "source Zarr untouched; German training loader must also mask 999",
        }
        atomic_text(output.with_suffix(".qc.json"), json.dumps(report, indent=2) + "\n")
        print(f"Built {output}: {emitted:,} rows", flush=True)


# ---------------------------------------------------------------------------
# Build and validation commands.
# ---------------------------------------------------------------------------

def run_full_uk() -> None:
    """Scan the UK archive, then build an index excluding flagged frames."""
    _, _, qc = paths("UK")
    manifest = qc / f"UK_bad_frames_negative_{QC_VERSION}.csv"
    summary = manifest.with_suffix(".json")

    if manifest.exists() or summary.exists():
        if not manifest.is_file() or not summary.is_file():
            raise FileExistsError("Incomplete UK QC manifest or summary")
        report = json.loads(summary.read_text())
        source_path = Path(SOURCES["UK"][0])
        if (
            report.get("source") != str(source_path.resolve())
            or report.get("manifest_sha256") != sha256_file(manifest)
        ):
            raise ValueError("Existing UK manifest does not match its source or checksum")
        print(f"Reusing checked UK manifest: {manifest}", flush=True)
    else:
        first_year, last_year = source_info("UK")
        for year in range(first_year, last_year + 1):
            print(f"UK: scanning {year}/{last_year}", flush=True)
            scan_uk_year(year)
        collect_uk_manifest()

    build_uk()


def selected_countries(mode: str) -> tuple[str, ...]:
    """Return all configured countries or a single selected country."""
    if mode == "all":
        return tuple(COUNTRY_CONFIG)
    if mode in COUNTRY_CONFIG:
        return (mode,)
    raise ValueError(f"Unknown country: {mode}")


def save_build_provenance(country: str, output: Path) -> None:
    """Record the catalog revision and local source used for a new index."""
    source_path = Path(SOURCES[country][0])
    attrs = dict(zarr.open_group(str(source_path), mode="r").attrs)
    manifest = {
        "country": country,
        "index_file": output.name,
        "catalog_repository": CATALOG_PROVENANCE["catalog_repository"],
        "catalog_commit": CATALOG_PROVENANCE["catalog_commit"],
        "catalog_source": CATALOG_PROVENANCE["sources"][country],
        "dataset_version_metadata": attrs.get("mlcast_dataset_version"),
        "dataset_identifier": attrs.get("mlcast_dataset_identifier"),
        "created_with": attrs.get("mlcast_created_with"),
        "sampling": {
            "time_depth": TIME_DEPTH,
            "width": WIDTH,
            "height": HEIGHT,
            "step_t": STEP_T,
            "step_x": STEP_X,
            "step_y": STEP_Y,
            "max_nan": MAX_NAN,
        },
    }
    atomic_text(output.with_suffix(".source.json"), json.dumps(manifest, indent=2) + "\n")


def build_country(country: str) -> None:
    """Build a country's index, without overwriting an existing result."""
    _, indexes, _ = paths(country)
    output = indexes / f"{country}_sampling_index.parquet"
    if output.exists():
        print(f"Keeping existing index: {output}", flush=True)
        return

    path = Path(SOURCES[country][0])
    if not path.is_dir():
        raise FileNotFoundError(
            f"No local Zarr mirror for {country}: {path}; use --source-override"
        )
    if country == "UK":
        run_full_uk()
    elif country == "DE":
        build_de()
    else:
        build_standard(country)
    if not output.is_file():
        raise RuntimeError(f"Index builder did not produce {output}")
    save_build_provenance(country, output)


def find_qc_report(country: str, index_dir: Path) -> Path:
    """Locate the QC report for a freshly built or previously renamed index."""
    root = output_root()
    report_dir = Path(os.environ.get("MLCAST_QC_REPORT_DIR", root / "indexes"))
    candidates = [
        index_dir / f"{country}_sampling_index.qc.json",
        report_dir / f"{country}_sampling_index.qc.json",
        report_dir / f"{country}_full_nan10_qc_{QC_VERSION}.qc.json",
        root / "release_v1" / f"{country}_full_nan10.qc.json",
        root / "qc" / f"{country}_sampling_index.qc.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {country} QC report. Looked in: {candidates}")


def find_uk_manifest(index_dir: Path) -> Path:
    """Find the UK exclusion manifest retained with the QC outputs."""
    root = output_root()
    override = os.environ.get("MLCAST_UK_MANIFEST")
    candidates = [Path(override)] if override else [
        root / "qc" / f"UK_bad_frames_negative_{QC_VERSION}.csv",
        index_dir / "UK_excluded_frames.csv",
        index_dir / f"UK_bad_frames_negative_{QC_VERSION}.csv",
        root / "release_v1" / "UK_excluded_frames.csv",
        root / "release_v1" / f"UK_bad_frames_negative_{QC_VERSION}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing UK exclusion manifest. Looked in: {candidates}")


def validate_index(country: str) -> None:
    """Run the MLCast validator and checks specific to the UK and DE indexes."""
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"Unknown country: {country}")

    index_dir = Path(os.environ.get("MLCAST_INDEX_DIR", output_root() / "indexes"))
    index = index_dir / f"{country}_sampling_index.parquet"
    if not index.is_file():
        raise FileNotFoundError(index)

    command = [str(mlcast_executable()), "validate-sampling-index", str(index)]
    if country == "DE":
        # The raw Zarr contains 999, unlike the corrected index statistics.
        command.append("--no-data-checks")
    subprocess.run(command, check=True)

    if country not in ("UK", "DE"):
        print(f"PASS: {country} index validation completed")
        return

    report = json.loads(find_qc_report(country, index_dir).read_text())
    rows = pq.read_metadata(index).num_rows
    if report.get("qc_version") != QC_VERSION:
        raise ValueError(f"Unexpected {country} QC version")

    if country == "UK":
        if report["rows_remaining"] != rows:
            raise ValueError("UK QC report row count differs from the index")
        manifest = find_uk_manifest(index_dir)
        if sha256_file(manifest) != report["manifest_sha256"]:
            raise ValueError("UK manifest checksum mismatch")
        bad_frames = read_bad_frames(manifest)
        if len(bad_frames) != report["bad_frames"]:
            raise ValueError("UK manifest frame count mismatch")
        checked = 0
        for batch in pq.ParquetFile(index).iter_batches(
            batch_size=500_000, columns=["t"]
        ):
            starts = batch.column(0).to_numpy(zero_copy_only=False)
            if np.any(overlaps_bad_frame(starts, bad_frames, TIME_DEPTH)):
                raise ValueError("UK index includes a frame flagged by QC")
            checked += len(starts)
        if checked != rows:
            raise ValueError("UK Parquet row count mismatch")
        print(f"UK QC PASS: {rows:,} candidates exclude {len(bad_frames):,} frames")

    else:
        if report["candidate_rows_written"] != rows:
            raise ValueError("German QC report row count differs from the index")
        metadata = pq.read_schema(index).metadata or {}
        if metadata.get(b"mlcast_qc_index_statistics_units") != b"mm h-1":
            raise ValueError("German index statistics do not have mm/h metadata")

        # Check four source cubes directly; the separate stratified verifier
        # provides broader coverage before the public dataset release.
        sample = next(pq.ParquetFile(index).iter_batches(batch_size=4))
        source = Path(report["source_zarr"])
        with RadarSource(source, "rainfall_amount") as reader:
            for row in sample.to_pylist():
                t, x, y = (int(row[key]) for key in ("t", "x", "y"))
                cube = np.stack(
                    [
                        reader.frame(i)[y : y + HEIGHT, x : x + WIDTH]
                        for i in range(t, t + TIME_DEPTH)
                    ]
                ).astype(np.float32)
                valid = np.isfinite(cube) & (cube != DE_MISSING_VALUE)
                values = cube[valid].astype(np.float64) * DE_AMOUNT_TO_RATE
                expected = {
                    "nan_count": int(cube.size - valid.sum()),
                    "sum": float(values.sum()),
                    "mean": float(values.mean()),
                    "frac_wet": float(np.mean(values >= WET_THRESHOLD_MM_H)),
                }
                if expected["nan_count"] != row["nan_count"]:
                    raise ValueError(f"German missing count differs at {(t, x, y)}")
                for key in ("sum", "mean", "frac_wet"):
                    if not np.isclose(row[key], expected[key], rtol=1e-5, atol=1e-4):
                        raise ValueError(f"German {key} differs at {(t, x, y)}")
        print(f"DE QC PASS: {rows:,} rows, rate units, four source cubes checked")

    print(f"PASS: {country} index validation completed")


# ---------------------------------------------------------------------------
# Zenodo: upload existing indexes without publishing or replacing files.
# ---------------------------------------------------------------------------

def zenodo_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def zenodo_checksum(value: str) -> str:
    return str(value).lower().removeprefix('md5:')


def zenodo_request(session, base: str, method: str, url: str, **kwargs):
    """Keep credentials on the selected Zenodo host; do not follow redirects."""
    parsed = urlparse(url)
    if (parsed.scheme, parsed.netloc) != ('https', urlparse(base).netloc):
        raise ValueError('Unexpected Zenodo API URL')
    try:
        response = session.request(method, url, timeout=(30, 600),
                                   allow_redirects=False, **kwargs)
    except Exception as exc:
        # Avoid printing request headers or token-bearing URLs.
        raise RuntimeError(f'Zenodo {method} request failed ({type(exc).__name__}); '
                           'check the draft before retrying') from None
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f'Zenodo {method} returned HTTP {response.status_code}; '
                           'check token, permissions, quota and draft state')
    return response.json()


def zenodo_plan(paths: list[Path], remote_files: list[dict]) -> list[tuple[Path, str, str, int]]:
    remote = {item['filename']: item for item in remote_files}
    plan = []
    for path in paths:
        size = path.stat().st_size
        print(f'Checking {path.name} ({size:,} bytes)', flush=True)
        checksum = zenodo_md5(path)
        if path.name not in remote:
            action = 'UPLOAD'
        else:
            item = remote[path.name]
            action = ('SKIP' if int(item['filesize']) == size and
                      zenodo_checksum(item['checksum']) == checksum else 'CONFLICT')
        print(f'{action}: {path.name} md5:{checksum}', flush=True)
        plan.append((path, action, checksum, size))
    if any(action == 'CONFLICT' for _, action, _, _ in plan):
        raise RuntimeError('Existing remote file differs; no files uploaded. '
                           'Review the draft manually; never overwrite published files.')
    return plan


def upload_indexes(args: argparse.Namespace) -> None:
    """Plan or stream selected indexes into an explicitly selected draft."""
    import requests
    from urllib.parse import quote

    if args.zenodo_create_draft and not args.apply:
        raise ValueError('--zenodo-create-draft requires --apply (creates a remote draft)')
    if not args.zenodo_create_draft and args.zenodo_deposition_id is None:
        raise ValueError('Specify --zenodo-deposition-id or --zenodo-create-draft')
    if args.zenodo_deposition_id is not None and args.zenodo_deposition_id <= 0:
        raise ValueError('Deposition ID must be positive')
    base = ('https://zenodo.org' if args.zenodo_production else
            'https://sandbox.zenodo.org')
    token_name = ('ZENODO_ACCESS_TOKEN' if args.zenodo_production else
                  'ZENODO_SANDBOX_TOKEN')
    token = os.environ.get(token_name)
    if not token:
        raise EnvironmentError(f'Set {token_name} in your environment')
    index_dir = (Path(os.environ['MLCAST_INDEX_DIR']).expanduser() if
                 'MLCAST_INDEX_DIR' in os.environ else output_root() / 'indexes')
    files = [index_dir / f'{country}_sampling_index.parquet'
             for country in selected_countries(args.country)]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing or empty index: {path}')
        # Check the Parquet footer, not the full scientific validation.
        pq.read_metadata(path)
    api = base + '/api/deposit/depositions'
    with requests.Session() as session:
        session.headers.update({'Authorization': f'Bearer {token}',
                                'User-Agent': 'mlcast-sampling-indexes/0.2.0'})
        if args.zenodo_create_draft:
            deposit = zenodo_request(session, base, 'POST', api, json={})
            print(f"Created draft {deposit['id']} at {base}/uploads/{deposit['id']}",
                  flush=True)
        else:
            deposit = zenodo_request(session, base, 'GET',
                                     f'{api}/{args.zenodo_deposition_id}')
        dep_id = int(deposit['id'])
        if args.zenodo_deposition_id is not None and dep_id != args.zenodo_deposition_id:
            raise RuntimeError('Zenodo returned an unexpected deposition ID')
        print(f'Zenodo target: {base}, deposition {dep_id}', flush=True)
        # Published deposits can be inspected read-only, never written to.
        if args.apply and (deposit.get('submitted') is not False or
                           deposit.get('state') not in ('inprogress', 'unsubmitted')):
            raise RuntimeError('Target is not an editable unpublished draft. '
                               'Create a new version in Zenodo if needed.')
        remote_url = f'{api}/{dep_id}/files'
        remote = zenodo_request(session, base, 'GET', remote_url)
        plan = zenodo_plan(files, remote)
        if not args.apply:
            print('Read-only preview; no files uploaded.', flush=True)
            return
        bucket = deposit['links']['bucket'].rstrip('/')
        for path, action, checksum, size in plan:
            if action == 'SKIP':
                continue
            # Recheck just before writing; do not intentionally replace files.
            current = zenodo_request(session, base, 'GET', remote_url)
            if any(item['filename'] == path.name for item in current):
                raise RuntimeError(f'{path.name} appeared in the draft; rerun preview')
            if path.stat().st_size != size:
                raise RuntimeError(f'Local file changed: {path}')
            print(f'Uploading {path.name} ...', flush=True)
            with path.open('rb') as stream:
                uploaded = zenodo_request(session, base, 'PUT',
                    f'{bucket}/{quote(path.name, safe="")}', data=stream,
                    headers={'Content-Type': 'application/octet-stream'})
            if (uploaded.get('key') != path.name or int(uploaded['size']) != size or
                    zenodo_checksum(uploaded['checksum']) != checksum):
                raise RuntimeError(f'Upload verification failed: {path.name}; '
                                   'inspect the draft before retrying')
            print(f'Verified: {path.name}', flush=True)
        print(f'Finished. Review metadata and publish manually: {base}/uploads/{dep_id}',
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sources", "build", "validate", "upload"))
    parser.add_argument(
        "country", nargs="?", default="all",
        help="Country code (BE, FR, IT, DE, DK, UK) or 'all' (default)",
    )
    parser.add_argument(
        "--catalog-repo", default=os.environ.get("MLCAST_DATASETS_REPO"),
        help="Local Git clone of mlcast-community/mlcast-datasets",
    )
    parser.add_argument(
        "--catalog-ref", help="mlcast-datasets release tag or full commit SHA",
    )
    parser.add_argument(
        "--data-root", help="Directory containing local mirrors of catalog Zarrs",
    )
    parser.add_argument(
        "--source-override", action="append", default=[], metavar="COUNTRY=PATH",
        help="Override a catalog Zarr's local mirror path (repeatable)",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--zenodo-deposition-id", type=int,
                        help="Existing Zenodo deposition ID")
    target.add_argument("--zenodo-create-draft", action="store_true",
                        help="Create an empty draft (requires --apply)")
    parser.add_argument("--zenodo-production", action="store_true",
                        help="Use zenodo.org instead of sandbox.zenodo.org")
    parser.add_argument("--apply", action="store_true",
                        help="Perform uploads; default is read-only preview")
    args = parser.parse_args()
    if args.action == "upload":
        upload_indexes(args)
        return
    if (args.apply or args.zenodo_create_draft or args.zenodo_deposition_id is not None
            or args.zenodo_production):
        parser.error("Zenodo options require the upload action")

    if args.action in ("sources", "build"):
        configure_sources(args)
    if args.action == "sources":
        show_sources()
        return

    for country in selected_countries(args.country):
        if args.action == "build":
            build_country(country)
        else:
            validate_index(country)


if __name__ == "__main__":
    main()
