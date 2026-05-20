#!/usr/bin/env python3
"""
Convert a CSV to Snappy-compressed Parquet (PyArrow or fastparquet).

Handles multi-line quoted fields (e.g. JSON in GENRES) via PyArrow
``newlines_in_values`` or pandas ``engine='python'``.

  python scripts/csv_to_parquet.py \\
    --input /Users/ronannayak/Downloads/current_singles_streams.csv \\
    --output model/data/current_singles_streams.parquet

Requires one of: pyarrow (recommended for read+write), fastparquet (+ pandas for CSV read).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _have_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _have_fastparquet() -> bool:
    try:
        import fastparquet  # noqa: F401

        return True
    except ImportError:
        return False


def _pandas_read_csv(src: Path, *, encoding: str, chunksize: int | None = None):
    """pandas engine=python does not support low_memory."""
    import pandas as pd

    kwargs = {"filepath_or_buffer": src, "encoding": encoding, "engine": "python"}
    if chunksize is not None:
        return pd.read_csv(chunksize=chunksize, **kwargs)
    return pd.read_csv(**kwargs)


def resolve_output_path(src: Path, output: Path) -> Path:
    """
    --output may be a directory; in that case write ``{input_stem}.parquet`` inside it.
    """
    out = output.expanduser().resolve()
    if out.is_dir():
        return out / f"{src.stem}.parquet"
    if out.suffix.lower() != ".parquet":
        return out.with_suffix(".parquet")
    return out


def _resolve_engine(requested: str) -> str:
    if requested == "auto":
        if _have_pyarrow():
            return "pyarrow"
        if _have_fastparquet():
            return "fastparquet"
        raise SystemExit(
            "No parquet engine installed. Run: pip install pyarrow  (or: pip install fastparquet)"
        )
    if requested == "pyarrow" and not _have_pyarrow():
        raise SystemExit("pyarrow is not installed. Run: pip install pyarrow")
    if requested == "fastparquet" and not _have_fastparquet():
        raise SystemExit("fastparquet is not installed. Run: pip install fastparquet")
    return requested


def read_csv_pyarrow(src: Path, *, encoding: str):
    import pyarrow.csv as pacsv

    read_options = pacsv.ReadOptions(encoding=encoding)
    parse_options = pacsv.ParseOptions(newlines_in_values=True)
    logger.info("Reading CSV with PyArrow (newlines_in_values=True)...")
    return pacsv.read_csv(
        str(src),
        read_options=read_options,
        parse_options=parse_options,
    )


def write_parquet_pyarrow(table, dst: Path, *, compression: str) -> int:
    import pyarrow.parquet as pq

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression=compression)
    return table.num_rows


def convert_pyarrow(
    src: Path,
    dst: Path,
    *,
    compression: str,
    encoding: str,
) -> int:
    table = read_csv_pyarrow(src, encoding=encoding)
    return write_parquet_pyarrow(table, dst, compression=compression)


def convert_fastparquet(
    src: Path,
    dst: Path,
    *,
    compression: str,
    encoding: str,
    chunksize: int | None,
) -> int:
    import pandas as pd

    dst.parent.mkdir(parents=True, exist_ok=True)

    if chunksize is None:
        logger.info("Reading CSV with pandas (engine=python)...")
        df = _pandas_read_csv(src, encoding=encoding)
        logger.info("Writing parquet with fastparquet (compression=%s)...", compression)
        df.to_parquet(dst, engine="fastparquet", compression=compression, index=False)
        return len(df)

    logger.info("Reading CSV in chunks of %d rows...", chunksize)
    total = 0
    first = True
    for i, chunk in enumerate(_pandas_read_csv(src, encoding=encoding, chunksize=chunksize)):
        total += len(chunk)
        if first:
            logger.info("Writing parquet with fastparquet (compression=%s)...", compression)
            chunk.to_parquet(
                dst,
                engine="fastparquet",
                compression=compression,
                index=False,
            )
            first = False
        else:
            chunk.to_parquet(
                dst,
                engine="fastparquet",
                compression=compression,
                index=False,
                append=True,
            )
        if (i + 1) % 10 == 0:
            logger.info("  ... %d chunks, %d rows so far", i + 1, total)
    return total


def convert_pandas_pyarrow_write(
    src: Path,
    dst: Path,
    *,
    compression: str,
    encoding: str,
) -> int:
    """Fallback when pyarrow.csv fails but pyarrow.parquet is available."""
    import pandas as pd

    logger.info("Reading CSV with pandas (engine=python)...")
    df = _pandas_read_csv(src, encoding=encoding)
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing parquet with pyarrow (compression=%s)...", compression)
    df.to_parquet(dst, engine="pyarrow", compression=compression, index=False)
    return len(df)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert CSV to Snappy-compressed Parquet.")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Source CSV path",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Destination .parquet file or directory (directory → {input_stem}.parquet)",
    )
    p.add_argument(
        "--engine",
        choices=("auto", "pyarrow", "fastparquet"),
        default="auto",
        help="Parquet writer (default: auto → pyarrow if installed, else fastparquet)",
    )
    p.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec (default: snappy)",
    )
    p.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV text encoding (default: utf-8-sig, strips BOM)",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Row chunk size for fastparquet append mode (optional; high RAM if unset)",
    )
    p.add_argument(
        "--pandas-read",
        action="store_true",
        help="Read CSV with pandas even when engine=pyarrow (write still uses pyarrow)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    src = args.input.expanduser().resolve()
    if not src.is_file():
        logger.error("Input not found: %s", src)
        return 1

    dst = resolve_output_path(src, args.output)
    if dst != args.output.expanduser().resolve():
        logger.info("Resolved output path: %s", dst)

    engine = _resolve_engine(args.engine)
    t0 = time.perf_counter()

    try:
        if engine == "pyarrow" and not args.pandas_read:
            try:
                table = read_csv_pyarrow(src, encoding=args.encoding)
            except Exception as e:
                logger.warning("PyArrow CSV read failed (%s); retrying via pandas", e)
                nrows = convert_pandas_pyarrow_write(
                    src, dst, compression=args.compression, encoding=args.encoding
                )
            else:
                nrows = write_parquet_pyarrow(
                    table, dst, compression=args.compression
                )
        elif engine == "pyarrow" and args.pandas_read:
            nrows = convert_pandas_pyarrow_write(
                src, dst, compression=args.compression, encoding=args.encoding
            )
        else:
            nrows = convert_fastparquet(
                src,
                dst,
                compression=args.compression,
                encoding=args.encoding,
                chunksize=args.chunksize,
            )
    except Exception:
        logger.exception("Conversion failed")
        return 1

    elapsed = time.perf_counter() - t0
    size_mb = dst.stat().st_size / (1024 * 1024)
    logger.info(
        "Wrote %s (%d rows, %.1f MB, %.1fs, engine=%s, compression=%s)",
        dst,
        nrows,
        size_mb,
        elapsed,
        engine,
        args.compression,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
