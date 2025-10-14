#!/usr/bin/env python3
"""
Standalone script to aggregate PAC chunk CSV files into a single consolidated CSV.

This script scans a directory for chunk files, merges them, removes duplicates,
and saves the final aggregated results.

Usage:
    python aggregate_pac_chunks.py <chunks_dir> [output_file]

    Or run interactively within the directory structure.

Example:
    python aggregate_pac_chunks.py /path/to/pac_results_1_stc_saccade_3-8_40-140_0.15-0.4_..._chunks/
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_pac_results(chunks_dir, output_fname=None):
    """
    Aggregate all chunk CSV files into a single consolidated CSV.

    This function reads all chunk files from the chunks directory,
    removes duplicates (keeping the first occurrence), and saves
    the aggregated results to the output file.

    Parameters:
    -----------
    chunks_dir : str
        Directory containing chunk_*.csv files
    output_fname : str, optional
        Path to save the aggregated CSV file. If None, saves to parent directory
        with same base name as chunks directory.
    """
    chunks_dir = Path(chunks_dir)

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    print(f"{'='*70}")
    print(f"PAC Chunk Aggregation")
    print(f"{'='*70}")
    print(f"Chunks directory: {chunks_dir}")

    # Find all chunk files
    chunk_files = sorted(chunks_dir.glob("chunk_*.csv"))

    if not chunk_files:
        print(f"\nERROR: No chunk files found in {chunks_dir}")
        print(f"Looking for files matching pattern: chunk_*.csv")
        return None

    print(f"\nFound {len(chunk_files)} chunk files:")
    for cf in chunk_files[:5]:  # Show first 5
        print(f"  - {cf.name}")
    if len(chunk_files) > 5:
        print(f"  ... and {len(chunk_files) - 5} more")

    # Read all chunk files
    all_chunks = []
    failed_files = []

    print(f"\nReading chunk files...")
    for chunk_file in chunk_files:
        try:
            chunk_df = pd.read_csv(chunk_file, index_col=0)
            all_chunks.append(chunk_df)
            print(f"  ✓ {chunk_file.name}: {len(chunk_df)} rows")
        except Exception as e:
            failed_files.append((chunk_file.name, str(e)))
            print(f"  ✗ {chunk_file.name}: ERROR - {e}")

    if failed_files:
        print(f"\nWarning: Failed to read {len(failed_files)} files:")
        for fname, error in failed_files:
            print(f"  - {fname}: {error}")

    if not all_chunks:
        print("\nERROR: No valid chunk files could be read")
        return None

    # Concatenate all chunks
    print(f"\n{'='*70}")
    print("Merging chunks...")
    aggregated_df = pd.concat(all_chunks, ignore_index=True)
    print(f"Total rows before deduplication: {len(aggregated_df)}")

    # Check columns
    print(f"\nColumns found: {list(aggregated_df.columns)}")

    # Remove duplicates based on channel and split_group (if present)
    if 'split_group' in aggregated_df.columns:
        print("\nRemoving duplicates based on (channel, split_group)...")
        n_before = len(aggregated_df)
        aggregated_df = aggregated_df.drop_duplicates(subset=['channel', 'split_group'], keep='first')
        n_after = len(aggregated_df)
        print(f"  Removed {n_before - n_after} duplicate entries")
    else:
        print("\nRemoving duplicates based on (channel)...")
        n_before = len(aggregated_df)
        aggregated_df = aggregated_df.drop_duplicates(subset=['channel'], keep='first')
        n_after = len(aggregated_df)
        print(f"  Removed {n_before - n_after} duplicate entries")

    print(f"\nTotal rows after deduplication: {len(aggregated_df)}")

    # Sort by channel for better readability
    aggregated_df = aggregated_df.sort_values('channel').reset_index(drop=True)

    # Determine output filename if not provided
    if output_fname is None:
        # Remove "_chunks" suffix from directory name
        base_name = chunks_dir.name.replace("_chunks", "")
        output_fname = chunks_dir.parent / f"{base_name}.csv"
    else:
        output_fname = Path(output_fname)

    # Save aggregated results
    print(f"\n{'='*70}")
    print(f"Saving aggregated results...")
    print(f"Output file: {output_fname}")
    aggregated_df.to_csv(output_fname)
    print(f"✓ Successfully saved {len(aggregated_df)} rows")

    # Display summary statistics
    print(f"\n{'='*70}")
    print("Aggregated Results Summary:")
    print(f"{'='*70}")

    if 'split_group' in aggregated_df.columns:
        print("\nBy split group:")
        summary = aggregated_df.groupby('split_group').agg({
            'pac': ['count', 'mean', 'std', 'min', 'max'],
            'n_epochs': 'first'
        })
        print(summary)

    print(f"\nOverall PAC statistics:")
    print(aggregated_df['pac'].describe())

    # Count significant channels
    if 'pac' in aggregated_df.columns:
        n_sig_196 = np.sum(aggregated_df["pac"] > 1.96)
        n_sig_165 = np.sum(aggregated_df["pac"] > 1.65)
        print(f"\nSignificance summary:")
        print(f"  z > 1.96 (p < 0.05): {n_sig_196}/{len(aggregated_df)} ({100*n_sig_196/len(aggregated_df):.1f}%)")
        print(f"  z > 1.65 (p < 0.10): {n_sig_165}/{len(aggregated_df)} ({100*n_sig_165/len(aggregated_df):.1f}%)")

    print(f"\n{'='*70}")
    print(f"✓ Aggregation complete!")
    print(f"{'='*70}\n")

    return aggregated_df


def find_chunk_directories(base_dir, pattern="*_chunks"):
    """
    Find all chunk directories matching a pattern.

    Parameters:
    -----------
    base_dir : str
        Base directory to search
    pattern : str
        Glob pattern for chunk directories

    Returns:
    --------
    list of Path objects
    """
    base_dir = Path(base_dir)
    chunk_dirs = sorted(base_dir.glob(pattern))
    return [d for d in chunk_dirs if d.is_dir()]


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate PAC chunk CSV files into a single consolidated file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate specific chunks directory
  python aggregate_pac_chunks.py /path/to/results/pac_results_1_stc_saccade_..._chunks/

  # Specify custom output filename
  python aggregate_pac_chunks.py /path/to/chunks/ /path/to/output.csv

  # Aggregate all chunk directories in current directory
  python aggregate_pac_chunks.py --all .
        """
    )

    parser.add_argument(
        "chunks_dir",
        nargs="?",
        default=".",
        help="Path to chunks directory (default: current directory)"
    )

    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Output CSV filename (default: auto-generated from chunks dir name)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Aggregate all *_chunks directories in the specified path"
    )

    args = parser.parse_args()

    if args.all:
        # Find and aggregate all chunk directories
        base_dir = Path(args.chunks_dir)
        chunk_dirs = find_chunk_directories(base_dir)

        if not chunk_dirs:
            print(f"No *_chunks directories found in {base_dir}")
            sys.exit(1)

        print(f"Found {len(chunk_dirs)} chunk directories to aggregate:\n")
        for chunk_dir in chunk_dirs:
            print(f"  - {chunk_dir.name}")

        print("\n")

        # Process each directory
        for i, chunk_dir in enumerate(chunk_dirs, 1):
            print(f"\n{'#'*70}")
            print(f"Processing {i}/{len(chunk_dirs)}: {chunk_dir.name}")
            print(f"{'#'*70}\n")

            try:
                aggregate_pac_results(chunk_dir)
            except Exception as e:
                print(f"ERROR processing {chunk_dir}: {e}")
                continue
    else:
        # Single directory mode
        chunks_dir = Path(args.chunks_dir)

        # If provided path is not a chunks directory, look for chunks subdirectories
        if not chunks_dir.name.endswith("_chunks"):
            chunk_dirs = find_chunk_directories(chunks_dir)

            if len(chunk_dirs) == 0:
                print(f"ERROR: No *_chunks directories found in {chunks_dir}")
                print(f"\nTip: Provide the full path to a chunks directory, or use --all flag")
                sys.exit(1)
            elif len(chunk_dirs) == 1:
                print(f"Found chunks directory: {chunk_dirs[0].name}")
                chunks_dir = chunk_dirs[0]
            else:
                print(f"Found multiple chunk directories in {chunks_dir}:")
                for i, d in enumerate(chunk_dirs, 1):
                    print(f"  {i}. {d.name}")
                print(f"\nPlease specify which one to aggregate, or use --all flag")
                sys.exit(1)

        # Aggregate
        try:
            aggregate_pac_results(chunks_dir, args.output_file)
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
