"""
This script is a command line tool to convert CNT files to FIF files using the MNE library.

Classes:
    CntToFif: A class to handle the conversion of CNT files to FIF files.

Usage:
    Run this script to convert all CNT files in the specified source directory to FIF files in the destination directory.

Example:
    python convert_to_fif.py source_dir dest_dir

    To view the help message with all available options:
    python convert_to_fif.py --help

Dependencies:
    - mne==1.6.1
    - pandas
    - tqdm

Author:
    Arjun Bingly
"""
# /// script
# dependencies = [
#     'mne==1.6.1',
#     'pandas',
#     'tqdm',
# ]
# ///
import argparse
import time
import threading
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from os import PathLike
from pathlib import Path
from typing import Optional

import mne
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", module='mne')

class CntToFif:
    """A class to handle the conversion of CNT files to FIF files.

    Attributes:
        source_dir (Path): The source directory containing CNT files.
        dest_dir (Path): The destination directory to save FIF files.
        overwrite (bool): Whether to overwrite existing FIF files.
        default is False.
    """

    def __init__(self, source_dir, dest_dir, overwrite=False, save_log=True, verbose=0):
        """Initializes the CntToFif class with the source and destination directories.

        Args:
            source_dir (str or Path): The source directory containing CNT files.
            dest_dir (str or Path): The destination directory to save FIF files.
            overwrite (bool): Whether to overwrite existing FIF files. Default is False.
            save_log (bool): Whether to save the conversion log.
                Saves to the destination directory with the name 'conversion_results.csv'.
                Default is True.
            verbose (int): Verbosity level. Default is 0.
        """
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.overwrite = overwrite
        self.save_log = save_log
        self.verbose = verbose

    @staticmethod
    def _convert_file(source_filepath, dest_filepath, overwrite=False, verbose=False) -> Optional[dict]:
        """Internal method to handle the conversion of CNT to FIF.

        Args:
            source_filepath (PathLike): The path to the CNT file.
            dest_filepath (PathLike): The path to save the FIF file.
            overwrite (bool): Whether to overwrite existing FIF files. Default is False.

        Returns:
            dict: Information about the conversion process, including any errors or warnings.
        """
        source_filepath = Path(source_filepath)
        dest_filepath = Path(dest_filepath)
        # print('TEST================================')
        if not source_filepath.is_file():
            raise ValueError(f'Path {source_filepath} is not a file')
        try:
            data = mne.io.read_raw_cnt(source_filepath, preload=False, verbose=verbose)
        except Exception as e:
            print(f'Error reading file {source_filepath}: {e}') if verbose else None
            return {
                'source_filename': str(source_filepath),
                'error': str(e)
            }

        info = {
            'source_filename': str(data.filenames[0]),
            'n_channels': int(data.info['nchan']),
            'n_samples': int(data.n_times),
            'freq': float(data.info['sfreq'])
        }

        if not overwrite and dest_filepath.exists():
            del data
            print(f'File {dest_filepath} already exists. Skipping.') if verbose else None
            info['warn'] = f'File {dest_filepath} already exists. Skipping.'
            return info

        try:
            data.save(dest_filepath, overwrite=overwrite)
        except Exception as e:
            print(f'Error saving file {dest_filepath}: {e}') if verbose else None
            info['error'] = str(e)
            del data
            return info

        info['dest_filepath'] = str(dest_filepath)
        print(f'Converted {source_filepath} to {dest_filepath}.') if verbose else None
        del data
        return info

    def convert_file(self, filepath: PathLike):
        """Converts a CNT file to a FIF file.

        Args:
            filepath (PathLike): The path to the CNT file.

        Returns:
            dict: The information extracted from the MNE Raw object.
        """
        filepath = Path(filepath)
        dest_filepath = self.dest_dir / filepath.relative_to(self.source_dir).with_suffix('.fif')
        dest_filepath.parent.mkdir(parents=True, exist_ok=True)
        info = self._convert_file(filepath, dest_filepath, overwrite=self.overwrite, verbose=self.verbose)
        return info

    @staticmethod
    def _process_results(results, save_path=None):
        """Processes the results of the conversion to get statistics and save to CSV.

        Args:
            results (list): The list of results from the conversion.
            save_path (str or Path, optional): The path to save the results as a CSV file. Default is None.
        """
        results = pd.DataFrame.from_records(results)
        total_files = len(results)
        num_error = 0
        if 'error' in results.columns:
            num_error = results['error'].count()
        num_skip = 0
        if 'warn' in results.columns:
            num_skip = results['warn'].count()
        print(f'Total Files: {total_files:,} (100%)')
        print(f'Errors: {num_error:,} ({num_error/total_files:.2%})')
        print(f'Skipped: {num_skip:,} ({num_skip/total_files:.2%})')
        print(f'Converted: {total_files - num_error - num_skip:,} ({(total_files - num_error - num_skip)/total_files:.2%})')
        if save_path is not None:
            results.to_csv(save_path, index=False)
            print(f'Saved conversion results to {save_path}.') if verbose else None

    def _convert_file_single_threaded(self):
        """Converts files using single-threading."""

        success_count = 0
        error_count = 0
        skip_count = 0
        results = []

        cnt_files = list(self.source_dir.glob('**/*.cnt'))
        with tqdm(total=len(cnt_files), desc="Converting files") as pbar:
            for file in cnt_files:
                result = self.convert_file(file)
                if result.get('error'):
                    error_count += 1
                elif result.get('warn'):
                    skip_count += 1
                else:
                    success_count += 1

                pbar.set_postfix(success=success_count, error=error_count, skip=skip_count)
                pbar.update(1)
                results.append(result)
        return results

    def _convert_file_multi_threaded(self):
        """Converts files using multi-threading."""
        success_count = 0
        error_count = 0
        skip_count = 0
        lock = threading.Lock()

        cnt_files = list(self.source_dir.glob('**/*.cnt'))
        with tqdm(total=len(cnt_files), desc="Converting files") as pbar:
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.convert_file, file): file for file in cnt_files}
                for future in futures:
                    result = future.result()
                    with lock:
                        if result.get('error'):
                            error_count += 1
                        elif result.get('warn'):
                            skip_count += 1
                        else:
                            success_count += 1

                        pbar.set_postfix(success=success_count, error=error_count, skip=skip_count)
                        pbar.update(1)
        return [future.result() for future in futures]


    def __call__(self, mode='multi', *args, **kwargs):
        """Converts all CNT files in the source directory to FIF files in the destination directory.

        Args:
            mode (str): The mode of operation, either 'multi' for multi-threaded or 'single' for single-threaded. Default is 'multi'.
        """
        start_time = time.time()

        if mode == 'multi':
            results = self._convert_file_multi_threaded()
        elif mode == 'single':
            results = self._convert_file_single_threaded()
        else:
            raise ValueError(f'Invalid mode {mode}. Choose either "multi" or "single".')

        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        print(f'Time taken: {formatted_time} seconds.') if self.verbose else None
        self._process_results(results, save_path=self.dest_dir/'conversion_results.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CNT files to FIF files using the MNE library.')
    parser.add_argument('source_dir', type=str, help='The source directory containing CNT files.')
    parser.add_argument('dest_dir', type=str, help='The destination directory to save FIF files.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing FIF files.')
    parser.add_argument('-m','--mode', type=str, choices=['multi', 'single'], default='multi', help='The mode of operation, either "multi" for multi-threaded or "single" for single-threaded. Default is "multi".')
    parser.add_argument('-v','--verbose', action='store_true', help='Whether to print verbose output.')

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    overwrite = args.overwrite
    mode = args.mode
    save_log = True
    verbose = args.verbose

    print(f'Converting CNT files from {source_dir} to FIF files in {dest_dir}.')
    converter = CntToFif(source_dir, dest_dir, overwrite=overwrite, save_log=save_log, verbose=verbose)
    converter(mode=mode)
