"""
pac_pipeline.py

This script computes Phase-Amplitude Coupling (PAC) for EEG data.
It allows the user to process a single file or all files in a specified directory.
The output can be saved as PAC matrices or visualized through plots.
The user can specify the method of PAC computation,
    as well as adjust various parameters related to data preprocessing and output options.

Example Usage:
    To run on a single file:
        python pac_pipeline.py -f ../Data/COGA/FZ/FZ_eec_1_a1_10024001_cnt_256.csv
    To run on all files in a directory:
        python pac_pipeline.py -d ../Data/COGA/FZ
    To change the output directory:
        python pac_pipeline.py -d ../Data/COGA/FZ -o ../Results

Arguments:
    -f, --filename: The filename of the EEG data.
    -d, --directory: The directory containing the EEG data, defaults to current directory.
    -o, --output_path: The output directory to save the results, defaults to '../Results'.
    --plot: If set, will generate and save plots of the PAC instead of saving the matrix.
    --method: The method to use for generating PAC, either 'split' or 'sliding_window'.

Functions:
    - get_info: Extracts relevant information from the filename.
    - remove_head_tail: Removes specified head and tail durations from the EEG data.
    - split_data: Splits the EEG data into epochs of a specified duration.
    - sliding_window: Generates sliding windows over phase and amplitude arrays.
    - gen_pac: Computes PAC values using the specified method.
    - gen_plots: Generates and saves plots of the PAC results.
    - save_pac: Saves the PAC results to CSV files.

Notes:
    - This script requires the 'tensorpac' library for PAC computation.
    - The script also utilizes 'numpy', 'matplotlib', and 'tqdm' for various functionalities.

Authors: Arjun Bingly, Christian Richard
Date: 20 Sept 2024
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorpac import Pac
from tqdm import tqdm

# TODO: Change user var names of params to CAPS

## Arguments
t_head = 30  # head to remove in secs
t_tail = 30  # tail to remove in secs

window_size = 10  # in secs
step_size = 3  # in secs

n_perm = 500  # number of surrogates to calculate

f_pha = [0.1, 13]  # frequency range phase for the coupling
f_amp = [4, 50]  # frequency range amplitude for the coupling
# channel = ['FC2']

pac_method = 1  # METHOD FOR PAC
surrogate_method = 2  # METHOD FOR COMPUTING SURROGATES
norm_method = 3  # METHOD FOR NORMALIZATION

random_state = 42  # random state for reproducibility, set to None for random


## Functions
def get_info(filename):
    """Extracts relevant information from the filename.

    Args:
        filename (Path): The path of the EEG data file.

    Returns:
        dict: A dictionary containing the filename, channel, vst, subject, and sample frequency.
    """
    _dict = {}
    _filename = filename.stem.split('_')
    _dict['filename'] = filename
    _dict['channel'] = _filename[0]
    _dict['vst'] = _filename[3]
    _dict['subject'] = _filename[4]
    _dict['sample_freq'] = int(_filename[-1])
    return _dict


def remove_head_tail(arr, sample_freq, t_head, t_tail, min_length=30):
    """Removes the specified head and tail from the EEG data array.

    Args:
        arr (np.ndarray): The EEG data array.
        sample_freq (int): The sampling frequency of the data.
        t_head (int): The duration in seconds to remove from the beginning.
        t_tail (int): The duration in seconds to remove from the end.
        min_length (int): The min duration in seconds of the resulting array.

    Returns:
        np.ndarray: The array after removing the head and tail.

    Raises:
        ValueError: If the resulting array length is too short, less than min_length.
    """
    head_index = sample_freq * t_head
    tail_index = len(arr) - (sample_freq * t_tail)
    if tail_index < sample_freq * min_length:
        raise ValueError(f"""The length of the array after removing the head and tail is too short. 
        It must be at least {min_length} secs long.""")
    return arr[head_index:tail_index]


def split_data(arr, sample_freq, epoch_dur):
    """Splits the EEG data into epochs of a specified duration.

    Args:
        arr (np.ndarray): The EEG data array.
        sample_freq (int): The sampling frequency of the data.
        epoch_dur (int): The duration of each epoch in seconds.

    Returns:
        list: A list of numpy arrays, each representing an epoch.
    """
    split_length = sample_freq * epoch_dur
    indices = np.arange(0, arr.shape[2], split_length)[1:]
    return np.array_split(arr, indices, axis=2)


def sliding_window(phases, amplitudes, window_size, step_size):
    """Generates a sliding window over the phase and amplitude arrays.

    Args:
        phases (np.ndarray): Array of phase values.
        amplitudes (np.ndarray): Array of amplitude values.
        window_size (int): Size of the sliding window.
        step_size (int): Step size for the sliding window.

    Yields:
        tuple: A tuple containing the current phase and amplitude windows.
    """
    for i in range(0, len(phases), step_size):
        yield phases[i:i + window_size], amplitudes[i:i + window_size]


# @njit(parallel=True)
def gen_pac(info, p, method='split'):
    """Generates Phase-Amplitude Coupling (PAC) values based on the specified method.

    Args:
        info (dict): Information dictionary containing file metadata.
        p (Pac): An instance of the Pac class used for filtering.
        method (str): The method for generating PAC ('split' or 'sliding_window').

    Returns:
        list: A list of PAC values for each epoch or window.

    Raises:
        ValueError: If an invalid method is specified.
    """
    data = np.loadtxt(info['filename'], delimiter=',', skiprows=1)
    sample_freq = info['sample_freq']

    data = remove_head_tail(data, sample_freq, t_head, t_tail)

    phases = p.filter(sample_freq, data, ftype='phase')
    amplitudes = p.filter(sample_freq, data, ftype='amplitude')

    pac_list = []
    if method == 'sliding_window':
        for phase, amp in zip(
                tqdm(sliding_window(phases, amplitudes, window_size, step_size), desc='Generating PAC', position=1,
                     leave=False), ):
            pac = p.fit(phase, amp, n_perm=n_perm, p=0.05, mcp='maxstat', random_state=random_state, verbose=False, )
            pac = pac.mean(-1)
            pac_list.append(pac)

    elif method == 'split':
        for phase, amp in zip(tqdm(phases, desc='Generating PAC', position=1, leave=False), amplitudes):
            pac = p.fit(phase, amp, n_perm=n_perm, p=0.05, mcp='maxstat', random_state=random_state, verbose=False, )
            pac = pac.mean(-1)
            pac_list.append(pac)
    else:
        raise ValueError('Invalid method. Method should be either "split" or "sliding_window"')
    return pac_list


def gen_plots(info, X, find_vmin_vmax=True):
    """Generates and saves plots of the PAC results.

    Args:
        info (dict): Information dictionary containing file metadata.
        X (list): A list of PAC values to plot.
        find_vmin_vmax (bool): If True, automatically find vmin and vmax for the plots else, use global vars.

    Returns:
        None
    """
    if find_vmin_vmax:
        vmin = np.floor(np.min(X))
        vmax = np.ceil(np.max(X))
    for i, x in enumerate(tqdm(X, desc='Saving plots', position=1, leave=False)):
        filename = info['filename']
        plot_name = f'{filename.name}_pac_{i}'
        plt.figure()
        p.comodulogram(x, title=plot_name, cmap='Reds', vmin=vmin, vmax=vmax, fz_labels=14, fz_title=18, fz_cblabel=14)
        plt.savefig(output_path / f'{plot_name}.png')
        plt.close()


def save_pac(info, X):
    """Saves the PAC results to CSV files.

    Args:
        info (dict): Information dictionary containing file metadata.
        X (list): A list of PAC values to save.

    Returns:
        None
    """
    for i, x in enumerate(tqdm(X, desc='Saving PAC', position=1, leave=False)):
        filename = info['filename']
        pac_name = f'{filename.name}_pac_{i}'
        # np.save(output_path/f'{pac_name}.npy', x)
        np.savetxt(output_path / f'{pac_name}.csv', x, delimiter=',')


if __name__ == "__main__":
    DEBUG = False

    if not DEBUG:
        parser = argparse.ArgumentParser(prog='PAC',
                                         description='This program for computing Phase-Amplitude Coupling (PAC)',
                                         epilog='HELP')  # TODO: Create Help

        parser.add_argument('-f', '--filename', type=str)
        parser.add_argument('-d', '--directory', default='.', type=str)
        parser.add_argument('-o', '--output_path', default='../Results', type=str)
        parser.add_argument('--plot', action='store_true')
        parser.add_argument('--method', type=str, default='split')

        args = parser.parse_args()

        if args.filename:
            print('Filename:', args.filename)
            filenames = [Path(args.filename)]
        elif args.directory:
            print('Directory:', args.directory)
            filenames = list(Path(args.directory).glob('*.csv'))
            print('Total Files:', len(filenames))
        else:
            raise ValueError('No filename or directory provided')

        output_path = Path(args.output_path)
        print('Output Path:', output_path)
    else:
        # Debug case
        print('NOT USING PARSER')
        filenames = [Path('../Data/COGA/FZ/FZ_eec_1_a1_10024001_cnt_256.csv')]
        output_path = Path('../Results')

    output_path.mkdir(parents=True, exist_ok=True)

    p = Pac(idpac=(pac_method, surrogate_method, norm_method), f_pha=(f_pha[0], f_pha[1], 1, 0.1),
            f_amp=(f_amp[0], f_amp[1], 1, 0.1), verbose=None)

    pbar = tqdm(filenames, desc='Processing Files', position=0)
    for file in pbar:
        pbar.set_postfix_str(f'File: {file.name}')
        info = get_info(file)
        X = gen_pac(info, p, method=args.method)
        if args.plot:
            gen_plots(info, X)
        else:
            save_pac(info, X)

# With MVL Total Processing Files: 5it [08:11, 98.38s/it]
# With MI Total Processing Files: 5it [3:25:18<00:00, 2463.63s/it]
