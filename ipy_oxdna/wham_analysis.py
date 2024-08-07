import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import shutil
import fileinput
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor


def write_com_files(base_umbrella):
    sim_dir = base_umbrella.production_sim_dir
    all_observables = base_umbrella.analysis.read_all_observables('prod')
    sim_dirs = [sim.sim_dir for sim in base_umbrella.production_sims]
    for df, sim_dir in zip(all_observables, sim_dirs):
        com_distance = df['com_distance']
        with open(os.path.join(sim_dir, 'com_distance.txt'), 'w') as f:
            f.write('\n'.join(map(str, com_distance)))
    
    return None


def copy_single_file(sim_dir, window, file, com_dir):
    shutil.copyfile(os.path.join(sim_dir, window, file),
                    os.path.join(com_dir, f'com_distance_{window}.txt'))


def unpack_and_run(args):
    return copy_single_file(*args)


def copy_com_files(sim_dir, com_dir):
    # Remove existing com_dir and create a new one
    if os.path.exists(com_dir):
        shutil.rmtree(com_dir)
    os.mkdir(com_dir)

    windows = [w for w in os.listdir(sim_dir) if w.isdigit()]

    tasks = []
    for window in windows:
        if os.path.isdir(os.path.join(sim_dir, window)):
            for file in os.listdir(os.path.join(sim_dir, window)):
                if 'com_distance' in file:
                    tasks.append((sim_dir, window, file, com_dir))

    with ProcessPoolExecutor() as executor:
        executor.map(unpack_and_run, tasks)
    
    return None


def copy_h_bond_files(sim_dir, com_dir):
    # copy com files from each to window to separate directory
    if os.path.exists(f'{com_dir}/h_bonds'):
        shutil.rmtree(f'{com_dir}/h_bonds')
    if not os.path.exists(f'{com_dir}/h_bonds'):
        os.mkdir(f'{com_dir}/h_bonds')
    windows = [w for w in os.listdir(sim_dir) if w.isdigit()]
    for window in windows:
        if os.path.isdir(os.path.join(sim_dir, window)):
            for file in os.listdir(os.path.join(sim_dir, window)):
                if 'hb_observable' in file:
                    shutil.copyfile(os.path.join(sim_dir, window, file),
                                    os.path.join(f'{com_dir}/h_bonds', f'hb_list_{window}.txt'))
    return None
    
    
def collect_coms(com_dir):
    # create a list of dataframes for each window
    com_list = []
    com_files = [f for f in os.listdir(com_dir) if f.endswith('.txt')]
    com_files.sort(key=sort_coms)
    for window, file in enumerate(com_files):
        com_list.append(pd.read_csv(os.path.join(com_dir, file), header=None, names=[window], usecols=[0]))
    return com_list


def autocorrelation(com_list):
    # create a list of autocorrelation values for each window
    autocorrelation_list = []
    for com in com_list:
        de = sm.tsa.acf(com, nlags=500000)
        low = next(x[0] for x in enumerate(list(de)) if abs(x[1]) < (1 / np.e))
        if int(low) == 1:
            low = 2
        autocorrelation_list.append(low)
    return autocorrelation_list


def process_file(filename):
    with open(filename, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for i, line in enumerate(lines, start=1):
            f.write(f'{i} {line}')


def number_com_lines(com_dir):
    # Create time_series directory and add com files with line number
    if not os.path.exists(com_dir):
        print('com_dir does not exist')
        return None
    
    os.chdir(com_dir)
    files = [os.path.join(com_dir, filename) for filename in os.listdir(com_dir) if os.path.isfile(os.path.join(com_dir, filename))]
    files.sort(key=sort_coms)
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, files)

    return com_dir


def sort_coms(file):
    # Method to sort com files by window number
    var = int(file.split('_')[-1].split('.')[0])
    return var


def get_r0_list(xmin, xmax, sim_dir):
    # Method to get r0 list
    n_confs = len(os.listdir(sim_dir))
    r0_list = np.round(np.linspace(float(xmin), float(xmax), n_confs)[1:], 3)
    return r0_list


def create_metadata(time_dir, autocorrelation_list, r0_list, k):
    # Create metadata file to run WHAM analysis with
    os.chdir(time_dir)
    com_files = [file for file in os.listdir(os.getcwd()) if 'com_distance' in file]
    com_files.sort(key=sort_coms)
    with open(os.path.join(os.getcwd(), 'metadata'), 'w') as f:
        if autocorrelation_list is None:
            for file, r0 in zip(com_files, r0_list):
                f.write(f'{file} {r0} {k}\n')
        else:
            for file, r0, auto in zip(com_files, r0_list, autocorrelation_list):
                f.write(f'{file} {r0} {k} {auto}\n')
    return None


def run_wham(wham_dir, time_dir, xmin, xmax, n_bins, tol, n_boot, temp):
    # Run WHAM analysis on metadata file to create free energy file
    wham = os.path.join(wham_dir, 'wham')
    seed = str(np.random.randint(0, 1000000))
    os.chdir(time_dir)
    output = subprocess.run([wham, xmin, xmax, n_bins, tol, temp, '0', 'metadata', 'freefile', n_boot, seed], capture_output=True)
    return output


def format_freefile(time_dir):
    # Format free file to be readable by pandas
    os.chdir(time_dir)
    with open("freefile", "r") as f:
        lines = f.readlines()
    lines[0] = "#Coor\tFree\t+/-\tProb\t+/-\n"
    with open("freefile", "w") as f:
        for line in lines:
            f.write(line)
    return None


def wham_analysis(wham_dir, sim_dir, com_dir, xmin, xmax, k, n_bins, tol, n_boot, temp):
    print('Running WHAM analysis...')
    copy_com_files(sim_dir, com_dir)
    if int(n_boot) > 0:
        com_list = collect_coms(com_dir)
        autocorrelation_list = autocorrelation(com_list)
    else:
        autocorrelation_list = None
    time_dir = number_com_lines(com_dir)
    r0_list = get_r0_list(xmin, xmax, sim_dir)
    create_metadata(time_dir, autocorrelation_list, r0_list, k)
    output = run_wham(wham_dir, time_dir, xmin, xmax, n_bins, tol, n_boot, temp)
    # print(output)
    format_freefile(time_dir)
    print('WHAM analysis completed')
    return output


def chunk_com_file(chunk_lower_bound, chunk_upper_bound, com_dir):
    com_files = [os.path.join(com_dir, f) for f in os.listdir(com_dir) if f.endswith('.txt')]
    com_files.sort(key=sort_coms)
    for file in com_files:
        for line in fileinput.input(file, inplace=True):
            if (int(fileinput.filelineno()) >= chunk_lower_bound) and (int(fileinput.filelineno()) <= chunk_upper_bound):
                sys.stdout.write(f'{line}')
    return None


def chunked_wham_analysis(chunk_lower_bound, chunk_upper_bound, wham_dir, sim_dir, com_dir, xmin, xmax, k, n_bins, tol, n_boot, temp):
    print('Running WHAM analysis...')
    copy_com_files(sim_dir, com_dir)
    chunk_com_file(chunk_lower_bound, chunk_upper_bound, com_dir)
    com_list = collect_coms(com_dir)
    autocorrelation_list = autocorrelation(com_list)
    time_dir = number_com_lines(com_dir)
    r0_list = get_r0_list(xmin, xmax, sim_dir)
    create_metadata(time_dir, autocorrelation_list, r0_list, k)
    output = run_wham(wham_dir, time_dir, xmin, xmax, n_bins, tol, n_boot, temp)
    #print(output)
    format_freefile(time_dir)
    print('WHAM analysis completed')
    return output
    
    
def copy_pos_files(sim_dir, pos_dir):
    # copy com files from each to window to separate directory
    if os.path.exists(pos_dir):
        shutil.rmtree(pos_dir)
    if not os.path.exists(pos_dir):
        os.mkdir(pos_dir)
    windows = [w for w in os.listdir(sim_dir) if w.isdigit()]
    for window in windows:
        if os.path.isdir(os.path.join(sim_dir, window)):
            for file in os.listdir(os.path.join(sim_dir, window)):
                if 'cms_positions' in file:
                    shutil.copyfile(os.path.join(sim_dir, window, file),
                                    os.path.join(pos_dir, f'cms_positions_{window}.txt'))


def collect_pos(pos_dir):
    # create a list of dataframes for each window
    com_list = []
    com_files = [f for f in os.listdir(pos_dir) if f.endswith('.txt')]
    for window, file in enumerate(com_files):
        com_list.append(pd.read_csv(os.path.join(pos_dir, file), header=None, names=[window], usecols=[0]))
    return com_list


def copy_com_pos(sim_dir, com_dir, pos_dir):
    copy_com_files(sim_dir, com_dir)
    copy_pos_files(sim_dir, pos_dir)
    return None


def get_xmax(com_dir_1, com_dir_2):
    com_list_1 = collect_coms(com_dir_1)
    com_list_2 = collect_coms(com_dir_2)
    xmax_1 = max([com.max().iloc[0] for com in com_list_1])
    xmax_2 = max([com.max().iloc[0] for com in com_list_2])
    xmax = max(xmax_1, xmax_2)
    return xmax