from ipy_oxdna.oxdna_simulation import Simulation, Force, Observable
from ipy_oxdna.wham_analysis import *
import multiprocessing as mp
import os
from os.path import join, exists
import numpy as np
import shutil
import pandas as pd
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
from json import load, dump
import traceback
import warnings
import pyarrow
import warnings


class BaseUmbrellaSampling:
    def __init__(self, file_dir, system, clean_build=False):
        self.clean_build = clean_build
        self.system = system
        self.file_dir = file_dir
        self.system_dir = join(self.file_dir, self.system)
        if not exists(self.system_dir):
            os.mkdir(self.system_dir)  
        
        self.windows = UmbrellaWindow(self)
        self.us_build = UmbrellaBuild(self)
        self.analysis = UmbrellaAnalysis(self)
        self.progress = UmbrellaProgress(self)
        self.info_utils = UmbrellaInfoUtils(self)
        self.observables = UmbrellaObservables(self)
        
        self.wham = WhamAnalysis(self)
        
        self.f = Force()
        self.obs = Observable()
        
        self.umbrella_bias = None
        self.com_by_window = None
        self.r0 = None
        
        self.read_progress()
        
    
    def queue_sims(self, simulation_manager, sim_list, continue_run=False):
        for sim in sim_list:
            simulation_manager.queue_sim(sim, continue_run=continue_run)        
     
             
    def wham_run(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot, all_observables=False):
        """
        Run the weighted histogram analysis technique (Grossfield, Alan http://membrane.urmc.rochester.edu/?page_id=126)
        
        Parameters:
            wham_dir (str): directory in which wham is complied using the install_wham.sh script.
            
            xmin (str): smallest value of the order parameter to be sampled.
            
            xmax (str): largest value of the order parameter to be sample.
            
            umbrella_stiff (str): stiffness of the umbrella potential.
            
            n_bins (str): number of bins of the resultant free energy profile.
            
            tol (str): the tolerance of convergence for the WHAM technique.
            
            n_boot (str): number of monte carlo bootstrapping steps to take.
        """
        self.wham.wham_dir = wham_dir
        self.wham.xmin = xmin
        self.wham.xmax = xmax
        self.wham.umbrella_stiff = umbrella_stiff
        self.wham.n_bins = n_bins
        self.wham.tol = tol
        self.wham.n_boot = n_boot
        
        if all_observables is True:
            write_com_files(self)
        self.wham.run_wham(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
        self.free = self.wham.to_si(n_bins, self.com_dir)
        self.mean = self.wham.w_mean(self.free)
        try:
            self.standard_error, self.confidence_interval = self.wham.bootstrap_w_mean_error(self.free)
        except:
            self.standard_error = 'failed'
                                    
    
    def read_progress(self):
        self.progress.read_pre_equlibration_progress()
        self.progress.read_equlibration_progress()
        self.progress.read_production_progress()
        self.progress.read_wham_progress()
        self.progress.read_convergence_analysis_progress()

class ComUmbrellaSampling(BaseUmbrellaSampling):
    def __init__(self, file_dir, system, clean_build=False):
        super().__init__(file_dir, system, clean_build=clean_build)

     
    def build_pre_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters, starting_r0, steps, observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False, protein=None, force_file=None):
        self.observables_list = []
        self.windows.pre_equlibration_windows(n_windows)
        self.rate_umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps)
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=False)
        if continue_run is False:
            self.us_build.build(self.pre_equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
        self.queue_sims(simulation_manager, self.pre_equlibration_sims, continue_run=continue_run)
    
    
    def build_equlibration_runs(self, simulation_manager,  n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                                observable=False, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                                protein=None, force_file=None):
        """
        Build the equlibration run
        
        Parameters:
            simulation_manager (SimulationManager): pass a simulation manager.
            
            n_windows (int): number of simulation windows.
            
            com_list (str): comma speperated list of nucleotide indexes.
            
            ref_list (str): comma speperated list of nucleotide indexex.
            
            stiff (float): stiffness of the umbrella potential.
            
            xmin (float): smallest value of the order parameter to be sampled.
            
            xmax (float): largest value of the order parameter to be sampled.
            
            input_parameters (dict): dictonary of oxDNA parameters.
            
            observable (bool): Boolean to determine if you want to print the observable for the equlibration simulations.
            
            sequence_dependant (bool): Boolean to use sequence dependant parameters.
            
            print_every (float): how often to print the order parameter.
            
            name (str): depreciated.
            
            continue_run (float): number of steps to continue umbrella sampling (i.e 1e7).
            
            protein (bool): Use a protein par file and run with the ANM oxDNA interaction potentials.
            
            force_file (bool): Add an external force to the umbrella simulations.
        """
        self.observables_list = []
        self.n_windows = n_windows
        self.windows.equlibration_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=False)
        if continue_run is False:
            self.us_build.build(self.equlibration_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file)
        self.queue_sims(simulation_manager, self.equlibration_sims, continue_run=continue_run)
        
        
    def build_production_runs(self, simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, input_parameters,
                              observable=True, sequence_dependant=False, print_every=1e4, name='com_distance.txt', continue_run=False,
                              protein=None, force_file=None):
        """
        Build the production run
        
        Parameters:
            simulation_manager (SimulationManager): pass a simulation manager.
            
            n_windows (int): number of simulation windows.
            
            com_list (str): comma speperated list of nucleotide indexes.
            
            ref_list (str): comma speperated list of nucleotide indexex.
            
            stiff (float): stiffness of the umbrella potential.
            
            xmin (float): smallest value of the order parameter to be sampled.
            
            xmax (float): largest value of the order parameter to be sampled.
            
            input_parameters (dict): dictonary of oxDNA parameters.
            
            observable (bool): Boolean to determine if you want to print the observable for the equlibration simulations, make sure to not have this as false for production run.
            
            sequence_dependant (bool): Boolean to use sequence dependant parameters.
            
            print_every (float): how often to print the order parameter.
            
            name (str): depreciated.
            
            continue_run (float): number of steps to continue umbrella sampling (i.e 1e7).
            
            protein (bool): Use a protein par file and run with the ANM oxDNA interaction potentials.
            
            force_file (bool): Add an external force to the umbrella simulations.
        """
        self.observables_list = []
        self.windows.equlibration_windows(n_windows)
        self.windows.production_windows(n_windows)
        self.umbrella_forces(com_list, ref_list, stiff, xmin, xmax, n_windows)
        if observable:
            self.initialize_observables(com_list, ref_list, print_every, name, force_energy_split=False)
        if continue_run is False:
            self.us_build.build(self.production_sims, input_parameters,
                                self.forces_list, self.observables_list,
                                observable=observable, sequence_dependant=sequence_dependant, protein=protein, force_file=force_file) 
        
        self.queue_sims(simulation_manager, self.production_sims, continue_run=continue_run)         
 

    def umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows):
        """ Build Umbrella potentials"""
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        umbrella_forces_1 = []
        umbrella_forces_2 = []
        
        for x_val in x_range:   
            self.umbrella_force_1 = self.f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                         
                rate='0',
            )        
            umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{x_val}',                       
                PBC='1',                          
                rate='0',
            )
            umbrella_forces_2.append(self.umbrella_force_2)  
        self.forces_list = np.transpose(np.array([umbrella_forces_1, umbrella_forces_2]))     


    def rate_umbrella_forces(self, com_list, ref_list, stiff, xmin, xmax, n_windows, starting_r0, steps):
        """ Build Umbrella potentials"""
        
        x_range = np.round(np.linspace(xmin, xmax, (n_windows + 1))[1:], 3)
        
        umbrella_forces_1 = []
        umbrella_forces_2 = []
        
        for x_val in x_range:   
            force_rate_for_x_val = (x_val - starting_r0) / steps
            
            self.umbrella_force_1 = self.f.com_force(
                com_list=com_list,                        
                ref_list=ref_list,                        
                stiff=f'{stiff}',                    
                r0=f'{starting_r0}',                       
                PBC='1',                         
                rate=f'{force_rate_for_x_val}',
            )        
            umbrella_forces_1.append(self.umbrella_force_1)
            
            self.umbrella_force_2= self.f.com_force(
                com_list=ref_list,                        
                ref_list=com_list,                        
                stiff=f'{stiff}',                    
                r0=f'{starting_r0}',                       
                PBC='1',                          
                rate=f'{force_rate_for_x_val}',
            )
            umbrella_forces_2.append(self.umbrella_force_2)  
        self.forces_list = np.transpose(np.array([umbrella_forces_1, umbrella_forces_2])) 
        
        
    def fig_ax(self):
        self.ax = self.wham.plt_fig(title='Free Energy Profile', xlabel='End-to-End Distance (nm)', ylabel='Free Energy / k$_B$T')
    
    
    def plot_free(self, ax=None, title='Free Energy Profile', c=None, label=None, fmt=None):
        self.wham.plot_free_energy(ax=ax, title=title, label=label)
       

    def initialize_observables(self, com_list, ref_list, print_every=1e4, name='all_observables.txt', force_energy_split=True):
        self.observables.com_distance_observable(com_list, ref_list, print_every=print_every, name=name)



class UmbrellaBuild:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def build(self, sims, input_parameters, forces_list, observables_list,
              observable=False, sequence_dependant=False, cms_observable=False, protein=None, force_file=None):

        self.prepare_simulation_environment(sims)

        workers = self.base_umbrella.info_utils.get_number_of_processes() - 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._build_simulation, sim, forces, input_parameters, observables_list,
                                       observable, sequence_dependant, cms_observable, protein, force_file)
                       for sim, forces in zip(sims, forces_list)]

            for future in futures:
                try:
                    future.result()  # Wait for each future to complete and handle exceptions
                except Exception as e:
                    print(f"An error occurred: {e}")
        self.parallel_force_group_name(sims)
    
    
    def _build_simulation(self, sim, forces, input_parameters, observables_list,
                          observable, sequence_dependant, cms_observable, protein, force_file):
        try:
            sim.build(clean_build='force')
            sim.input_file(input_parameters)
            if (protein is not None) and (protein is not False):
                sim.add_protein_par()
            if (force_file is not None) and  (force_file is not False):
                sim.add_force_file()
            for force in forces:
                sim.add_force(force)
            if observable:
                for observables in observables_list:
                    sim.add_observable(observables)
            if (cms_observable is not False) and (cms_observable is not None):
                for cms_obs_dict in cms_observable:
                    sim.oxpy_run.cms_obs(cms_obs_dict['idx'],
                                         name=cms_obs_dict['name'],
                                         print_every=cms_obs_dict['print_every'])
            if sequence_dependant:
                sim.make_sequence_dependant()
            sim.sim_files.parse_current_files()
        except Exception as e:
            error_traceback = traceback.format_exc() 
            print(error_traceback) # Gets the full traceback
            raise e

    def parallel_force_group_name(self, sims):
        workers = self.base_umbrella.info_utils.get_number_of_processes() - 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(self.process_simulation, sims)

    def process_simulation(self, sim):
        sim.sim_files.parse_current_files()
        with open(sim.sim_files.force, 'r') as f:
            force_js = load(f)
        force_js_modified = {key: {'group_name': key, **value} for key, value in force_js.items()}
        with open(sim.sim_files.force, 'w') as f:
            dump(force_js_modified, f, indent=4)       


    def prepare_simulation_environment(self, sims):
        umbrella_stage = sims[0].sim_dir.split('/')[-2]
        
        if exists(join(self.base_umbrella.system_dir, umbrella_stage)) and bool(os.listdir(join(self.base_umbrella.system_dir, umbrella_stage))):
            if self.base_umbrella.clean_build is True:
                answer = input('Are you sure you want to delete all simulation files? Type y/yes to continue or anything else to return use UmbrellaSampling(clean_build=str(force) to skip this message')
                if answer.lower() not in ['y', 'yes']:
                    sys.exit('\nContinue a previous umbrella simulation using:\ncontinue_run=int(n_steps)')
            
                elif answer.lower() in ['y', 'yes']:
                    shutil.rmtree(join(self.base_umbrella.system_dir, umbrella_stage))
                    os.mkdir(join(self.base_umbrella.system_dir, umbrella_stage))  
            
            elif self.base_umbrella.clean_build == False:
                sys.exit('\nThe simulation directory already exists, if you wish to write over the directory set:\nUmbrellaSampling(clean_build=str(force)).\n\nTo continue a previous umbrella simulation use:\ncontinue_run=int(n_steps)')
                
                
            elif self.base_umbrella.clean_build in 'force':
                shutil.rmtree(join(self.base_umbrella.system_dir, umbrella_stage))
                os.mkdir(join(self.base_umbrella.system_dir, umbrella_stage))

        return None
    

class UmbrellaInfoUtils:
    def __init__(self, base_umbrella):
        self.base = base_umbrella
        
    def get_wham_biases(self):
        #I need to get the freefile
        #The free file is in the com_dim
        with open(f'{self.base.com_dir}/freefile', 'r') as f:
            data = f.readlines()
            
        for idx, line in enumerate(data):
            if '#Window' in line:
                break
            else:
                pass
        data = data[idx:]
        data = [line[1:] for line in data]
        data = [line.replace('\t', ' ') for line in data]
        data = data[1:]
        data = [line.split() for line in data]
        data = [np.double(line[1]) for line in data]
 
        return data
   
    def get_com_distance_by_window(self):
        com_distance_by_window = {}
        for idx,sim in enumerate(self.base.production_sims):
            sim.sim_files.parse_current_files()
            df = pd.read_csv(sim.sim_files.com_distance, header=None, engine='pyarrow', dtype=np.double, names=['com_distance'])
            com_distance_by_window[idx] = df
        self.base.com_by_window = com_distance_by_window
                
    def get_r0_values(self):
        self.base.r0 = []
        force_files = [sim.sim_files.force for sim in self.base.equlibration_sims]
        for force_file in force_files:
            with open(force_file, 'r') as f:
                force_js = load(f)
            forces = list(force_js.keys())
            self.base.r0.append(float(force_js[forces[-1]]['r0']))
            
    def get_stiff_value(self):
        force_files = [sim.sim_files.force for sim in self.base.equlibration_sims]
        for force_file in force_files:
            with open(force_file, 'r') as f:
                force_js = load(f)
            forces = list(force_js.keys())
            break
        self.base.stiff = float(force_js[forces[-1]]['stiff'])
        
    def get_temperature(self):
        pre_temp = self.base.equlibration_sims[0].input.input['T']
        if ('C'.upper() in pre_temp) or ('C'.lower() in pre_temp):
            self.base.temperature = (float(pre_temp[:-1]) + 273.15) / 3000
        elif ('K'.upper() in pre_temp) or ('K'.lower() in pre_temp):
             self.base.temperature = float(pre_temp[:-1]) / 3000
             
    def get_n_particles_in_system(self):
        top_file = self.base.equlibration_sims[0].sim_files.top
        with open(top_file, 'r') as f:
            self.base.n_particles_in_system = np.double(f.readline().split(' ')[0])
             
    def get_n_windows(self):
        
        try:
            self.base.n_windows = len(self.base.production_sims)
        except:
            pass
        try:
            self.base.n_windows = len(self.base.equlibration_sims)
        except:
            pass
        try:
            self.base.n_windows = len(self.base.pre_equlibration_sims)
        except:
            print('No simulations found')
            
            
    def get_n_external_forces(self):
        try:
            force_file = [file for file in os.listdir(self.base.file_dir) if file.endswith('force.txt')][0]
            force_file = f'{self.base.file_dir}/{force_file}'
        
            number_of_forces = 0

            with open(force_file, 'r') as f:
                for line in f:
                    if '{' in line:
                        number_of_forces += 1
        except:
            number_of_forces = 0
            
        return number_of_forces + 2
    
    
    def get_number_of_processes(self):
        try:
            # Try using os.sched_getaffinity() available on some Unix systems
            if sys.platform.startswith('linux'):
                return len(os.sched_getaffinity(0))
            else:
                # Fallback to multiprocessing.cpu_count() which works cross-platform
                return mp.cpu_count()
        except Exception as e:
            # Handle possible exceptions (e.g., no access to CPU info)
            print(f"Failed to determine the number of CPUs: {e}")
            return 1  # Safe fallback if number of CPUs can't be determined
            
            
    def copy_last_conf_from_eq_to_prod(self):
        for eq_sim, prod_sim in zip(self.base.equlibration_sims, self.base.production_sims):
            shutil.copyfile(eq_sim.sim_files.last_conf, f'{prod_sim.sim_dir}/last_conf.dat')


class UmbrellaProgress:
    def __init__(self, base_umbrella):
        self.base = base_umbrella
        
        
    def read_pre_equlibration_progress(self):
        if exists(join(self.base.system_dir, 'pre_equlibration')):
            self.base.pre_equlibration_sim_dir = join(self.base.system_dir, 'pre_equlibration')
            n_windows = len(os.listdir(self.base.pre_equlibration_sim_dir))
            self.base.pre_equlibration_sims = []
            for window in range(n_windows):
                self.base.pre_equlibration_sims.append(Simulation(join(self.base.pre_equlibration_sim_dir, str(window)), join(self.base.pre_equlibration_sim_dir, str(window))))

            
    def read_equlibration_progress(self):
        if exists(join(self.base.system_dir, 'equlibration')):
            self.base.equlibration_sim_dir = join(self.base.system_dir, 'equlibration')
            self.base.n_windows = len(os.listdir(self.base.equlibration_sim_dir))
            self.base.equlibration_sims = []
            for window in range(self.base.n_windows):
                self.base.equlibration_sims.append(Simulation(join(self.base.equlibration_sim_dir, str(window)), join(self.base.equlibration_sim_dir, str(window))))
    
    
    def read_production_progress(self):              
        if exists(join(self.base.system_dir, 'production')):
            self.base.production_sim_dir = join(self.base.system_dir, 'production')
            n_windows = len([d for d in os.listdir(self.base.production_sim_dir) if d.isdigit()])
            self.base.production_window_dirs = [join(self.base.production_sim_dir, str(window)) for window in range(n_windows)]
            
            self.base.production_sims = []
            for s, window_dir, window in zip(self.base.equlibration_sims, self.base.production_window_dirs, range(n_windows)):
                self.base.production_sims.append(Simulation(self.base.equlibration_sims[window].sim_dir, str(window_dir))) 
    
    
    def read_wham_progress(self):          
        if exists(join(self.base.system_dir, 'production', 'com_dir', 'freefile')):
            self.base.com_dir = join(self.base.system_dir, 'production', 'com_dir')
            with open(join(self.base.production_sim_dir, 'com_dir', 'freefile'), 'r') as f:
                file = f.readlines()
            file = [line for line in file if not line.startswith('#')]
            self.base.n_bins = len(file)
            
            with open(join(self.base.com_dir, 'metadata'), 'r') as f:
                lines = [line.split(' ') for line in f.readlines()]
            
            self.base.wham.xmin = float(lines[0][1])
            self.base.wham.xmax = float(lines[-1][1])
            self.base.wham.umbrella_stiff = float(lines[0][-1])
            self.base.wham.n_bins = self.base.n_bins
            # self.wham.get_n_data_per_com_file()
            self.base.free = self.base.wham.to_si(self.base.n_bins, self.base.com_dir)
            self.base.mean = self.base.wham.w_mean(self.base.free)
            try:
                self.base.standard_error, self.base.confidence_interval = self.base.wham.bootstrap_w_mean_error(self.base.free)
            except:
                self.base.standard_error, self.base.confidence_interval = ('failed', 'failed')
        
        
    def read_convergence_analysis_progress(self):   
        if exists(join(self.base.system_dir, 'production', 'com_dir', 'convergence_dir')):
            try:
                self.base.convergence_dir = join(self.base.com_dir, 'convergence_dir')
                self.base.chunk_convergence_analysis_dir = join(self.base.convergence_dir, 'chunk_convergence_analysis_dir')
                self.base.data_truncated_convergence_analysis_dir = join(self.base.convergence_dir, 'data_truncated_convergence_analysis_dir')  
                self.base.wham.chunk_dirs = [join(self.base.chunk_convergence_analysis_dir, chunk_dir) for chunk_dir in os.listdir(self.base.chunk_convergence_analysis_dir)]
                self.base.wham.data_truncated_dirs = [join(self.data_truncated_convergence_analysis_dir, chunk_dir) for chunk_dir in os.listdir(self.base.data_truncated_convergence_analysis_dir)]
                
                self.base.chunk_dirs_free = [self.base.wham.to_si(self.base.n_bins, chunk_dir) for chunk_dir in self.base.wham.chunk_dirs]
                self.base.chunk_dirs_mean = [self.base.wham.w_mean(free_energy) for free_energy in self.base.chunk_dirs_free]
                try:
                    self.base.chunk_dirs_standard_error, self.base.chunk_dirs_confidence_interval = zip(
                        *[self.base.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.base.chunk_dirs_free]
                    )
                except:
                    self.base.chunk_dirs_standard_error = ['failed' for _ in range(len(self.base.chunk_dirs_free))]
                    self.base.chunk_dirs_confidence_interval = ['failed' for _ in range(len(self.base.chunk_dirs_free))]
    
                    
                self.base.data_truncated_free = [self.base.wham.to_si(self.base.n_bins, chunk_dir) for chunk_dir in self.base.wham.data_truncated_dirs]
                self.base.data_truncated_mean = [self.base.wham.w_mean(free_energy) for free_energy in self.base.data_truncated_free]
                try:
                    self.base.data_truncated_standard_error, self.data_truncated_confidence_interval = zip(
                        *[self.base.wham.bootstrap_w_mean_error(free_energy) for free_energy in self.data_truncated_free]
                    )
                except:
                    self.base.data_truncated_standard_error = ['failed' for _ in range(len(self.base.data_truncated_free))]
                    self.base.data_truncated_confidence_interval = ['failed' for _ in range(len(self.base.data_truncated_free))]
            except:
                pass
    
    
class UmbrellaObservables:
    def __init__(self, base_umbrella):
        self.base = base_umbrella
        self.obs = Observable()
        
        
    def com_distance_observable(self, com_list, ref_list,  print_every=1e4, name='com_distance.txt'):
        """ Build center of mass observable"""
        com_observable = self.obs.distance(
            particle_1=com_list,
            particle_2=ref_list,
            print_every=f'{print_every}',
            name=f'{name}',
            PBC='1'
        )  
        self.base.observables_list.append(com_observable)
    
    
class UmbrellaAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
        
    def view_observable(self, sim_type, idx, sliding_window=False, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
        
        if sim_type == 'pre_eq':
            self.base_umbrella.pre_equlibration_sims[idx].analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)
        
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)

        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)
    
    def hist_observable(self, sim_type, idx, bins=10, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
            
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.hist_observable(observable,
                                                                               fig=False, bins=bins)

        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.hist_observable(observable,
                                                                             fig=False, bins=bins)
            
    def hist_cms(self, sim_type, idx, xmax, print_every, bins=10, fig=True):
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.hist_cms_obs(xmax, print_every, bins=bins, fig=fig)
        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.hist_cms_obs(xmax, print_every,bins=bins, fig=fig)
          
    def view_cms(self, sim_type, idx, xmax, print_every, sliding_window=10, fig=True):
        if sim_type == 'eq':
            self.base_umbrella.equlibration_sims[idx].analysis.view_cms_obs(xmax, print_every, sliding_window=sliding_window, fig=fig)
        if sim_type == 'prod':
            self.base_umbrella.production_sims[idx].analysis.view_cms_obs(xmax, print_every, sliding_window=sliding_window, fig=fig)
  
    def combine_hist_observable(self, observable, idxes, bins=10, fig=True):
        for sim in self.base_umbrella.production_sims[idxes]:    
            file_name = observable['output']['name']
            conf_interval = float(observable['output']['print_every'])
            df = pd.read_csv(f"{self.sim.sim_dir}/{file_name}", header=None, engine='pyarrow')
            df = np.concatenate(np.array(df))
            H, bins = np.histogram(df, density=True, bins=bins)
            H = H * (bins[1] - bins[0])
    
    def view_observables(self, sim_type, sliding_window=False, observable=None):
        if observable == None:
            observable=self.base_umbrella.observables_list[0]
            
        if sim_type == 'eq':
            plt.figure()
            for sim in self.base_umbrella.equlibration_sims:
                sim.analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)

        if sim_type == 'prod':
            plt.figure(figsize=(15,3))
            for sim in self.base_umbrella.production_sims:
                sim.analysis.plot_observable(observable, fig=False, sliding_window=sliding_window)
    
    def view_last_conf(self, sim_type, window):
        if sim_type == 'pre_eq':
            try:
                self.base_umbrella.pre_equlibration_sims[window].analysis.view_last()
            except:
                self.base_umbrella.pre_equlibration_sims[window].analysis.view_init()
                
        if sim_type == 'eq':
            try:
                self.base_umbrella.equlibration_sims[window].analysis.view_last()
            except:
                self.base_umbrella.equlibration_sims[window].analysis.view_init()
        if sim_type == 'prod':
            try:
                self.base_umbrella.production_sims[window].analysis.view_last()
            except:
                self.base_umbrella.production_sims[window].analysis.view_init()    
    
        
        
class UmbrellaWindow:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def pre_equlibration_windows(self, n_windows):
        self.base_umbrella.pre_equlibration_sim_dir = join(self.base_umbrella.system_dir, 'pre_equlibration')
        if not exists(self.base_umbrella.pre_equlibration_sim_dir):
            os.mkdir(self.base_umbrella.pre_equlibration_sim_dir)
        self.base_umbrella.pre_equlibration_sims = [Simulation(self.base_umbrella.file_dir, join(self.base_umbrella.pre_equlibration_sim_dir, str(window))) for window in range(n_windows)]
    
    def equlibration_windows(self, n_windows):
        """ 
        Sets a attribute called equlibration_sims containing simulation objects for all equlibration windows.
        
        Parameters:
            n_windows (int): Number of umbrella sampling windows.
        """
        self.base_umbrella.n_windows = n_windows
        self.base_umbrella.equlibration_sim_dir = join(self.base_umbrella.system_dir, 'equlibration')
        if not exists(self.base_umbrella.equlibration_sim_dir):
            os.mkdir(self.base_umbrella.equlibration_sim_dir)
        
        if hasattr(self.base_umbrella, 'pre_equlibration_sim_dir'):
            self.base_umbrella.equlibration_sims = [Simulation(sim.sim_dir, join(self.base_umbrella.equlibration_sim_dir, sim.sim_dir.split('/')[-1])) for sim in self.base_umbrella.pre_equlibration_sims]
        
        else:
            self.base_umbrella.equlibration_sims = [Simulation(self.base_umbrella.file_dir,join(self.base_umbrella.equlibration_sim_dir, str(window))) for window in range(n_windows)]
     
    
    def production_windows(self, n_windows):
        """ 
        Sets a attribute called production_sims containing simulation objects for all production windows.
        
        Parameters:
            n_windows (int): Number of umbrella sampling windows.
        """
        self.base_umbrella.production_sim_dir = join(self.base_umbrella.system_dir, 'production')
        if not exists(self.base_umbrella.production_sim_dir):
            os.mkdir(self.base_umbrella.production_sim_dir)
        self.base_umbrella.production_window_dirs = [join(self.base_umbrella.production_sim_dir, str(window)) for window in range(n_windows)]
        self.base_umbrella.production_sims = []
        for s, window_dir, window in zip(self.base_umbrella.equlibration_sims, self.base_umbrella.production_window_dirs, range(n_windows)):
            self.base_umbrella.production_sims.append(Simulation(self.base_umbrella.equlibration_sims[window].sim_dir, str(window_dir)))
    


class WhamAnalysis:
    def __init__(self, base_umbrella):
        self.base_umbrella = base_umbrella
    
    def run_wham(self, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        """
        Run Weighted Histogram Analysis Method on production windows.
        
        Parameters:
            wham_dir (str): Path to wham executable.
            xmin (str): Minimum distance of center of mass order parameter in simulation units.
            xmax (str): Maximum distance of center of mass order parameter in simulation units.
            umbrella_stiff (str): The parameter used to modified the stiffness of the center of mass spring potential
            n_bins (str): number of histogram bins to use.
            tol (str): Convergence tolerance for the WHAM calculations.
            n_boot (str): Number of monte carlo bootstrapping error analysis iterations to preform.

        """
        self.base_umbrella.com_dir = join(self.base_umbrella.production_sim_dir, 'com_dir')
        self.base_umbrella.info_utils.get_temperature()
        wham_analysis(wham_dir,
                      self.base_umbrella.production_sim_dir,
                      self.base_umbrella.com_dir,
                      str(xmin),
                      str(xmax),
                      str(umbrella_stiff),
                      str(n_bins),
                      str(tol),
                      str(n_boot),
                      str(self.base_umbrella.temperature))
        
        self.get_n_data_per_com_file()

    def get_n_data_per_com_file(self):
        com_dist_file = [file for file in os.listdir(self.base_umbrella.com_dir) if 'com_distance' in file][0]
        first_com_distance_file_name = join(self.base_umbrella.com_dir, com_dist_file)
        with open(first_com_distance_file_name, 'rb') as f:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
        self.base_umbrella.n_data_per_com_file = int(last_line.split()[0])
    
    def chunk_convergence_analysis(self, n_chunks):
        """
        Seperate your data into equal chunks
        """
        chunk_size = (self.base_umbrella.n_data_per_com_file // n_chunks)
        chunk_ends = [chunk_size * n_chunk for n_chunk in range(n_chunks + 1)]
        
        for idx, chunk in enumerate(chunk_ends):
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.chunk_convergence_analysis_dir, f'{chunk_ends[idx - 1]}_{chunk}')
                if not exists(chunk_dir):
                    os.mkdir(chunk_dir)
        
        print(chunk_ends)
        
        self.chunk_dirs = []
        
        for idx, chunk in enumerate(chunk_ends):
            chunk_lower_bound = chunk_ends[idx - 1]
            chunk_upper_bound = chunk
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.chunk_convergence_analysis_dir, f'{chunk_ends[idx - 1]}_{chunk}')
                self.chunk_dirs.append(chunk_dir)
                chunked_wham_analysis(chunk_lower_bound, chunk_upper_bound,
                                      self.wham_dir,
                                      self.base_umbrella.production_sim_dir,
                                      chunk_dir,
                                      str(self.xmin),
                                      str(self.xmax),
                                      str(self.umbrella_stiff),
                                      str(self.n_bins),
                                      str(self.tol),
                                      str(self.n_boot),
                                      str(self.base_umbrella.temperature))
                        
        return print(f'chunk convergence analysis')
    
    def data_truncated_convergence_analysis(self, data_added_per_iteration):
        """
        Seperate your data into equal chunks
        """
        chunk_size = (self.base_umbrella.n_data_per_com_file // data_added_per_iteration)
        chunk_ends = [chunk_size * n_chunk for n_chunk in range(data_added_per_iteration + 1)]
        
        for idx, chunk in enumerate(chunk_ends):
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.data_truncated_convergence_analysis_dir, f'0_{chunk}')
                if not exists(chunk_dir):
                    os.mkdir(chunk_dir)
        
        print(chunk_ends)
        
        self.data_truncated_dirs = []
        
        for idx, chunk in enumerate(chunk_ends):
            chunk_lower_bound = 0
            chunk_upper_bound = chunk
            if chunk == 0:
                pass
            else:
                chunk_dir = join(self.base_umbrella.data_truncated_convergence_analysis_dir, f'0_{chunk}')
                self.data_truncated_dirs.append(chunk_dir)
                chunked_wham_analysis(chunk_lower_bound, chunk_upper_bound,
                                      self.wham_dir,
                                      self.base_umbrella.production_sim_dir,
                                      chunk_dir,
                                      str(self.xmin),
                                      str(self.xmax),
                                      str(self.umbrella_stiff),
                                      str(self.n_bins),
                                      str(self.tol),
                                      str(self.n_boot),
                                      str(self.base_umbrella.temperature))
                        
        return print(f'chunk convergence analysis')  
    

    def convergence_analysis(self, n_chunks, data_added_per_iteration, wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot):
        """
        Split your data into a set number of chunks to check convergence.
        If all chunks are the same your free energy profile if probabily converged
        Also create datasets with iteraterativly more data, to check convergence progress
        """
        self.wham_dir = wham_dir
        self.xmin = xmin
        self.xmax = xmax
        self.umbrella_stiff = umbrella_stiff
        self.n_bins = n_bins
        self.tol = tol
        self.n_boot = n_boot
        
        if not exists(self.base_umbrella.com_dir):
            self.base_umbrella.wham_run(wham_dir, xmin, xmax, umbrella_stiff, n_bins, tol, n_boot)
            
        self.base_umbrella.convergence_dir = join(self.base_umbrella.com_dir, 'convergence_dir')
        self.base_umbrella.chunk_convergence_analysis_dir = join(self.base_umbrella.convergence_dir, 'chunk_convergence_analysis_dir')
        self.base_umbrella.data_truncated_convergence_analysis_dir = join(self.base_umbrella.convergence_dir, 'data_truncated_convergence_analysis_dir')
        
        if exists(self.base_umbrella.convergence_dir):
            shutil.rmtree(self.base_umbrella.convergence_dir)
            
        if not exists(self.base_umbrella.convergence_dir):
            os.mkdir(self.base_umbrella.convergence_dir)
            os.mkdir(self.base_umbrella.chunk_convergence_analysis_dir)
            os.mkdir(self.base_umbrella.data_truncated_convergence_analysis_dir) 
  
        self.chunk_convergence_analysis(n_chunks)
        
        self.base_umbrella.chunk_dirs_free = [self.to_si(self.n_bins, chunk_dir) for chunk_dir in self.chunk_dirs]
        self.base_umbrella.chunk_dirs_mean = [self.w_mean(free_energy) for free_energy in self.base_umbrella.chunk_dirs_free]
        try:
            self.base_umbrella.chunk_dirs_standard_error, self.base_umbrella.chunk_dirs_confidence_interval = zip(
                *[self.bootstrap_w_mean_error(free_energy) for free_energy in self.base_umbrella.chunk_dirs_free]
            )
        except:
            self.base_umbrella.chunk_dirs_standard_error = ['failed' for _ in range(len(self.base_umbrella.chunk_dirs_free))]
            self.base_umbrella.chunk_dirs_confidence_interval = ['failed' for _ in range(len(self.base_umbrella.chunk_dirs_free))]


        self.data_truncated_convergence_analysis(data_added_per_iteration)
        
        self.base_umbrella.data_truncated_free = [self.to_si(self.n_bins, chunk_dir) for chunk_dir in self.data_truncated_dirs]
        self.base_umbrella.data_truncated_mean = [self.w_mean(free_energy) for free_energy in self.base_umbrella.data_truncated_free]
        try:
            self.base_umbrella.data_truncated_standard_error, self.base_umbrella.data_truncated_confidence_interval = zip(
                *[self.bootstrap_w_mean_error(free_energy) for free_energy in self.base_umbrella.data_truncated_free]
            )
        except:
            self.base_umbrella.data_truncated_standard_error = ['failed' for _ in range(len(self.base_umbrella.data_truncated_free))]
            self.base_umbrella.data_truncated_confidence_interval = ['failed' for _ in range(len(self.base_umbrella.data_truncated_free))]
        
        return None
    
    
    def to_si(self, n_bins, com_dir):
        free = pd.read_csv(f'{com_dir}/freefile', sep='\t', nrows=int(n_bins))
        free['Free'] = free['Free'].div(self.base_umbrella.temperature)
        free['+/-'] = free['+/-'].div(self.base_umbrella.temperature)
        free['#Coor'] *= 0.8518
        return free     
    
    
    def w_mean(self, free_energy):
        free = free_energy.loc[:, 'Free']
        coord = free_energy.loc[:, '#Coor']
        prob = np.exp(-free) / sum(np.exp(-free))
        mean = sum(coord * prob)
        return mean
    
    
    def bootstrap_w_mean_error(self, free_energy, confidence_level=0.99):
        coord = free_energy.loc[:, '#Coor']
        free = free_energy.loc[:, 'Free'] 
        prob = np.exp(-free) / sum(np.exp(-free))
    
        err = free_energy.loc[:, '+/-']
        mask = np.isnan(err)
        err[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), err[~mask])
        cov = np.diag(err**2)
    
        estimate = np.array(multivariate_normal.rvs(mean=free, cov=cov, size=10000, random_state=None))
        est_prob = [np.exp(-est) / sum(np.exp(-est)) for est in estimate]
        means = [sum(coord * e_prob) for e_prob in est_prob]
        standard_error = np.std(means)
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        confidence_interval = z_score * (standard_error / np.sqrt(len(means)))
        return standard_error, confidence_interval
        
    def plt_fig(self, title='Free Energy Profile', xlabel='End-to-End Distance (nm)', ylabel='Free Energy / k$_B$T'):
        from matplotlib.ticker import MultipleLocator
        with plt.style.context(['science', 'no-latex', 'bright']):
            plt.figure(dpi=200, figsize=(5.5, 4.5))
            # plt.title(title)
            plt.xlabel(xlabel, size=12)
            plt.ylabel(ylabel, size=12)
            #plt.rcParams['text.usetex'] = True
            # plt.rcParams['xtick.direction'] = 'in'
            # plt.rcParams['ytick.direction'] = 'in'
            # plt.rcParams['xtick.major.size'] = 6
            # plt.rcParams['xtick.minor.size'] = 4
            # plt.rcParams['ytick.major.size'] = 6
            # #plt.rcParams['ytick.minor.size'] = 4
            # plt.rcParams['axes.linewidth'] = 1.25
            # plt.rcParams['mathtext.fontset'] = 'stix'
            # plt.rcParams['font.family'] = 'STIXGeneral'
            ax = plt.gca()
            # ax.set_aspect('auto')
            # ax.xaxis.set_minor_locator(MultipleLocator(5))
            #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
            # ax.tick_params(axis='both', which='major', labelsize=9)
            # ax.tick_params(axis='both', which='minor', labelsize=9)
            ax.yaxis.set_ticks_position('both')
        return ax

    
    def plot_indicator(self, indicator, ax, c='black', label=None):
        target = indicator[0]
        nearest = self.base_umbrella.free.iloc[(self.base_umbrella.free['#Coor'] -target).abs().argsort()[:1]]
        near_true = nearest
        x_val = near_true['#Coor']
        y_val = near_true['Free']
        ax.scatter(x_val, y_val, s=50)
        return None
    
    def plot_chunks_free_energy(self, ax=None, title='Free Energy Profile', label=None,errorevery=1):
        if ax is None:
            ax = self.plt_fig()
        for idx, df in enumerate(self.base_umbrella.chunk_dirs_free):
            # if label is None:
            label = self.chunk_dirs[idx].split('/')[-1]

            indicator = [self.base_umbrella.chunk_dirs_mean[idx], self.base_umbrella.chunk_dirs_standard_error[idx]]
            try:
                ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                         yerr=df.loc[:, '+/-'], capsize=2.5, capthick=1.2,
                         linewidth=1.5, errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
                if indicator is not None:
                    self.plot_indicator(indicator, ax, label=label)
            except:
                ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)  
                
    def plot_truncated_free_energy(self, ax=None, title='Free Energy Profile', label=None,errorevery=1):
        if ax is None:
            ax = self.plt_fig()
        for idx, df in enumerate(self.base_umbrella.data_truncated_free):
            # if label is None:
            label = self.data_truncated_dirs[idx].split('/')[-1]

            indicator = [self.base_umbrella.data_truncated_mean[idx], self.base_umbrella.data_truncated_standard_error[idx]]
            try:
                ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                         yerr=df.loc[:, '+/-'], capsize=2.5, capthick=1.2,
                         linewidth=1.5, errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
                if indicator is not None:
                    self.plot_indicator(indicator, ax, label=label)
            except:
                ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)  
    
    def plot_free_energy(self, ax=None, title='Free Energy Profile', label=None, errorevery=1, confidence_level=0.95):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
    
        df = self.base_umbrella.free
        try:
            # Calculate the Z-value from the confidence level
            z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    
            # Calculate the confidence interval
            confidence_interval = z_value * self.base_umbrella.standard_error
            ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Free'],
                        yerr=confidence_interval,  # Use confidence_interval here
                        capsize=2.5, capthick=1.2,
                        linewidth=1.5, errorevery=errorevery,
                        label=f'{label} {self.base_umbrella.mean:.2f} nm \u00B1 {confidence_interval:.2f} nm')
            if self.base_umbrella.mean is not None:
                self.plot_indicator([self.base_umbrella.mean, confidence_interval], ax, label=label)
        except:
            ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Free'], label=label)
  
        

    def prob_plot_indicator(self, indicator, ax, label=None):
        target = indicator[0]
        nearest = self.base_umbrella.free.iloc[(self.base_umbrella.free['#Coor'] -target).abs().argsort()[:1]]
        near_true = nearest
        x_val = near_true['#Coor']
        y_val = near_true['Prob']
        ax.scatter(x_val, y_val, s=50)
        return None
            
    def plot_probability(self, ax=None, title='Probability Distribution', label=None, errorevery=15):
        if ax is None:
            ax = self.plt_fig()
        if label is None:
            label = self.base_umbrella.system
        indicator = [self.base_umbrella.mean, self.base_umbrella.standard_error]
        df = self.base_umbrella.free
        try:
            ax.errorbar(df.loc[:, '#Coor'], df.loc[:, 'Prob'],
                     yerr=df.loc[:, '+/-.1'], capsize=2.5, capthick=1.2,
                     linewidth=1.5,errorevery=errorevery, label=f'{label} {indicator[0]:.2f} nm \u00B1 {indicator[1]:.2f} nm')
            if indicator is not None:
                self.prob_plot_indicator(indicator, ax, label=label)
        except:
            ax.plot(df.loc[:, '#Coor'], df.loc[:, 'Prob'], label=label)          