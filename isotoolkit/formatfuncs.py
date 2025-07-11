
import re
import logging
import numpy as np
from pathlib import Path

from isotoolkit import setup_logging


def log_surface_gravity(Mf,L,Teff):
    """
    Parameters
    ----------
    Mf: scalar or 1d-array
        Present-day stellar mass, in M_solar
    L: scalar or 1d-array
        Stellar luminosity in L_solar
    Teff: scalar or 1d-array
        Effective temperature, K

    Returns
    ----------
    Surface grvity, log10(cm * s-2)
    """
    
    G = 6.67*10**(-11)      # Gravitational constant, m3*kg-1*s-2
    sigma = 5.67*10**(-8)   # Stefan-Boltzmann constant, W*m−2*K−4
    L_sun = 3.828*10**26    # Solar luminosity, W
    M_sun = 1.988*10**30    # Solar mass, kg

    L_W = np.multiply(L,L_sun)
    M_kg = np.multiply(Mf,M_sun)
    
    g_sgs = 4*np.pi*sigma*Teff**4*G*M_kg/L_W*1e2  # Surface gravity, cm*s-2
        
    return np.log10(g_sgs)


class BaseIsochroneFormatter:
    """Base class for PARSEC / MIST / BaSTI isochrone formatting"""
    def __init__(self, input_base_dir, output_base_dir, age_grid, phot_dirs, tolerance=0.025, separate_astro=True, custom_columns={}, **kwargs):
        """
        Parameters
        ----------
        input_base_dir: str
            Path to a root folder with isochrones downloaded with one of the loader classes (ParsecLoader, ...)
        output_base_dir: str
            Path to a root folder (existng or not yet existing) for the output 
        age_grid: 1d-array
            List of ages in Gyr for which the isochrones were downloaded. 
            It's also possible to use a different (sparser) grid, but this has not been tested. 
        phot_dirs: list(str)
            List of strings of the following format 'input_photometric_dir:photometry_to_be_extracted'. 
            E.g., if isochrones in folder input_base_dir/UBVplus contain columns of both UBVRIJHK and Gaia GDR3
            photometry, phot_dirs parameter can be ['UBVplus:UBVRIJHK','UBVplus:GDR3']. If each photometry is 
            in a separate input folder, this parameter can look like ['UBVplus:UBVRIJHK','Gaia:GDR3']. 
            Only two output photometric systems are currently supported: UBVRIJHK and GDR3. 
        tolerance: scalar
            Optional. Half-width of age bins in age_grid for extraction of single-age isochrones from the input data. 
            Unit is Gyr. Default is 0.025 Gyr.
        separate_astro: bool
            Optional. If True, astrophysical parameters (mass, luminosity, surface gravity, effective temperature, 
            metallicity, evolutionary phase) will be extracted and saved in a separate folder from photometric columns. Default is True.
        custom_columns: dict
            Optional. Dictionary with additional columns to be extracted and their positions in the isochrone table. 
            Should be organized as {'astro':{'col1':pos1,...},'phot':{'col2':pos2,...}}, where 'phot' should be the actual name 
            of the photometric system ('UBVRIJHK' or 'GDR3' to extract - but check if possible - more columns of this systems, or custom one).
        **kwargs: dict
            Optional. Other parameters. 
        """

        self.lib_name = self.get_library_name()
        self.input_base_dir = Path(input_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.phot_dirs = phot_dirs
        self.age_grid = age_grid
        self.extra = kwargs
        self.separate_astro = separate_astro
        self.custom_columns = custom_columns
        self.tolerance = tolerance
        self.check_tolerance()

        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.column_info = {
            'Mini':'Initial mass, M_solar',
            'Mf':'Final mass, M_solar',
            'logL':'Luminosity log10(L/L_solar)',
            'logg':'Surface gravity log10(cm/s^2)',
            'logT':'Surface effective temperature, log10(K)',
            'FeHf':'Final metallicity (including diffusion)',
            'phase':'Approximate evolutionary phase, use with caution',
            'U_Bessell':'U mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'B_Bessell':'B mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'V_Bessell':'V mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'R_Bessell':'R mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'I_Bessell':'I mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'J_Bessell':'J mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'H_Bessell':'H mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'K_Bessell':'K mag - Maiz-Apellaniz (2006), Bessell (1990), Bessell & Brett (1988)',
            'J_2MASS':'2MASS J mag - Cohen et al. (2003)',
            'H_2MASS':'2MASS H mag - Cohen et al. (2003)',
            'Ks_2MASS':'2MASS K mag - Cohen et al. (2003)',
            'G_Gaia_DR3':'Gaia DR3 G mag - Riello et al. (2021), originally introduced in EDR3',
            'G_BP_Gaia_DR3':'Gaia DR3 G_BP mag - Riello et al. (2021), originally introduced in EDR3',
            'G_RP_Gaia_DR3':'Gaia DR3 G_RP mag - Riello et al. (2021), originally introduced in EDR3',
            'G_RVS_Gaia_DR3':'Gaia DR3 G_RVS mag'
            }
        
        # Column groups
        # Astrophysical parameters
        self.columns_astro = ['Mini','Mf','logL','logT','logg','FeHf','phase'] 
            
        # UBVRIHJK 
        self.columns_ubvrijhk = ['U_Bessell','B_Bessell','V_Bessell','R_Bessell','I_Bessell','J_Bessell','H_Bessell','K_Bessell',
                                 'J_2MASS','H_2MASS','Ks_2MASS']
        # Gaia DR3
        self.columns_gdr3 = ['G_Gaia_DR3','G_BP_Gaia_DR3','G_RP_Gaia_DR3','G_RVS_Gaia_DR3']

        self.column_groups = {'Astro':self.columns_astro,
                              'UBVRIJHK':self.columns_ubvrijhk,
                              'GDR3':self.columns_gdr3
                             }

        # Create logger for tracking the progress
        self.logger = setup_logging(self.output_base_dir/"execution_formatting.log", logging.DEBUG, logging.INFO)

        # Initialization of several library-specific qualtities
        self.column_positions = {}
        self.n_skip = None
        

    def get_library_name(self):
        raise NotImplementedError
    
    def add_custom_columns(self):
        """Must be called in child classes, otherwise custom columns will be redefined in child init."""
        if self.custom_columns != {}:
            for key in self.custom_columns:
                if key in self.column_groups.keys():
                    self.column_groups[key] += list(self.custom_columns[key].keys())
                else:
                    self.column_groups[key] = list(self.custom_columns[key].keys())

            # Add positions of the custom columns
            for key in self.column_groups.keys():
                for column in self.column_groups[key]:
                    if column not in self.column_positions.keys():
                        self.column_positions[column] = self.custom_columns[key][column]
    
    def get_preformatted_isochrone(self,source_dir,age_column_unit):
        raise NotImplementedError

    def print_column_info(self):
        """Displays columns available for a chosen stellar evolution libraary"""
        print('-'*105 + '\n' +  '{:<15}'.format('Column'), '{:<15}'.format('Availability'), 'Description\n' + '-'*105)
        for key,value in self.column_info.items():
            try:
                position = '-' if self.column_positions[key]==None else '+'
                print('{:<20}'.format(key), '{:<10}'.format(position), value)
            except:
                pass
        print('-'*105)

    def select_phot_dir(self,detected_phot_dirs):
        """Checks if the given in parameter phot_dirs input photometric directories exist;
        Reads which photometric columns group should be extracted from each folder.
        """
        dirs_to_process, phot_systems = [], [] 
        for element in self.phot_dirs:
            split = element.split(':')
            if split[0] not in [str(item.name) for item in detected_phot_dirs]:
                raise ValueError(f"Specified photometric input folder {split[0]} is not found.")
            else:
                dirs_to_process.append(self.input_base_dir / split[0])
                phot_systems.append(split[1])
        return {'input_phot_dirs':dirs_to_process,'output_phot_systems':phot_systems}
        
    def prepare_output_directories(self,feh_dir,phot_system,**kwargs):
        """Creates directory tree in output_base_dir (astrophysical and photometric columns can be stored separately)."""

        output_dirs = {}
        
        output_phot_feh_dir = self.output_base_dir / phot_system / feh_dir.parts[-1]
        output_phot_feh_dir.mkdir(parents=True,exist_ok=True)
        output_dirs['output_phot_feh_dir'] = output_phot_feh_dir

        if self.separate_astro:
            self.logger.debug(f"Output directory for photometry: {output_phot_feh_dir}")
        else:
            self.logger.debug(f"Output directory for photometry and astrophysical parameters: {output_phot_feh_dir}")

        if self.separate_astro==True and kwargs['astro_columns']==True:
            output_astro_feh_dir = self.output_base_dir / 'Astro' / feh_dir.parts[-1]
            output_astro_feh_dir.mkdir(parents=True, exist_ok=True)
            output_dirs['output_astro_feh_dir'] = output_astro_feh_dir

            self.logger.debug(f"Output directory for astrophysical parameters: {output_astro_feh_dir}")
        
        return output_dirs
    
    def process_age_groups(self, iso, age_indices_dict, phot_system, output_dirs, **kwargs):
        """Routine for processing isochrone(s) in a single-metallicity subfolder.
        - Splits isochrone(s) into single-age tables according to age_grid and tolerance parameters.
        - For PARSEC only: removes two last rows with unrealistic values of parameters
        - Sorts age-tables by initial mass column (relevant for PARSEC)
        - extracts and saves astrophysical and photometric columns
        """

        # Process each age group
        self.logger.debug("Processing ages... ")

        for age, indices in age_indices_dict.items():

            if len(indices) == 0:
                self.logger.debug(f"Empty age bin for age = {age} Gyr")
                continue

            # Subselect single age
            iso_age = iso[indices, :]

            if 'n_skip' in self.extra:
                iso_age = iso_age[:-self.extra['n_skip'], :]
                #self.logger.debug("Removed last {self.extra['n_skip']} rows")

            # Sort by mass
            iso_age = BaseIsochroneFormatter.sort_by_mass(iso_age, mass_column_index=self.column_positions['Mini'])
            #self.logger.debug("Sorted by mass column")

            if self.separate_astro: 
                # Add ID column
                ids = np.add(1000,np.arange(len(iso_age))).reshape(-1,1)
                
                phot_cols = {key:self.column_positions[key] for key in self.column_groups[phot_system] if self.column_positions[key] != None}
                self._extract_columns_and_save(age,iso_age,phot_cols,output_dirs['output_phot_feh_dir'],ids=ids)

                if kwargs['astro_columns']==True:
                    astro_cols = {key:self.column_positions[key] for key in self.column_groups['Astro'] if self.column_positions[key] != None}
                    self._extract_columns_and_save(age,iso_age,astro_cols,output_dirs['output_astro_feh_dir'],ids=ids)
            else:
                # Extract astrophysical and photometric columns together and save to the folder with photometry name
                astro_cols = {key:self.column_positions[key] for key in self.column_groups['Astro'] if self.column_positions[key] != None}
                phot_cols = {key:self.column_positions[key] for key in self.column_groups[phot_system] if self.column_positions[key] != None}
                all_cols = {**astro_cols, **phot_cols}
                self._extract_columns_and_save(age,iso_age,all_cols,output_dirs['output_phot_feh_dir'])


    def process_metallicity_folder(self, feh_dir, phot_system, age_column_unit, **kwargs):
        """
        Process one isochrone from a single-metallicity folder. In case of multiple isochrones, 
        this method can be used in a cycle to process the whole folder.
        - Reads isochrone in the folder
        - Splits an isochrone into single-age tables
        - Applies processing steps from process_age_groups
        """
        
        feh_value = str(feh_dir).split("iso_feh")[-1]
        self.logger.info(f"Processing metallicity [Fe/H] = {feh_value}")

        iso = self.get_preformatted_isochrone(feh_dir, age_column_unit)  
                                              
        # Split isochrone into single-age tables
        age_indices_dict = self.split_by_age(iso)
        
        # Prepare output directories: metallicity folder in Astro (if not existing yet) and current photometry
        output_dirs = self.prepare_output_directories(feh_dir,phot_system,**kwargs)

        # Process age groups
        self.process_age_groups(iso, 
                                age_indices_dict,
                                phot_system, 
                                output_dirs, 
                                **kwargs
                                )

    @staticmethod
    def sort_by_mass(isochrone, mass_column_index=0, axis=0):
        """Sort isochrone entries by initial mass."""
        if axis == 1:
            mass = isochrone[mass_column_index]
            indices = np.argsort(mass)
            return np.array([col[indices] for col in isochrone])
        else:
            mass = isochrone[:, mass_column_index]
            indices = np.argsort(mass)
            return isochrone[indices]

    @staticmethod
    def to_gyr(age_array: np.ndarray, unit: str) -> np.ndarray:
        """Convert ages to Gyr from their input format (yr, log(yr), Myr)"""
        if unit == 'logyr':
            return 10**age_array / 1e9
        elif unit == 'yr':
            return age_array / 1e9
        elif unit == 'myr':
            return age_array / 1e3
        elif unit == 'gyr':
            return age_array
        else:
            raise ValueError(f"Unsupported age unit: {unit}")
    
    def split_by_age(self, iso):
        """
        Splits indices of isochrone entries by rounded age (in Gyr).
        Returns a dictionary of {age: indices}.
        """
        age_column = iso[:, self.column_positions['Age']]
        
        age_indices = {}
        for age in self.age_grid:
            age = np.round(age, 2)
            indices = np.where((age_column >= age - self.tolerance) & (age_column < age + self.tolerance))[0]
            if len(indices)==0:
                self.logger.debug(f"No data found for age = {age} Gyr")
            age_indices[str(age)] = indices

        # Consistency check
        len_initial = len(age_column)
        len_final_sum = sum(map(len,age_indices.values()))
        if len_final_sum < len_initial:
            self.logger.info(f"Number of lost rows during age sorting: {len_initial - len_final_sum}/{len_initial}")
        if len_final_sum > len_initial:
            self.logger.info(f"Number of excessive rows after age sorting: {len_final_sum - len_initial}/{len_initial}. Decrease tolerance.")

        self.logger.debug('Sorted age column into the given age grid')

        return age_indices

    def save_isochrone_block(self, output_path, data, col_fmt=None, col_names=None):
        """
        Save a 2D NumPy array with sign-aware alignment for negative values.
        """
        if not col_fmt:
            col_fmt = ['%.8f' for element in len(data[0,:])]

        def extract_width(fmt):
            if fmt == '%d':
                return 10
            return 15

        # Extract column widths
        field_widths = [extract_width(fmt) for fmt in col_fmt]

        def format_row(row, is_header=False):
            cells = []
            for val, fmt, width in zip(row, col_fmt, field_widths):
                if is_header:
                    cells.append(f"{val:>{width}}")  # Right-align headers
                elif '%d' in fmt:
                    cells.append(f"{int(val):>{width}}")
                else:
                    val_str = fmt % val
                    cells.append(f"{val_str:>{width}}")
            return ' '.join(cells) + '\n'
        
        with open(output_path, 'w') as f:
            f.write(format_row(col_names, is_header=True))
            for row in data:
                f.write(format_row(row))


    @staticmethod
    def list_files_in_folder(path):
        """List folder content, returns dict with 'folders' and 'files' keywords."""
        output = {'folders':[], 'files':[]}

        path = Path(path)

        for item in path.iterdir():
            if item.is_dir():
                output['folders'].append(item)
            elif item.is_file():
                output['files'].append(item)
        
        return output
    
    def check_tolerance(self):
        """Ensure that age tolerance mathces the provided age grid."""
        min_age_step = round(min(np.diff(self.age_grid)),3)
        if self.tolerance > min_age_step/2:
            msg = f"Age tolerance {self.tolerance} Gyr is too large for the provided age grid. Use value less than {min_age_step/2} Gyr"
            self.logger.debug(msg)
            raise ValueError(msg)
        
        return self.tolerance

    def include_astro_columns(self,n_feh_dirs):
        """Check whether astrophysical parameter have already been extracted for this metallicity"""
        astro_path = self.output_base_dir / 'Astro'
        if astro_path.exists():
            feh_subfolders = BaseIsochroneFormatter.list_files_in_folder(astro_path)['folders']
            if len(feh_subfolders) == n_feh_dirs:
                return False
        return True
    
    def _extract_columns_and_save(self,age,iso_age,cols,output_dir,**kwargs):
        """Get specified columns from siochrone and save the subtable."""
        if 'ids' in kwargs:
            iso_sub = np.hstack((kwargs['ids'],iso_age[:, list(cols.values())]))
            col_fmt = ['%d'] + ['%.8f' for element in range(len(cols))]
            col_names = ['ID'] + list(cols.keys())
        else:
            iso_sub = iso_age[:, list(cols.values())]
            col_fmt = ['%.8f' for element in range(len(cols))]
            col_names = list(cols.keys())

        if 'phase' in cols:
            col_phase_idx = list(cols.keys()).index('phase')
            if 'ids' in kwargs:
                col_fmt[col_phase_idx + 1] = '%d'
            else:
                col_fmt[col_phase_idx] = '%d'
        
        self.save_isochrone_block(output_dir / f"iso_age{age}.txt",
                                  iso_sub,
                                  col_fmt=col_fmt,
                                  col_names=col_names
                                  )
    
    def run(self):
        """Process all input"""

        self.logger.info(f"{self.lib_name} Isochrone Formatter\n" + '='*100)

        # Detect all photometric folders in the root input directory
        phot_dirs = self.list_files_in_folder(self.input_base_dir)['folders']
        self.logger.info("Detected photometric folders: " + ', '.join([str(item) for item in phot_dirs])+'\n' + '-'*100)

        if len(phot_dirs) == 0:
            raise ValueError(f"No folders detected in {self.input_base_dir}!")

        phot_info = self.select_phot_dir(phot_dirs)
        phot_dirs_for_processing = phot_info['input_phot_dirs']
        output_phot_systems = phot_info['output_phot_systems']

        for phot_dir_for_processing, output_phot_system in zip(phot_dirs_for_processing,output_phot_systems):
            self.logger.info(f"Photometric folder for processing: {phot_dir_for_processing}")
            self.logger.info(f"Output photometric system: {output_phot_system}")

            if self.separate_astro:
                self.astro_dir = self.output_base_dir / 'Astro'
                self.astro_dir.mkdir(parents=True, exist_ok=True)

            # Check if output phot directory exists
            output_phot_dir = self.output_base_dir / output_phot_system
            output_phot_dir.mkdir(parents=True, exist_ok=True)

            # Detect metallicity subfolders
            feh_dirs = self.list_files_in_folder(phot_dir_for_processing)['folders']

            def extract_feh(path):
                match = re.search(r"iso_feh(-?\d+\.\d+)", path.name)
                return float(match.group(1)) if match else float('inf')  # use inf if no match

            # Sort by extracted value
            feh_dirs = sorted(feh_dirs, key=extract_feh)

            self.logger.info(f"Detected {len(feh_dirs)} metallicity subfolders")

            kwargs = {}
            if self.separate_astro:
                kwargs['astro_columns'] = self.include_astro_columns(len(feh_dirs))

            for feh_dir in feh_dirs:
                self.process_metallicity_folder(feh_dir, 
                                                output_phot_system, 
                                                self.extra['age_column_unit'], 
                                                **kwargs, 
                                                )
            self.logger.info("Finished processng photometric folder\n" + '-'*100)

        self.logger.info("Done")

    
class ParsecIsochroneFormatter(BaseIsochroneFormatter):
    def __init__(self, input_base_dir, output_base_dir, age_grid, phot_dirs, **kwargs):            
        super().__init__(input_base_dir, output_base_dir, age_grid, phot_dirs, **kwargs)

        self.column_positions = {
            'Age':2,'Mini':3,'Mf':5,'logL':6,'logg':8,'logT':7,'FeHf':None,'phase':9,
            'U_Bessell':28,'B_Bessell':29,'V_Bessell':30,'R_Bessell':31,'I_Bessell':32,
            'J_Bessell':33,'H_Bessell':34,'K_Bessell':35,'J_2MASS':None,'H_2MASS':None,'Ks_2MASS':None,
            'G_Gaia_DR3':28,'G_BP_Gaia_DR3':29,'G_RP_Gaia_DR3':30,'G_RVS_Gaia_DR3':None
            }
        self.add_custom_columns()
        self.extra['age_column_unit'] = 'logyr'
        self.extra['n_skip'] = 2 

    def get_library_name(self):
        return 'PARSEC'

    def read_isochrone(self, feh_dir_path):
        """For PARSEC header and footer should be skipped"""
        iso_path = feh_dir_path / f"{feh_dir_path.name}.txt"
        iso = np.genfromtxt(iso_path, skip_header=1, skip_footer=1)

        return iso
    
    def get_preformatted_isochrone(self,feh_dir,age_column_unit):
        iso = self.read_isochrone(feh_dir)
        iso[:, self.column_positions['Age']] = BaseIsochroneFormatter.to_gyr(iso[:, self.column_positions['Age']],age_column_unit)

        self.logger.debug('Preformatted isochrone: age to Gyr')
        return iso


class MistIsochroneFormatter(BaseIsochroneFormatter):
    def __init__(self, input_base_dir, output_base_dir, age_grid, phot_dirs=None, **kwargs):
        super().__init__(input_base_dir, output_base_dir, age_grid, phot_dirs=phot_dirs, **kwargs)
        
        self.column_positions = {
            'Age':1,'Mini':2,'Mf':3,'logL':6,'logg':5,'logT':4,'FeHf':8,'phase':33,
            'U_Bessell':9,'B_Bessell':10,'V_Bessell':11,'R_Bessell':12,'I_Bessell':13,
            'J_Bessell':None,'H_Bessell':None,'K_Bessell':None,'J_2MASS':14,'H_2MASS':15,'Ks_2MASS':16,
            'G_Gaia_DR3':30,'G_BP_Gaia_DR3':31,'G_RP_Gaia_DR3':32,'G_RVS_Gaia_DR3':None
            }
        self.column_groups['UBVRIJHK+GDR3'] = self.columns_ubvrijhk + self.columns_gdr3
        self.add_custom_columns()
        self.extra['age_column_unit'] = 'yr'

    def get_library_name(self):
        return 'MIST'
    
    def read_isochrone(self, feh_dir_path):
        """Each MIST metallicity folder contains several parts of an isochrone for the full 0-13 Gyr 
        age grid (if step is 0.05 Gyr). All tables are concatenated into a single full isochrone."""
        iso_paths = BaseIsochroneFormatter.list_files_in_folder(feh_dir_path)['files']        
        iso_parts = [np.loadtxt(part) for part in iso_paths]
        iso = np.concatenate(iso_parts, axis=0)
        return iso

    def get_preformatted_isochrone(self, feh_dir,age_column_unit):

        iso = self.read_isochrone(feh_dir)
        iso[:, self.column_positions['Age']] = BaseIsochroneFormatter.to_gyr(iso[:, self.column_positions['Age']],age_column_unit)
        
        self.logger.debug('Preformatted isochrone: age to Gyr, sorted by initial mass')

        return iso


class BastiIsochroneFormatter(BaseIsochroneFormatter):

    def __init__(self, input_base_dir, output_base_dir, age_grid, phot_dirs, **kwargs):
        super().__init__(input_base_dir, output_base_dir, age_grid, phot_dirs, **kwargs)

        self.column_positions = {
            'Mini':0,'Mf':1,'logL':2,'logg':None,'logT':3,'FeHf':None,'phase':None,
            'U_Bessell':4,'B_Bessell':6,'V_Bessell':7,'R_Bessell':8,'I_Bessell':9,
            'J_Bessell':10,'H_Bessell':11,'K_Bessell':12,'J_2MASS':None,'H_2MASS':None,'Ks_2MASS':None,
            'G_Gaia_DR3':4,'G_BP_Gaia_DR3':5,'G_RP_Gaia_DR3':6,'G_RVS_Gaia_DR3':7
            }
        self.add_custom_columns()
        self.extra['age_column_unit'] = 'myr'

    def get_library_name(self):
        return 'BaSTI'
    
    def read_isochrone(self,iso_path):
        iso = np.loadtxt(iso_path)
        return iso
    
    def get_isochrone_age(self,iso_path,age_unit):
        """Single-age BaSTI isochrones are stored in a separate files already after the download.
        Age is written in a header, this method extracts its value and converts to Gyr."""
        with open(iso_path, 'r') as f:
            for _ in range(5):
                header_line = f.readline()
            age = float(header_line.split()[-1])
            age = BaseIsochroneFormatter.to_gyr(age,age_unit)
        return age

    def get_preformatted_isochrone_basti(self,iso_path,age_unit):
        """Read single-age isochrone, add logg column, extract age"""
        iso = self.read_isochrone(iso_path)
        iso = BaseIsochroneFormatter.sort_by_mass(iso, mass_column_index=self.column_positions['Mini'])
        logg = log_surface_gravity(iso[:,self.column_positions['Mf']],
                                    10**iso[:,self.column_positions['logL']],
                                    10**iso[:,self.column_positions['logT']]
                                    )
        np.column_stack((iso, logg))
        self.column_positions['logg'] = iso.shape[1] - 1
                        

        iso_age = self.get_isochrone_age(iso_path,age_unit)

        self.logger.debug('Preformatted isochrone: age to Gyr, sorted by initial mass, added logg')

        return iso, iso_age

    def process_basti_age_groups(self, feh_dir, phot_system, age_column_unit, output_dirs, **kwargs):
        """Process all ages for single metallicity."""

        # Process each age group
        self.logger.debug("Processing ages... ")

        # Go to met folder and find iso files
        iso_files = BaseIsochroneFormatter.list_files_in_folder(feh_dir)['files']

        for filename in iso_files:
            
            iso_age, age = self.get_preformatted_isochrone_basti(filename,age_column_unit)

            if self.separate_astro:
                # Add ID column
                ids = np.add(1000,np.arange(len(iso_age))).reshape(-1,1)
            
                phot_cols = {key:self.column_positions[key] for key in self.column_groups[phot_system] if self.column_positions[key] != None}
                self._extract_columns_and_save(age,iso_age,phot_cols,output_dirs['output_phot_feh_dir'],ids=ids)

                if kwargs['astro_columns']==True:
                    astro_cols = {key:self.column_positions[key] for key in self.column_groups['Astro'] if self.column_positions[key] != None}
                    self._extract_columns_and_save(age,iso_age,astro_cols,output_dirs['output_astro_feh_dir'],ids=ids)
            else:
                astro_cols = {key:self.column_positions[key] for key in self.column_groups['Astro'] if self.column_positions[key] != None}
                phot_cols = {key:self.column_positions[key] for key in self.column_groups[phot_system] if self.column_positions[key] != None}
                all_cols = {**astro_cols, **phot_cols}
                self._extract_columns_and_save(age,iso_age,all_cols,output_dirs['output_phot_feh_dir'],**kwargs)
                                

    def process_metallicity_folder(self, feh_dir, phot_system, age_column_unit, **kwargs):
        """Process all ages in a single-metallicity folder"""

        feh_value = str(feh_dir).split("iso_feh")[-1]
        self.logger.info(f"Processing metallicity [Fe/H] = {feh_value}")
        
        # Prepare output directories: metallicity folder in Astro (if not existing yet) and current photometry
        output_dirs = self.prepare_output_directories(feh_dir,phot_system,**kwargs)

        # Process age groups
        self.process_basti_age_groups(feh_dir,
                                      phot_system,
                                      age_column_unit,
                                      output_dirs, 
                                      **kwargs
                                      )