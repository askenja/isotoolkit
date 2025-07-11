
import os
import time
import zipfile
import tarfile
import logging
import requests
import numpy as np
from pathlib import Path
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from isotoolkit import setup_logging


def retry_action(action, attempts, delay, logger, success_msg=None, failure_msg=None):
    """Retry an action multiple times with a delay between attempts."""
    for _ in range(attempts):
        try:
            result = action()
            if success_msg:
                logger.debug(success_msg)
            return result
        except Exception as e:
            time.sleep(delay)
    if failure_msg:
        logger.info(failure_msg)
    return None

def max_digits_after_point(arr):
    """Calculate the maximum number of digits 
    after the decimal point in a list of numbers."""
    max_digits = 0
    for num in arr:
        s = str(num)
        if '.' in s:
            digits = len(s.split('.')[-1].rstrip('0'))
            max_digits = max(max_digits, digits)
    return max_digits

def compute_split_points(x_start, x_end, x_step, n_lim):
    """Compute split points for a list given the maximum allowed number of elements in a split."""

    total_points = int((x_end - x_start) // x_step + 1)
    precision = max_digits_after_point([x_start, x_end, x_step])

    if total_points <= n_lim:
        return [round(x_start, precision),round(x_end, precision)]
    
    splits = []
    i = 0

    while i < total_points:
        splits.append(i)
        i += n_lim
    
    # Check if the last chunk is too short
    if total_points - i < n_lim // 2:
        splits.pop()
        last_start = splits[-1]
        middle = last_start + (total_points - last_start) // 2
        splits.append(middle)
    
    splits.append(total_points)

    # Convert split *indices* to *x-values*
    x_splits = [round(x_start + i * x_step, precision) for i in splits]

    return x_splits

def configure_browser(dir_out):
    """Configure the Selenium WebDriver with Chrome options for downloading files."""
    # Prepare the full path to the download directory
    download_dir = str(Path(dir_out).resolve())

    # Set Chrome preferences for automatic downloads
    chrome_prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }

    options = Options()
    options.add_experimental_option("prefs", chrome_prefs)

    # Automatically install and use the correct ChromeDriver
    service = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)

    return browser

def configure_target_dir(dir_out, feh, logger=None):
    """Configure the target directory for isochrones of chosen metallicity."""
    target_dir = dir_out / f"iso_feh{feh}"
    target_dir.mkdir(parents=True,exist_ok=True)
    if logger:
        logger.debug(f"Output directory: {target_dir}")
    return target_dir


class IsochroneHandler:
    """Base class for handling downloaded isochrone files."""
    def __init__(self, filepath, dir_out, feh, logger):
        self.filepath = filepath
        self.dir_out = dir_out
        self.feh = feh
        self.logger = logger
        self.target_dir = configure_target_dir(self.dir_out, self.feh, logger=self.logger)

    def unpack_and_organize(self):
        retry_action(
            action=self._unpack,
            attempts=12,
            delay=5,
            logger=self.logger,
            failure_msg="Cannot find or unpack isochrone file."
        )

    def _unpack(self):
        raise NotImplementedError("Subclasses must implement _unpack method.")
    

class BastiIsochroneHandler(IsochroneHandler):
    def _unpack(self):
        """Unpacks the downloaded tar.gz BaSTI isochrones"""
        with tarfile.open(self.filepath, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]  # only files

            for member in members:
                member_name = os.path.basename(member.name)  # strip any internal folder
                target_path = self.target_dir / member_name

                # Optional: skip if file already exists
                if target_path.exists():
                    if self.logger:
                        self.logger.debug(f"File already exists, skipping: {member_name}")
                    continue

                # Override member name temporarily to strip the path
                member.name = member_name
                tar.extract(member, path=self.target_dir)

        os.remove(self.filepath)
        self.logger.debug("Removed the original tar file.")


class MistIsochroneHandler(IsochroneHandler):
    def _unpack(self):
        """Unpacks the downloaded zip MIST isochrones"""
        zip_ref = zipfile.ZipFile(self.filepath, 'r')
        zip_ref.extractall(self.dir_out)
        zip_ref.close()
        filextrname = ''.join((self.filepath[:-3],'cmd'))
        self.logger.debug("Extracted file to " + filextrname)
        i = 0 
        newname = self.target_dir / ''.join(('iso_fe',str(self.feh),'_p',str(i),'.txt'))
        self.logger.debug("Renaming file to " + newname)
        while os.path.isfile(newname):
            i += 1
            newname = self.target_dir / ''.join(('iso_fe',str(self.feh),'_p',str(i),'.txt'))
        os.rename(filextrname,newname)
        os.remove(self.filepath)
        self.logger.debug("Renaimed file and removed original zip")


class ParsecIsochroneHandler(IsochroneHandler):
    def _unpack(self):
        """Renames the downloaded Padova isochrone"""
        newname = self.target_dir / ''.join(('iso_fe',str(self.feh),'.txt'))
        self.logger.debug("Renaming file to " + newname)
        os.rename(self.filepath,newname)
        os.remove(self.filepath)
        self.logger.debug("Renaimed file and removed original isochrone")


class BaseIsochroneForm:
    def __init__(self, browser, dir_out=None, FeH=None, agemin=None, agemax=None, dage=None, photometry = None, n_lim=None, **kwargs):
        
        self.lib_name = self.get_library_name()

        self.browser = browser
        self.FeH_inp = FeH
        self.agemin = agemin
        self.agemax = agemax
        self.dage = dage
        self.photometry = photometry
        self.n_lim = n_lim
        self.extra = kwargs

        if np.isscalar(self.FeH_inp):
            self.FeH_inp = [self.FeH_inp]

        # Set output directory
        self.dir_out = Path(dir_out) if dir_out else Path(os.getcwd())
        self.dir_out.mkdir(parents=True, exist_ok=True)

        self.photometric_systems = {}

        self.logger = setup_logging(self.dir_out / 'execution_downloading.log', logging.DEBUG, logging.INFO)

    def get_library_name(self):
        raise NotImplementedError

    def load_site(self):
        raise NotImplementedError

    def set_alpha_and_grid(self):
        pass 

    def set_ages(self, age_start, age_end, age_step):
        raise NotImplementedError

    def convert_ages(self, agemin, agemax, dage):
        # Default: no conversion
        return agemin, agemax, dage

    def compute_age_breakpoints(self):
        if self.n_lim is None:
            return [self.agemin, self.agemax]
        else:
            return compute_split_points(self.agemin, self.agemax, self.dage, self.n_lim)

    def print_photometric_systems(self):
        """Print available photometric systems."""
        print(f"Available photometric systems for {self.get_library_name()}\n"+'-'*75)
        for key in self.photometric_systems.keys():
            print(key)

    def set_photometry(self):
        raise NotImplementedError

    def set_metallicity(self):
        raise NotImplementedError

    def submit(self):
        raise NotImplementedError
    
    def get_download_link(self):
        raise NotImplementedError
    
    def download(self, download_link):
        raise NotImplementedError
    
    def unpack_and_organize(self):
        pass
    
    def run(self):

        if self.logger:
            self.logger.info(f"{self.lib_name} Isochrone Retriever\n" + '='*75)

        self.load_site()
        
        for k in range(len(self.FeH_inp)):

            self.FeH = self.FeH_inp[k]
        
            if self.logger:
                self.logger.info(f"Processing metallicity [Fe/H] = {self.FeH}\n" + '-'*75)

            breakpoints = self.compute_age_breakpoints()

            for i in range(len(breakpoints) - 1):
                age_start = breakpoints[i]
                age_end = breakpoints[i + 1]
                if self.logger:
                    self.logger.info(f"Processing age range: {age_start} - {age_end} Gyr")

                self.set_alpha_and_grid()
                self.set_metallicity()
                self.set_photometry()
                self.set_ages(age_start, age_end, self.dage)
                self.submit()
                if self.logger:
                    self.logger.debug("Request sent")

                self.unpack_and_organize()
            
                if self.logger:
                    self.logger.debug('-'*40) 

        if self.logger:
            self.logger.info("Done.")


class BastiForm(BaseIsochroneForm):

    def __init__(self, browser, **kwargs):
        super().__init__(browser, **kwargs)

        self.photometric_systems = {
            "HR":'HR diagram',
            "2MASS":'2MASS',
            "CFHT":'CFHT',
            "DECam":'DECam',
            "Euclid":'Euclid (VIS+NISP)',
            "GAIA-DR1":'GAIA DR1',
            "GAIA-DR2":'GAIA DR2',
            "GAIA-EDR3":'GAIA EDR3',
            "GAIA-DR3":'GAIA DR3',
            "GALEX":'GALEX',
            "HAWKI":'VLT HAWK-I',
            "Tycho":'Hipparcos+Tycho',                     
            "WFPC2":'HST (WFPC2)',
            "ACS":'HST (ACS)',
            "WFC3":'HST (WFC3)',
            "JPLUS":'J-PLUS',
            "Johnson-Cousins":'Johnson-Cousins',
            "JWST_NIRCam_PostLaunch":'JWST (NIRCam Vega ZP)',
            "JWST_NIRCam_zpS":'JWST (NIRCam Sirio ZP)',
            "JWST_NIRISS":'JWST (NIRISS)',
            "Kepler":'Kepler',
            "PanSTARSS1":'PanSTARSS1',
            "Roman":'Roman',
            "SAGE":'SAGE',
            "SkyMapper":'SkyMapper',
            "Sloan":'Sloan',
            "Spitzer_IRAC":'Spitzer (IRAC)',
            "Stromgren":'Strömgren',
            "Subaru_HSC":'Subaru (HSC)',
            "SWIFT_UVOT":'SWIFT (UVOT)',
            "TESS":'TESS',
            "UVIT":'UVIT (FUV+NUV+VIS)',                    
            "LSST":'Vera C. Rubin Obs. (LSST)',
            "VISTA":'VISTA',
            "WFIRST":'WFIRST (WFI)',
            "WISE":'WISE',
        }

    def get_library_name(self):
        return 'BaSTI'
    
    def load_site(self):
        self.browser.get('http://basti-iac.oa-abruzzo.inaf.it/isocs.html')

    def convert_ages(self, agemin, agemax, dage):
        # from Gyr (input) to Myr (required on website)
        return agemin*1e3, agemax*1e3, dage*1e3

    def set_photometry(self):

        if self.photometry not in self.photometric_systems:
            raise ValueError(f"Photometry must be one of:\n{
                '\n'.join(self.photometric_systems.keys())}." +\
                '\nCheck https://basti-iac.oa-abruzzo.inaf.it/isocs.html for details.'
                )
        
        self.browser.find_element(By.XPATH,
            "//select[@name='bcsel' and @id='text-bolcor']/option[@value=\"" + self.photometry + "\"]"
        ).click()

    def set_metallicity(self):
        feh_input = self.browser.find_element(By.NAME, "imetalh")
        feh_input.clear()
        feh_input.send_keys(str(self.FeH))

    def set_alpha_and_grid(self):
        # Select alpha and grid 
        alpha = self.extra.get("alpha")
        grid = self.extra.get("grid")

        if not alpha or not grid:
            raise ValueError("Parameters 'alpha' and 'grid' must be provided.")

        if alpha not in ['depleted', 'solar-scaled', 'enhanced']:
            raise ValueError("Parameter 'alpha' must be one of: 'depleted', 'solar-scaled', 'enhanced'.")
        alpha_value = {
            'depleted': 'M02',
            'solar-scaled': 'P00',
            'enhanced': 'P04'
            }[alpha]
        self.browser.find_element(By.XPATH,
            "//select[@name='alpha' and @id='text-alpha']/option[@value='" + alpha_value + "']"
        ).click()
        time.sleep(1)

        grids = {
            'depleted': ['M02O1D1E1Y247'],
            'solar-scaled': ['P00O0D0E0Y247', 'P00O1D1E1Y247', 'P00O1D0E0Y247', 'P00O1D0E1Y247'],
            'enhanced': ['P04O0D0E0Y247', 'P04O1D1E1Y247', 'P04O1D0E0Y247', 'P04O1D0E1Y247']
        }
        grid_codes = {
            'O0D0E0Y247': 'O0D0E0Y247 - Overshooting: No, Diffusion: No, Mass loss: η = 0.0, He = 0.247',
            'O1D1E1Y247': 'O1D1E1Y247 - Overshooting: Yes, Diffusion: Yes, Mass loss: η = 0.3, He = 0.247',
            'O1D0E0Y247': 'O1D0E0Y247 - Overshooting: Yes, Diffusion: No, Mass loss: η = 0.0, He = 0.247',
            'O1D0E1Y247': 'O1D0E1Y247 - Overshooting: Yes, Diffusion: No, Mass loss: η = 0.3, He = 0.247'
        }

        if grid not in grids[alpha]:
            raise ValueError(f"Parameter 'grid' must be one of:\n {'\n'.join([alpha_value + grid_codes[el[3:]] for el in grids[alpha]])}.")

        self.browser.find_element(By.XPATH,
            "//select[@name='grid' and @id='text-grid']/option[@value='" + grid + "']"
        ).click()

    def set_ages(self, age_start, age_end, age_step):

        age_start_myr, age_end_myr, dage_myr = self.convert_ages(age_start, age_end, age_step)
        age_input = self.browser.find_element(By.NAME, "iage")
        age_input.clear()
        age_input.send_keys(f"{age_start_myr}--{age_end_myr},{dage_myr}")
        time.sleep(0.5)

    def submit(self):
        self.browser.find_element(By.XPATH, "//input[@value='Submit']").click()

    def get_download_link(self):

        def find_link():
            link = self.browser.find_element(By.XPATH,'//a[contains(@href,".tar.gz")]').get_attribute("href")
            self.logger.debug("Retrieved isochrone link: " + link)
            return link
        
        download_link = retry_action(
            action=find_link,
            attempts=48,
            delay=10,
            logger=self.logger,
            failure_msg="Cannot get isochrone. Slow internet or interpolation problem."
        )
        return download_link
    
    def download(self, download_link):
        self.browser.back()
        self.logger.debug("Browser set back to the previous page")

        filename = self.dir_out / "iso.tar.gz"
        self.logger.debug("Downloading isochrone tar.gz file...")

        response = requests.get(download_link, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        self.logger.debug("Isochrone iso.tar.gz file downloaded successfully.")

        return filename

    def unpack_and_organize(self):
        
        download_link = self.get_download_link()

        if download_link:
            filename = self.download(download_link)
            handler = BastiIsochroneHandler(filepath=filename, dir_out=self.dir_out, feh=self.FeH, logger=self.logger)
            handler.unpack_and_organize()


class MistForm(BaseIsochroneForm):

    def __init__(self, browser, **kwargs):
        super().__init__(browser, **kwargs)

        self.photometric_systems = {
          "CFHTugriz":'CFHT/MegaCam',
          "DECam":'DECam',
          "HST_ACSHR":'HST ACS/HRC',
          "HST_ACSWF":'HST ACS/WFC',
          "HST_WFC3":'HST WFC3/UVIS+IR',
          "HST_WFPC2":'HST WFPC2',
          "IPHAS":'INT / IPHAS',
          "GALEX":'GALEX',
          "JWST":'JWST',
          "LSST":'LSST',
          "PanSTARRS":'PanSTARRS',
          "SDSSugriz":'SDSS',
          "SkyMapper":'SkyMapper',
          "SPITZER":'Spitzer IRAC',
          "SPLUS":'S-PLUS',
          "HSC":'Subaru Hyper Suprime-Cam',
          "Swift":'Swift',
          "UBVRIplus":'UBV(RI)c + 2MASS + Kepler + Hipparcos + Gaia (DR2/MAW/EDR3) + Tess',
          "UKIDSS":'UKIDSS',
          "UVIT":'UVIT',
          "VISTA":'VISTA',
          "WashDDOuvby":'Washington + Strömgren + DDO51',
          "WFIRST":'WFIRST (preliminary)',
          "WISE":'WISE'
        }

    def get_library_name(self):
        return 'MIST'

    def load_site(self):
        self.browser.get('https://waps.cfa.harvard.edu/MIST/interp_isos.html')

    def convert_ages(self, agemin, agemax, dage):
        # from Gyr (input) to yr (required by website)
        return agemin*1e9, agemax*1e9, dage*1e9

    def set_photometry(self):
        if self.photometry not in self.photometric_systems:
            raise ValueError(f"Photometry must be one of:\n{
                '\n'.join(self.photometric_systems.keys())}." +\
                '\nCheck https://waps.cfa.harvard.edu/MIST/resources.html for details.'
                )
        
        output_photometry = self.browser.find_element(By.XPATH,
            ".//input[@type='radio' and @name='output_option' and @value='photometry']")
        output_photometry_type = self.browser.find_element(By.NAME,"output")
        output_photometry_type.send_keys(self.photometry)
        output_photometry.click()

    def set_metallicity(self):

        feh = self.browser.find_element(By.NAME,"FeH_value")
        feh.clear()
        feh.send_keys(str(self.FeH))

    def set_ages(self, age_start, age_end, age_step):

        age_start_yr, age_end_yr, age_step_yr = self.convert_ages(age_start, age_end, age_step)

        # Select linear age scale 
        self.browser.find_element(By.NAME,"age_scale").click()
        self.browser.find_element(By.XPATH,".//input[@type='radio' and @name='age_type' and @value='range']").click()
                                  
        # Give age range and step
        agemin = self.browser.find_element(By.NAME,"age_range_low")
        agemin.clear()
        agemin.send_keys(str(age_start_yr))

        agemax = self.browser.find_element(By.NAME,"age_range_high")
        agemax.clear()
        agemax.send_keys(str(age_end_yr))
        
        agestep = self.browser.find_element(By.NAME,"age_range_delta")
        agestep.clear()
        agestep.send_keys(str(age_step_yr))

        time.sleep(0.5)

    def submit(self):
        self.browser.find_element(By.XPATH,".//button[@type='submit']").click()

    def get_download_link(self):

        def find_link():
            link = self.browser.find_element(By.XPATH,'//a[contains(@href,".zip")]').get_attribute("href")
            self.logger.debug("Retrieved isochrone link: " + link)
            return link
        
        download_link = retry_action(
            action=find_link,
            attempts=30,
            delay=10,
            logger=self.logger,
            failure_msg="Cannot get isochrone. Slow internet or interpolation problem."
        )
        return download_link

    def download(self, download_link):
        
        self.browser.back()
        self.logger.debug("Browser set back to the previous page")
    
        #download_link.click()
        #linkname = download_link.get_attribute("href")
        filename = self.dir_out / download_link[::-1][:download_link[::-1].index('/')][::-1]

        self.logger.debug("Downloading isochrone zip file...")
        response = requests.get(download_link, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        self.logger.debug("Isochrone zip file downloaded successfully.")        

        return filename

    def unpack_and_organize(self):
        
        download_link = self.get_download_link()

        if download_link:
            filename = self.download(download_link)
            handler = MistIsochroneHandler(filepath=filename, dir_out=self.dir_out, feh=self.FeH, logger=self.logger)
            handler.unpack_and_organize()   
                    

class ParsecForm(BaseIsochroneForm):
    def __init__(self, browser, **kwargs):
        super().__init__(browser, **kwargs)

        self.photometric_systems = {
            "2MASS + Spitzer (IRAC+MIPS)" : "YBC_tab_mag_odfnew/tab_mag_2mass_spitzer.dat",
            "2MASS + Spitzer (IRAC+MIPS) + WISE" : "YBC_tab_mag_odfnew/tab_mag_2mass_spitzer_wise.dat",
            "2MASS JHKs" : "YBC_tab_mag_odfnew/tab_mag_2mass.dat",
            "OGLE + 2MASS + Spitzer (IRAC+MIPS)" : "YBC_tab_mag_odfnew/tab_mag_ogle_2mass_spitzer.dat",
            "UBVRIJHK (cf. Maiz-Apellaniz 2006 + Bessell 1990)" : "YBC_tab_mag_odfnew/tab_mag_ubvrijhk.dat",
            "UBVRIJHKLMN (cf. Bessell 1990 + Bessell &amp; Brett 1988)" : "YBC_tab_mag_odfnew/tab_mag_bessell.dat",
            "AKARI" : "YBC_tab_mag_odfnew/tab_mag_akari.dat",
            "BATC" : "YBC_tab_mag_odfnew/tab_mag_batc.dat",
            "CFHT Megacam + Wircam (all ABmags)" : "YBC_tab_mag_odfnew/tab_mag_megacam_wircam.dat",
            "CFHT Wircam" : "YBC_tab_mag_odfnew/tab_mag_wircam.dat",
            "CFHT/Megacam post-2014 u*g'r'i'z'" : "YBC_tab_mag_odfnew/tab_mag_megacam_post2014.dat",
            "CFHT/Megacam pre-2014 u*g'r'i'z'" : "YBC_tab_mag_odfnew/tab_mag_megacam.dat",
            "CIBER" : "YBC_tab_mag_odfnew/tab_mag_ciber.dat",
            "CLUE + GALEX (Vegamags)" : "YBC_tab_mag_odfnew/tab_mag_clue_galex.dat",
            "CSST (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_CSST.dat",
            "DECAM (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_decam.dat",
            "DENIS" : "YBC_tab_mag_odfnew/tab_mag_denis.dat",
            "DMC 14 filters" : "YBC_tab_mag_odfnew/tab_mag_dmc14.dat",
            "DMC 15 filters" : "YBC_tab_mag_odfnew/tab_mag_dmc15.dat",
            "ESO/EIS (WFI UBVRIZ + SOFI JHK)" : "YBC_tab_mag_odfnew/tab_mag_eis.dat",
            "ESO/WFI" : "YBC_tab_mag_odfnew/tab_mag_wfi.dat",
            "ESO/WFI2" : "YBC_tab_mag_odfnew/tab_mag_wfi2.dat",
            "Euclid VIS+NISP (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_euclid_nisp.dat",
            "GALEX FUV+NUV (Vegamag) + SDSS ugriz (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_galex_sloan.dat",
            "GALEX FUV+NUV + Johnson's UBV (Maiz-Apellaniz version), all Vegamags" : "YBC_tab_mag_odfnew/tab_mag_galex.dat",
            "Gaia DR1 + Tycho2 + 2MASS (all Vegamags)" : "YBC_tab_mag_odfnew/tab_mag_gaia_tycho2_2mass.dat",
            "Gaia DR2 + Tycho2 + 2MASS (all Vegamags, Gaia passbands from Evans et al. 2018)" : "YBC_tab_mag_odfnew/tab_mag_gaiaDR2_tycho2_2mass.dat",
            "Gaia DR2 + Tycho2 + 2MASS (all Vegamags, Gaia passbands from Weiler 2018)" : "YBC_tab_mag_odfnew/tab_mag_gaiaDR2weiler_tycho2_2mass.dat",
            "Gaia EDR3 (all Vegamags, Gaia passbands from ESA/Gaia website)" : "YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat",
            "Gaia's DR1 G, G_BP and G_RP (Vegamags)" : "YBC_tab_mag_odfnew/tab_mag_gaia.dat",
            "Gaia's DR2 G, G_BP and G_RP (Vegamags, Gaia passbands from Evans et al. 2018)" : "YBC_tab_mag_odfnew/tab_mag_gaiaDR2.dat",
            "Gaia's DR2 G, G_BP and G_RP (Vegamags, Gaia passbands from Maiz-Apellaniz and Weiler 2018)" : "YBC_tab_mag_odfnew/tab_mag_gaiaDR2maiz.dat",
            "Gaia's DR2 G, G_BP and G_RP (Vegamags, Gaia passbands from Weiler 2018)" : "YBC_tab_mag_odfnew/tab_mag_gaiaDR2weiler.dat",
            "HST+GALEX+Swift/UVOT UV filters" : "YBC_tab_mag_odfnew/tab_mag_UVbright.dat",
            "HST/ACS HRC" : "YBC_tab_mag_odfnew/tab_mag_acs_hrc.dat",
            "HST/ACS WFC (c.f. 2007 revision, pos-04jul06)" : "YBC_tab_mag_odfnew/tab_mag_acs_wfc_pos04jul06.dat",
            "HST/ACS WFC - updated filters and zeropoints, 2021" : "YBC_tab_mag_odfnew/tab_mag_acs_wfc_202101.dat",
            "HST/NICMOS AB" : "YBC_tab_mag_odfnew/tab_mag_nicmosab.dat",
            "HST/NICMOS Vega" : "YBC_tab_mag_odfnew/tab_mag_nicmosvega.dat",
            "HST/STIS imaging mode, Vegamag" : "YBC_tab_mag_odfnew/tab_mag_stis.dat",
            "HST/WFC3 all W+LP+X filters (UVIS1+IR, final throughputs)" : "YBC_tab_mag_odfnew/tab_mag_wfc3_wideverywide.dat",
            "HST/WFC3 long-pass and extremely wide filters (UVIS) - updated filters and zeropoints, 2021" : "YBC_tab_mag_odfnew/tab_mag_wfc3_202101_verywide.dat",
            "HST/WFC3 medium filters (UVIS+IR) - updated filters and zeropoints, 2021" : "YBC_tab_mag_odfnew/tab_mag_wfc3_202101_medium.dat",
            "HST/WFC3 wide filters (UVIS+IR) - updated filters and zeropoints, 2021" : "YBC_tab_mag_odfnew/tab_mag_wfc3_202101_wide.dat",
            "HST/WFPC2 (Vegamag, cf. Holtzman et al. 1995)" : "YBC_tab_mag_odfnew/tab_mag_wfpc2.dat",
            "Hipparcos+Tycho+Gaia DR1 (Vegamags)" : "YBC_tab_mag_odfnew/tab_mag_hipparcos.dat",
            "INT/WFC (Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_int_wfc.dat",
            "IPHAS" : "YBC_tab_mag_odfnew/tab_mag_iphas.dat",
            "JWST MIRI wide filters, Vegamags" : "YBC_tab_mag_odfnew/tab_mag_jwst_miri_wide.dat",
            "JWST NIRCam narrow filters, Vegamags" : "tab_mag_odfnew/tab_mag_jwst_narrow.dat",
            "JWST NIRCam wide+verywide filters, Vegamags" : "YBC_tab_mag_odfnew/tab_mag_jwst_nircam_wide.dat",
            "JWST NIRCam wide+verywide+medium filters (Nov 2022), Vegamags" : "tab_mag_odfnew/tab_mag_jwst_nircam_widemedium_nov22.dat",
            "JWST NIRCam wide+verywide+medium filters, Vegamags" : "YBC_tab_mag_odfnew/tab_mag_jwst_nircam_widemedium.dat",
            "JWST NIRISS filters (Nov 2022), Vegamags" : "tab_mag_odfnew/tab_mag_jwst_niriss_nov22.dat",
            "JWST Nirspec filters, Vegamags" : "YBC_tab_mag_odfnew/tab_mag_jwst_nirspec.dat",
            "JWST custom, Vegamags" : "tab_mag_odfnew/tab_mag_jwst_fnl.dat",
            "Kepler + SDSS griz + DDO51 (in ABmags)" : "YBC_tab_mag_odfnew/tab_mag_kepler.dat",
            "Kepler + SDSS griz + DDO51 (in ABmags) + 2MASS (~Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_kepler_2mass.dat",
            "KiDS/VIKING (VST/OMEGAM + VISTA/VIRCAM, all ABmags)" : "YBC_tab_mag_odfnew/tab_mag_vst_vista.dat",
            "LBT/LBC (Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_lbt_lbc.dat",
            "LSST (ABmags) + WFIRST proposed filters (Vegamags)" : "YBC_tab_mag_odfnew/tab_mag_lsst_wfirst_proposed2017.dat",
            "LSST ugrizY, March 2012 total filter throughputs (all ABmags)" : "YBC_tab_mag_odfnew/tab_mag_lsst.dat",
            "LSST ugrizy, Oct 2017 total filter throughputs for DP0 (all ABmags)" : "tab_mag_odfnew/tab_mag_lsstDP0.dat",
            "LSST ugrizy, Sept 2023, total filter throughputs R1.9 (all ABmags)" : "tab_mag_odfnew/tab_mag_lsstR1.9.dat",
            "NOAO/CTIO/MOSAIC2 (Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_noao_ctio_mosaic2.dat",
            "OGLE-II" : "YBC_tab_mag_odfnew/tab_mag_ogle.dat",
            "PLATO plus some other filters" : "YBC_tab_mag_odfnew/tab_mag_PLATO.dat",
            "Pan-STARRS1" : "YBC_tab_mag_odfnew/tab_mag_panstarrs1.dat",
            "Roman (ex-WFIRST) 2021 filters, Vegamags" : "tab_mag_odfnew/tab_mag_Roman2021.dat",
            "S-PLUS (Vegamags), revised on Nov. 2017" : "YBC_tab_mag_odfnew/tab_mag_splus.dat",
            "SDSS ugriz" : "YBC_tab_mag_odfnew/tab_mag_sloan.dat",
            "SDSS ugriz + 2MASS JHKs" : "YBC_tab_mag_odfnew/tab_mag_sloan_2mass.dat",
            "SDSS ugriz + UKIDSS ZYJHK" : "YBC_tab_mag_odfnew/tab_mag_sloan_ukidss.dat",
            "SWIFT/UVOT UVW2, UVM2, UVW1,u (Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_swift_uvot.dat",
            "SkyMapper (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_skymapper.dat",
            "Spitzer IRAC+MIPS" : "YBC_tab_mag_odfnew/tab_mag_spitzer.dat",
            "Stroemgren-Crawford" : "YBC_tab_mag_odfnew/tab_mag_stroemgren.dat",
            "Subaru/Hyper Suprime-Cam (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_hsc.dat",
            "Subaru/Suprime-Cam (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_suprimecam.dat",
            "SuperBIT, all ABmags" : "tab_mag_odfnew/tab_mag_SuperBIT.dat",
            "TESS + 2MASS (Vegamags)" : "YBC_tab_mag_odfnew/tab_mag_TESS_2mass.dat",
            "TESS + 2MASS (Vegamags) + Kepler + SDSS griz + DDO51 (in ABmags)" : "YBC_tab_mag_odfnew/tab_mag_TESS_2mass_kepler.dat",
            "UKIDSS ZYJHK (Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_ukidss.dat",
            "UNIONS (CFHT/Megacam u+r, Subaru/HSC g+z, Pan-STARRS i+z, all ABmags)" : "tab_mag_odfnew/tab_mag_unions.dat",
            "UVIT (all ABmags)" : "YBC_tab_mag_odfnew/tab_mag_uvit.dat",
            "VISIR" : "YBC_tab_mag_odfnew/tab_mag_visir.dat",
            "VISTA ZYJHKs (Vegamag)" : "YBC_tab_mag_odfnew/tab_mag_vista.dat",
            "VPHAS+ (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_vphas.dat",
            "VST/OMEGACAM (ABmag)" : "YBC_tab_mag_odfnew/tab_mag_vst_omegacam.dat",
            "Vilnius" : "YBC_tab_mag_odfnew/tab_mag_vilnius.dat",
            "WFC3/UVIS around CaHK" : "YBC_tab_mag_odfnew/tab_mag_wfc3_uvisCaHK.dat",
            "Washington CMT1T2 + DDO51" : "YBC_tab_mag_odfnew/tab_mag_washington_ddo51.dat",
            "ZTF (ABmags)" : "YBC_tab_mag_odfnew/tab_mag_ztf.dat",
            "deltaa (Paunzen) + UBV (Maiz-Apellaniz), in Vegamags" : "YBC_tab_mag_odfnew/tab_mag_deltaa.dat"
        }

    def get_library_name(self):
        return 'PARSEC'

    def load_site(self):
        self.browser.get('https://stev.oapd.inaf.it/cgi-bin/cmd')

    def convert_ages(self, agemin, agemax, dage):
        # from Gyr (input) to yr (required by website)
        return agemin*1e9, agemax*1e9, dage*1e9
    
    def set_photometry(self):

        if self.photometry not in self.photometric_systems.keys():
            raise ValueError(f"Photometry must be one of:\n{
                '\n'.join(self.photometric_systems.keys())}." +\
                '\nCheck https://stev.oapd.inaf.it/cmd_3.8/photsys.html for details.'
                )

        self.browser.find_element(By.XPATH,
            "//select[@name='photsys_file']/option[@value='" + self.photometric_systems[self.photometry] + "']"
            ).click()
        
    def set_metallicity(self):

        self.browser.find_element(By.XPATH,".//input[@type='radio' and @name='isoc_ismetlog' and @value='1']").click()

        fehmin = self.browser.find_element(By.NAME,"isoc_metlow")
        fehmin.clear()
        fehmin.send_keys(str(self.FeH))

        fehstep = self.browser.find_element(By.NAME,"isoc_dmet")
        fehstep.clear()
        fehstep.send_keys('0.0')
        
    
    def set_ages(self, age_start, age_end, age_step):
        
        age_start_yr, age_end_yr, age_step_yr = self.convert_ages(age_start, age_end, age_step)

        agemin = self.browser.find_element(By.NAME,"isoc_agelow")
        agemin.clear()
        agemin.send_keys(str(age_start_yr))

        agemax = self.browser.find_element(By.NAME,"isoc_ageupp")
        agemax.clear()
        agemax.send_keys(str(age_end_yr))

        agestep = self.browser.find_element(By.NAME,"isoc_dage")
        agestep.clear()
        agestep.send_keys(str(age_step_yr))

    def submit(self):
        button = self.browser.find_element(By.NAME,"submit_form")
        button.click()

    def get_download_link(self):

        def find_link():
            link = self.browser.find_element(By.XPATH,'//a[contains(@href,".dat")]')
            self.logger.debug("Retrieved isochrone link: " + link.get_attribute('href'))
            return link
        
        download_link = retry_action(
            action=find_link,
            attempts=30,
            delay=10,
            logger=self.logger,
            failure_msg="Cannot get isochrone. Slow internet or interpolation problem."
        )
        return download_link

    def download(self, download_link):
        
        target_dir = configure_target_dir(self.dir_out, self.FeH, logger=self.logger)

        download_link.click()

        # Get only the plain text (without HTML tags)
        data_text = self.browser.find_element(By.TAG_NAME, 'pre').text  # Most .dat files render inside <pre>

        # Save to file
        self.logger.debug("Downloading isochrone file...")
        filename = target_dir / ''.join(('iso_feh',str(self.FeH),'.txt'))
        with open(filename, "w", encoding="utf-8") as f:
            f.write(data_text)
        self.logger.debug("Isochrone file downloaded successfully.")

        self.browser.back()
        self.browser.back()
        self.logger.debug("Browser set back to the previous page")

        return filename
    
    def unpack_and_organize(self):
        
        download_link = self.get_download_link()
        if download_link:
            self.download(download_link)

