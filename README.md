

## Isotoolkit

### 🚀 Installation

You can install **isotoolkit** easily from source.

#### Standard user installation
```bash
pip install git+https://github.com/askenja/isotoolkit.git
```

#### Installation in the development mode
```bash
git clone https://github.com/askenja/isotoolkit.git
cd isotoolkit
pip install -e .
```

### Usage

The package is designed for automated downloading of [PARSEC](https://stev.oapd.inaf.it/cgi-bin/cmd), [MIST](https://waps.cfa.harvard.edu/MIST/interp_isos.html), and [BaSTI](http://basti-iac.oa-abruzzo.inaf.it/isocs.html) isochrones.  
Each of these stellar evolution libraries offers a wide range of outputs, including theoretical isochrones, synthetic photometry, stellar evolution tracks, luminosity functions, horizontal branch (HB) or white dwarf (WD) models, and asteroseismic parameters.  
This package focuses specifically on retrieving **isochrones with synthetic photometry**, using standard input settings (though some configuration options are available — see below).


#### Step 1: Download and unpack

Regardless of the chosen library, the user must specify the following parameters to retrieve isochrones:

```python
# Metallicity [Fe/H] dex, can be a single value or a 1D array
FeH = [-0.5, -0.2, 0.01] 

# Parameters for the age grid, all values in Gyr
agemin, agemax, dage = 0.05, 13.0, 0.05

# Choose photometric system
photometric_system = 'Gaia EDR3 (all Vegamags, Gaia passbands from ESA/Gaia website)'  # for PARSEC
# photometric_system = 'UBVRIplus'  # for MIST
# photometric_system = 'GAIA-DR3'   # for BaSTI

# Output directory
dir_out = "./isochrones/gaia"
```

**Important:**  
For **PARSEC**, use photometric system names exactly as they appear in the dropdown menu on the website.  
For **MIST** and **BaSTI**, used abbreviated names may differ from those displayed in their menus.  
To check the accepted photometric system names, you can first initialize the retriever (even with an incorrect name for photometry) and then call:

```python
retriever.print_photometric_systems()
```

Next, you need to configure the browser: 
```python
from isotoolkit.loadfuncs import configure_browser

browser = configure_browser(dir_out)
```

From this point onward, the usage may differ slightly depending on the selected stellar evolution library.
 

##### PARSEC

For **PARSEC**, everything is ready, just initialize the retriever and run:

```python
from isotoolkit.loadfuncs import ParsecForm

retriever = ParsecForm(browser,
                       dir_out=dir_out,
                       FeH=FeH,
                       agemin=agemin, 
                       agemax=agemax, 
                       dage=dage,
                       photometry=photometry
                       )
retriever.run()
```

Output will have the following structure:
```
output_base_dir/
├── gaia/
│   ├── iso_feh-1.2/
│   │   └── iso_feh-1.2.txt
│   ├── ...
│   └── iso_feh0.3/
│       └── iso_feh0.3.txt
└── execution_downloading.log
```
For a given metallicity and age range, **PARSEC** generates a single table that includes all specified ages. 
This file is saved in the corresponding metallicity folder and uses the same name as the folder.

Details of the run are stored in a log file, which can be helpful for debugging — it contains more information than what is printed to the terminal.


##### MIST

Like PARSEC, **MIST** also produces one table per metallicity covering the specified age range.  
However, MIST isochrones use a much finer mass resolution and therefore produce significantly more rows per age than PARSEC.

If the age grid has a small step, the interpolator may freeze due to the volume of data.  
To avoid this, large age grids should be split into shorter intervals before submitting a request.  
As a result, metallicity folders may contain multiple output tables.

The maximum length of an age grid that can be submitted in a single request is controlled by the `n_lim` parameter.  
For **MIST**, the recommended value of `n_lim` is **60**.

```python
from isotoolkit.loadfuncs import MistForm

retriever = MistForm(browser,
                     dir_out=dir_out,
                     FeH=FeH,
                     agemin=agemin, 
                     agemax=agemax, 
                     dage=dage,
                     photometry=photometry,
                     n_lim=60
                    )
retriever.run()
```

Output will have the following structure:
```
output_base_dir/
└── gaia/
    ├── iso_feh-1.2/
    │   ├── iso_feh-1.2_p1.txt
    │   ├── iso_feh-1.2_p2.txt
    │   └── ...
    └── iso_feh0.3/
        ├── iso_feh0.3_p1.txt
        ├── iso_feh0.3_p2.txt
        └── ...
```


##### BaSTI

For **BaSTI**, two additional parameters must be specified: **heavy element mixture** and **grid type**.

##### Heavy Element Mixture Options

| Heavy Element Mixture | Code |
|------------------------|------|
| depleted               | M02  |
| solar-scaled           | P00  |
| enhanced               | P04  |

##### Available Grid Configurations

| Grid Description                                                   | Code       |
|--------------------------------------------------------------------|------------|
| Overshooting: No, Diffusion: No, Mass loss: η = 0.0, He = 0.247    | O0D0E0Y247 |
| Overshooting: Yes, Diffusion: Yes, Mass loss: η = 0.3, He = 0.247  | O1D1E1Y247 |
| Overshooting: Yes, Diffusion: No, Mass loss: η = 0.0, He = 0.247   | O1D0E0Y247 |
| Overshooting: Yes, Diffusion: No, Mass loss: η = 0.3, He = 0.247   | O1D0E1Y247 |     

##### Alpha Mixture and Grid Compatibility

The possible combinations of heavy-element mixture and grid codes are:

| Heavy Element Mixture | O0D0E0Y247 | O1D1E1Y247 | O1D0E0Y247 | O1D0E1Y247 |
|------------------------|------------|------------|------------|------------|
| depleted               | –          | ✅         | –          | –          |
| solar-scaled           | ✅         | ✅         | ✅         | ✅         |
| enhanced               | ✅         | ✅         | ✅         | ✅         |

Thus, use valid combinations like:

- `alpha = 'depleted'`, `grid = 'M02O1D1E1Y247'`
- `alpha = 'solar-scaled'`, `grid = 'P00O1D0E0Y247'`

---

**Note:**  
**BaSTI** interpolator saves single-age isochrones in separate files. Therefore, each metallicity folder will contain as many files as there are age values in the grid.

However, for large age grids, the interpolator may fail. It's recommended to split fine-step age grids into shorter intervals using the `n_lim` parameter.  For **BaSTI**, the recommended value of `n_lim` is **100**. 

```python
from isotoolkit.loadfuncs import BastiForm

retriever = BastiForm(browser,
                      dir_out=dir_out,
                      FeH=FeH,
                      agemin=agemin, 
                      agemax=agemax, 
                      dage=dage,
                      photometry=photometry,
                      n_lim=100
                     )
retriever.run()
```

After downloading, the isochrones will be organized in the following structure:
```
output_base_dir/
└── gaia/
    ├── iso_feh-1.2/
    │   ├── 50z0009900y248P00O1D1E1.isc_gaia-dr3
    │   ├── 100z0009900y248P00O1D1E1.isc_gaia-dr3
    │   └── ...
    └── iso_feh0.3/
        ├── 50z0282896y284P00O1D1E1.isc_gaia-dr3
        ├── 100z0282896y284P00O1D1E1.isc_gaia-dr3
        └── ...
```

#### Step 2: Formatting and column extraction

Additional formatting operations can be performed on the downloaded isochrones. After formatting, isochrones from all libraries will be organized in the following way:
- Base output directory contains folder(s) named by the contained photometry and (optionally) folder `Astro`
- Each age is saved in a separate file within the corresponding metallicity folder.

```
output_base_dir/
├── Astro/
│   ├── iso_feh-1.2/
│   │   ├── iso_age12.0.txt
│   │   ├── ...
│   │   └── iso_age0.5.txt
│   ├── ...
│   └── iso_feh0.3/
├── GDR3/
│   ├── ...
└── UBVRIJHK/
    ├── ...
```

By default, astrophysical and photometric columns are stored in separate folders named `Astro` and `[the name of the chosen photometric system]`. An `ID` column is also appended to each isochrone to enable easy matching between files. This approach is especially useful when multiple photometric systems are used in modeling: instead of loading one large isochrone file with all photometric colors, you can load two lightweight files — one with astrophysical parameters and the other with the needed photometry.

This column splitting can be disabled using the `separate_astro` parameter.

The formatted tables contain only selected columns from the original downloaded isochrones. While some commonly used columns are pre-defined in the code, users can also specify custom columns to extract.

Columns currently prescribed for extraction are listed in the table:

| Name             | PARSEC | MIST | BaSTI | Astro/Phot.sys | Description |
|------------------|:------:|:----:|:-----:|----------------|-------------|
| `Mini`           |   +    |  +   |  +    | Astro          | Initial mass (M☉) |
| `Mf`             |   +    |  +   |  +    | Astro          | Final mass (M☉) |
| `logL`           |   +    |  +   |  +    | Astro          | Luminosity log₁₀(L/L☉) |
| `logg`           |   +    |  +   |  +*   | Astro          | Surface gravity log₁₀(cm/s²) |
| `logT`           |   +    |  +   |  +    | Astro          | Surface effective temperature log₁₀(K) |
| `FeHf`           |   -    |  +   |  -    | Astro          | Final metallicity (including diffusion) |
| `phase`          |   +    |  +   |  -    | Astro          | Approximate evolutionary phase (use with caution) |
| `U_Bessell`      |   +    |  +   |  +    | UBVRIJHK       | U mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `B_Bessell`      |   +    |  +   |  +    | UBVRIJHK       | B mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `V_Bessell`      |   +    |  +   |  +    | UBVRIJHK       | V mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `R_Bessell`      |   +    |  +   |  +    | UBVRIJHK       | R mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `I_Bessell`      |   +    |  +   |  +    | UBVRIJHK       | I mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `J_Bessell`      |   +    |  -   |  +    | UBVRIJHK       | J mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `H_Bessell`      |   +    |  -   |  +    | UBVRIJHK       | H mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `K_Bessell`      |   +    |  -   |  +    | UBVRIJHK       | K mag – Maíz-Apellániz (2006), Bessell (1990), Bessell & Brett (1988) |
| `J_2MASS`        |   -    |  +   |  -    | UBVRIJHK       | 2MASS J mag – Cohen et al. (2003) |
| `H_2MASS`        |   -    |  +   |  -    | UBVRIJHK       | 2MASS H mag – Cohen et al. (2003) |
| `Ks_2MASS`       |   -    |  +   |  -    | UBVRIJHK       | 2MASS Ks mag – Cohen et al. (2003) |
| `G_Gaia_DR3`     |   +    |  +   |  +    | GDR3           | Gaia DR3 G mag – Riello et al. (2021), introduced in EDR3 |
| `G_BP_Gaia_DR3`  |   +    |  +   |  +    | GDR3           | Gaia DR3 G<sub>BP</sub> mag – Riello et al. (2021), introduced in EDR3 |
| `G_RP_Gaia_DR3`  |   +    |  +   |  +    | GDR3           | Gaia DR3 G<sub>RP</sub> mag – Riello et al. (2021), introduced in EDR3 |
| `G_RVS_Gaia_DR3` |   -    |  -   |  +    | GDR3           | Gaia DR3 G<sub>RVS</sub> mag |

> **Note**: Surface gravity (`logg`) is not provided in BaSTI, but it is calculated by the code from `Mf`, `logT`, and `logL`.

You can view the columns that are predefined for extraction by calling `formatter.print_column_info()` (see next code example).

Currently, only `GDR3` (Gaia DR3) and `UBVRIJHK` (Bessell UBVRIJHK, with 2MASS JHK for MIST) are the default supported photometric systems for extraction. 

If you have downloaded and want to extract a different photometric system and/or additional columns from the default ones, you need to manually specify the positions of the columns you want to extract, along with the photometric system type they belong to. Refer to the respective stellar evolution library documentation for the full list of available columns and photometries.

The usage of the formatter is similar for all three libraries — just use the corresponding `IsochroneFormatter` class:

```python
import numpy as np
from isotoolkit.formatfuncs import ParsecIsochroneFormatter, MistIsochroneFormatter, BastiIsochroneFormatter

agemin, agemax, dage = 0.05, 13.0, 0.05 # Gyr
age_grid = np.arange(agemin, agemax + dage, dage)

# Specify input data folder and choose another folder for the formatted output
input_base_path = './isochrones'
output_base_path = './isochrones_formatted'

formatter = ParsecIsochroneFormatter(input_base_path, 
                                     output_base_path, 
                                     age_grid,
                                     phot_dirs=['ubv:UBVRIJHK','gaia:GDR3'],
                                     )
formatter.print_column_info()
formatter.run()
```

An important parameter is `phot_dirs`, which specifies which columns to extract. It is a list of elements that map existing input photometric folders (where the downloaded isochrones are stored) to the type of photometry that should be extracted from them. 

The prescribed names for output photometry folders are `'UBVRIJHK'` and `'GDR3'`. If, at the previous step, you downloaded photometry that is not currently supported by the formatter, you can manually specify the columns for extraction using the optional parameter `custom_columns`.

**Example:**  
Suppose your downloaded PARSEC isochrones for two photometric systems: Gaia DR3 and SDSS. They are saved in subfolders named `gaia` and `sdss`, respectively. To extract SDSS photometry (which is not supported by default), you need to:
- Define the SDSS column names and the target folder name in `custom_columns`
- Map input to output in `phot_dirs`

```python
formatter = ParsecIsochroneFormatter(input_base_path, 
                                     output_base_path,
                                     age_grid,
                                     phot_dirs=['gaia:GDR3','sdss:SDSS'],
                                     custom_columns = {'SDSS':{'u':10, 'g':11, 'r':12, 'i':13, 'z':14}}
                                     separate_astro=False,
                                     )
formatter.run()
```

Similarly, you can define additional columns for extraction if something important is missing from the default set. 

For example, if you want to extract two extra astrophysical columns from **PARSEC**, you can do it like this:

```python
formatter = ParsecIsochroneFormatter(input_base_path, 
                                     output_base_path, 
                                     age_grid,
                                     phot_dirs=['ubv:UBVRIJHK'],                                     
                                     custom_columns = {'astro':{'McoreTP':10,'pmode':17}},
                                     #custom_columns = {'UBVRIJHK':{'McoreTP':10,'pmode':17}}, # same but for separate_astro=False
                                     )
formatter.run()
```

And if downloaded isochrones contain synthetic magnitudes for several photometric systems, you can apply extraction 
multiple times to the same data folder:
```python
formatter = MistIsochroneFormatter(input_base_path, 
                                   output_base_path, 
                                   age_grid,
                                   phot_dirs=['ubvplus:UBVRIJHK','ubvplus:GDR3','ubvplus:Tycho'],
                                   custom_columns={'Tycho':{'Tycho_B':20,'Tycho_V':21}}
                                   separate_astro=False,
                                   )
formatter.run()
```

### License

This code is published under the MIT License. If you use this code in your research, please include a reference to this repository.


### Feedback and Contact

If you have comments, questions, or experience problems using **isotoolkit**, you can reach the developer via email <k.sysoliatina@gmail.com>. 

If you find a bug, please raise an issue on GitHub.
