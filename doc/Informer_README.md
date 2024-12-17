# DeepHydra Informer Setup Guide
---
## 1. Replace Packages in `informers_conda_env.txt`
### Replace all packages sourced from the Intel channel in `informers_conda_env.txt` with equivalent packages from public channels (e.g., `defaults`, `conda-forge`, `pytorch`). Once done, convert `informers_conda_env.txt` into `informers_conda_env.yml`.
#### **Note:** Although ideally we could merge the packages from `informers_python_requirements.txt` into a single `informer_env.yml` file, keeping them separate helps avoid extremely long environment build times.
---

## 2. Update Your Conda Configuration
```bash
vim ~/.condarc
```
### Add the following lines to your .condarc:
```bash
channels:
   - defaults
   - conda-forge
   - pytorch
show_channel_urls: true
ssl_verify: false
restore_free_channel: false
remote_connect_timeout_secs: 60.0
remote_read_timeout_secs: 120.0
```
                          
## **3. Create the Conda Environment**
```bash line start
conda env create -f /path/to/DeepHydra/envs/informer_conda_env.yml --name informer_conda_env
```
#### Running above line could have the problem like "CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-11.3.1-h2bc3f7f_2.conda> Elapsed: - An HTTP error occurred when trying to retrieve this URL. HTTP errors are often intermittent, and a simple retry will get you on your way. CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-11.3.1-h2bc3f7f_2.conda> Elapsed: - An HTTP error occurred when trying to retrieve this URL. HTTP errors are often intermittent, and a simple retry will get you on your way."  
#### This error typically indicates a network-related issue. Common causes include intermittent connectivity problems, firewalls blocking access, or SSL certificate verification issues. Just try the following code (might need to do multiple times):
```bash line start
conda clean --all
conda env create -f /path/to/DeepHydra/envs/informer_conda_env.yml --name informer_conda_env
```
---
## **4. Activate the created environment:**
```bash line start
conda activate informer_conda_env
```
### Activate the created env
---
## **5. Download the python requirements**
```bash line start
pip install -r  /path/to/DeepHydra/envs/informers_python_requirements.txt
```
## **6. Navigate to the DeepHydra informer working notebook:**
```bash line start
cd DeepHydra/analysis_scripts
```
---
## **7.Open jupyter notebook **
```bash line start
jupyter notebook 
```
### Find the Strada_train.ipynb and edit it to run informer!
