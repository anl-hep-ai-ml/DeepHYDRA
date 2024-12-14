# DeepHydra Merlin Setup Guide
This guide provides step-by-step instructions for setting up and running the `Merlin` environment in the `DeepHydra` project.
---
## **1. Replace `merlin_conda_env.txt`**
### Replace the Intel channel packages in the `merlin_conda_env.txt` file with equivalent packages from public channels. Convert the dependencies into a `.yml` file (e.g., `merlin_conda_env.yml`).
--- 
## **2. Create the Conda Environment**
```bash line start
conda env create -f /path/to/merlin_conda_env.yml --name merlin_env
```
### Run above command to create the Conda environment
---
## **3. Verify the Environment**
```bash line start
conda env list
```
### Verify the env
---
## **4. Activate the created environment:**
```bash line start
conda activate merline_env
```
### Activate the created env
---
## **5. Navigate to the DeepHydra Directory:**

```bash line start
cd DeepHydra/baselines/merlin
```
### Change to the DeepHydra/baselines/merlin directory
---
## **6. Grant Execute Permission:**
```bash line start
chmod +x ./run_merlin.s
```
### Make the run_merlin.sh script executable
---

## **7. Download Merlin Library**
### Refer to the instructions in the Py-Merlin GitLab repository: https://gitlab.com/dlr-dw/py-merlin  for the Standalone part. Build and install this package inside the Merlin environment. 

## **8.Return to the Parent Directory**
```bash line start
cd ..
```
### cd back to DeepHydra/baselines/merlin
---
## **9.Skip Steps**
### If you copied my branch JJ with all fixes applied, you can skip steps 10 to 13. But still need to change all the output path for merlin_hlt_datasets.py and merlin_hlt_datasets_reduced.py
---
## **10.Fixed the typo in run_merlin.sh**
### change to " python3 merlin_hlt_datasets.py " instead of python3 run_merlin_hlt_datasets.py 
### change to " python3 merlin_hlt_datasets_reduced.py" instead of python3 run_merlin_hlt_datasets_reduced.py
---
## **11.Fixed the complainted**
### Got the complained about the pylikwid("Like I Knew What I'm Doing"), so I comment it out all(comment out inside the function  ofget_merlin_flops at line 244) since it is used for analyze CPU performance, memory usage, and optimize code for high-performance computing.

--- 

## **12.Fixed the complainted**
### For file "merlin_hlt_datasets.py", line 447, in <module>  hlt_data_pd.iloc[run_endpoints[-2]:-1, index 3685 is out of bounds for axis 1 with size 2622, fixed this problem by: Filter Out Invalid Indices Check if the indices in channels_to_delete_last_run exist in the DataFrame and only keep the valid ones: " valid_channels = [ch for ch in channels_to_delete_last_run if ch < hlt_data_pd.shape[1]] " and " hlt_data_pd.iloc[run_endpoints[-2]:-1, valid_channels] = 0 "
---
## **13.Fixed the complainted**
### Fixed the problem for unmatched output from merlin function. For File "merlin_hlt_datasets.py", line 469, in <module> run_merlin(hlt_data_np, discords, distances, lengths, parameters =\    ValueError: not enough values to unpack (expected 4, got 3), I fixed it by removing the parameters, because I think this it isn't important to us: discords, distances, lengths =\

---
## **14. Run Merlin**
```bash line start
./run_merlin.sh > ./merlin_output.txt
```
### Run Merlin
---
## **15.Estimated Runtime**
### Running merlin_hlt_datasets.py takes approximately 23 hours. Running merlin_hlt_datasets_reduced takes around 1 hour.
