# Tools for detecting SVs in Hi-C using machine learning.
## Installation and usage
1. Create new pip virtual environment
```bash
python -m venv hict_venv
```
2. Activate new pip virtual environment
Unix/MacOS:
```bash
source hict_venv/bin/activate
```
Windows:
```bash
hict_venv\Scripts\activate
```
3. Install wheel file
```bash
pip install hict_patterns-0.1-py3-none-any.whl
```
4. Check installation
```bash
hict_patterns -h
```
5. Unzip weight.zip to the working directory from which you are going to execute the rest of the commands
Done!

## Script parameters
Main script of project including all modules together can be run as console tool.
```bash
hict_patterns file_path [--search_in_1k] [-B BATCH_SIZE] [--device DEVICE] 
```
### Required input

#### `--file_path` or first argument
Path to HiC file - .mcool format, should have 50Kb, 10Kb, 5Kb resolitions and 1kb resolution if --search_in_1k option used. If haven't file with this resolutions use [cooler_zoomify](https://cooler.readthedocs.io/en/latest/cli.html#cooler-zoomify).

### Optional input

#### `--search_in_1k`
Whether to perform or not detection on 1Kb resolution. Default is not.

#### `-B BATCH_SIZE`
 Size of data batch processed simultaneously by neural network, larger size reduces time of work but requires more RAM and VRAM. Default is 512.

### Output
Output file is a table in .csv format. It consist from 3 columns.
First two is whole genome range coordinates of structural variation. Third is identified сlass.

## Example run and data
You could test work on sample [file](https://niuitmo-my.sharepoint.com/:u:/g/personal/264893_niuitmo_ru/EYSr1RfxP9VHpfrxHqQ51cMBg2Sij1twoj1O9JslkNOERA?e=4FeG2U)
```bash
hict_patterns dong_colluzzii.mcool  --search_in_1k --device auto
```
You should find 3 structural variations in the **``result.csv``** file
