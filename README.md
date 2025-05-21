# Instructions
## How to set up:
1. Download code files (`main.py`, `assignment.py` and `testing.py`) and image files into a folder.
2. Create and activate conda environment if you have not.
```bash
conda create -n conda_env_name
conda activate conda_env_name
```
3. Install requried libraries (with their dependencies)
- OpenCV
```bash
conda install -c conda-forge opencv
```
- customtkinter
```bash
pip install customtkinter
```

## How to run the code:
1. In terminal, change directory to the folder with the code files and image files.
2. In terminal, run the `main.py` file
```bash
python main.py
```
3. Embed Watermark
- Click the button labelled "Embed Watermark"
- Choose carrier image `camera.png`
- Choose watermark image `penguin.png`
- Choose directory to save the embed image
- See embed watermark results
4. Recover Watermark
- Click the button labelled "Recover Watermark"
- Choose carrier image
- Choose watermark image
- See recover watermark results
5. Detect Tampering
- Click the button labelled "Detect Tampering"
- Choose carrier image
- Choose watermark image
- See detect tampering results

Note: You may have to scroll to see the results.

## How to test the code:
1. Run the code once to get the `WatermarkEncodedImg.png` file
2. Run the `testing.py` file to get the tampered images
```bash
python testing.py
```
3. Detect Tampering where the selected carrier image is
- original carrier image `camera.png` -> tampering detected, all watermark points are marked
- embedded carrier image `WatermarkEncodedImg.png` -> no tampering detected
- resized embedded carrier image `camera_watermarkedPeguin_resized.png` -> tampering detected, some watermark points are marked
- cropped embedded carrier image `camera_watermarkedPeguin_cropped.png` -> tampering detected, corner watermark points are marked
- rotated embedded carrier image `camera_watermarkedPeguin_rotated.png` -> tampering detected, some watermark points are marked

