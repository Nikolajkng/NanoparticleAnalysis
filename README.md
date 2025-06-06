# NanoparticleAnalysis
Bachelor Project 2025


# Installation
Install pyinstaller with 'pip install pyinstaller'

## Windows:
0. Navigate to root of project folder
1. Run the following: 'pyinstaller --noconfirm --noconsole --name NanoAnalyzer  --add-data "src/data/model/UNet_best_06-06.pt;src/data/model" main.py '
2. If you do not have a pre-trained model, do not run with the --add-data argument
3. To run the application, navigate to dist folder and execute/open: 'main.exe'

## Linux (Ubuntu): 
0. Navigate to root of project folder
1. Give build script exec. permission: 'sudo chmod +x build_app_mkl.sh'
2. Run the script 'build_app_mkl.sh'
3. To run the application, navigate to dist folder and execute: './NP_segmenter_linux'


## MacOS: 
0. Navigate to root of project folder
1. Give build script exec. permission: 'sudo chmod +x build_app_mac.sh'
2. Run the script 'build_app_mac.sh'
3. To run the application, navigate to dist folder and execute: './NP'
