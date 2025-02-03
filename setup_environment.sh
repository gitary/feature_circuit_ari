#!/bin/bash

if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python3."
    exit
fi

apt-get update && apt-get upgrade

# Install unzip and vim if not already installed
apt-get install -y unzip vim python3.11.5 python3-venv

# Create a virtual environment
python3 -m venv env_fc

# Activate the virtual environment
source env_fc/bin/activate

# Upgrade pip within the virtual environment
pip install --upgrade pip

# Install Jupyter and IPython kernel
pip install jupyter
pip install ipykernel

# Install required packages from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "File requirements.txt not found!"
fi

# Update git submodules
git submodule update --init

# Check if folder dictionary doesn't exist and if the pretrained dictionary downloader script exists and run it
if [ -f "./dictionary_learning/pretrained_dictionary_downloader.sh" ] && [ ! -d "dictionaries/" ]; then
    ./dictionary_learning/pretrained_dictionary_downloader.sh
else
    echo "Script ./dictionary_learning/pretrained_dictionary_downloader.sh not found or dictionaries/ directory already exists!"
fi

# Unzip the downloaded dictionary
if [ -f "dictionaries_pythia-70m-deduped_10.zip" ] && [ ! -d "dictionaries/" ]; then
    unzip dictionaries_pythia-70m-deduped_10.zip
else
    echo "File dictionaries_pythia-70m-deduped_10.zip not found or dictionaries/ directory already exists!"
fi

# Install IPython kernels
python3 -m ipykernel install --user --name=env_fc