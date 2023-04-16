#!/bin/bash
# This script downloads the datasets from an repository.

# create a directory for the data.
mkdir ./data
cd ./data
mkdir ./PEMS04
mkdir ./PEMS08

# download PeMSD4.
cd ./PEMS04
wget https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz
cd ..

# download PeMSD8.
cd ./PEMS08
wget https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS08/pems08.npz
