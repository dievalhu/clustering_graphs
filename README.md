# **G-CLUS**  
**Graph-Based Clustering Algorithm with Size Constraints**  

## _**Description**_  
This project aims to develop a graph-based clustering algorithm that applies specific size constraints to the formed clusters. This approach is relevant for practical applications where fairness and effective resource management are crucial, such as resource allocation, team formation, event organization, and social marketing.  

## _**Dependencies**_  
Libraries: networkx, matplotlib, scikit-learn, pymetis, pyvis.  

## _**Installation**_  
1. Install [Miniconda3](https://docs.anaconda.com/miniconda/). During installation, check the option to add conda to the PATH; installation for Windows systems.  
2. Install Miniconda3 for Linux systems using the following commands:  
   ```sh
   apt update -y
   apt upgrade -y
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/miniconda3/bin/activate
   conda init
   ```
3. Verify the correct installation of Miniconda3.  
   ```sh
   conda --version
   ```
4. Open the terminal in the project's root folder and run the following commands:  
   ```sh
   pip install networkx
   pip install matplotlib
   pip install scikit-learn
   pip install pyvis
   conda update conda
   conda install conda-forge::pymetis
   ```
## _**Algorithm Testing**_  
1. From the root folder, the clustering algorithm can be tested on both Windows and Linux systems using:  
   ```sh
   python test.py
   ```