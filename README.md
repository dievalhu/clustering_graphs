# **G-CLUS**
**Algoritmo de Clustering con Restricciones de Tamaño Basado en Grafos**

## _**Descripción**_
Este proyecto tiene como objetivo desarrollar un algoritmo de clustering basado en grafos que aplique restricciones de tamaño específicas a los clusters formados. Este enfoque es relevante para aplicaciones prácticas donde la equidad y la gestión efectiva de recursos son fundamentales, como en la asignación de recursos, formación de equipos, organización de eventos y marketing social.

## _**Dependencias**_
Librerías: networkx, matplotlib, scikit-learn, pymetis, pyvis.

## _**Instalación**_
1. Instalar [Miniconda3](https://docs.anaconda.com/miniconda/) . Durante la instalación, marcar la opción de agregar conda al PATH; instalación para sistemas Windows.
2. Instalar Miniconda3, para sistemas Linux, usando los siguientes comandos:
   ```sh
   apt update -y
   apt upgrade -y
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/miniconda3/bin/activate
   conda init
    ```
3. Verificar la correcta instalación de Miniconda3.
   ```sh
   conda --version
    ```
4. Abrir la terminal en la carpeta raíz del proyecto y ejecutar los siguientes comandos:
   ```sh
   pip install networkx
   pip install matplotlib
   pip install scikit-learn
   pip install pyvis
   conda update conda
   conda install conda-forge::pymetis
    ```
## _**Prueba del algoritmo**_
1. Desde la carpeta raíz se puede probar, en sistemas Windows y Linux, el algoritmo de clustering con:
   ```sh
   python test.py
    ``` 
