# Using Machine Learning in Veterinary Medical Education: An Introduction for Veterinary Medicine Educators
This repository contains the demo data and the example python code for veterinary educators and administrators who are reading the manuscript entitled 
"Using Machine Learning in Veterinary Medical Education: An Introduction for Veterinary Medicine Educators"
Manuscript available at TBD.

## Overview
This repository contains the simulated student data, the Python code to create the data, the Python coce to create the machine learning models, and a package version file.

## Data Simulation and dataset

The mock student dataset was created specifically for this project and the Python code used to generate the dataset is found under Creation_mock_student_data within this repository.
The 400 mock student records are available in the Excel file entittled "MockData.xlsx".
There are several tabs within the Excel file.  The Sheet1 tab contains the full dataset.  The other three tabs contain the dataset with missing GRE data.
In tab “BiasedGRE1”, we removed the lowest GRE from 14 of the students who experienced failure and from 71 of the students who did not experience failure.  
In tab “BiasedGRE2”, we randomly removed GRE scores from 200 student records.  

## Package version file
The text file entitiled "package_requirements" contains all the packages and the specific versions that were used during creation of the simulated data.  They can be installed prior to running the models to ensure there are no package conflicts.

## Python Code
The Python code contained within the manuscript as well as the additional Python code required to generate all the random forest models using the biased datasets is available in the ExamplePythonCode.py file.

## How to Use
1) Download Anaconda Distribution at https://www.anaconda.com/download.
2) Create a vritual environment specifically within Anaconda.
3) Open Spyder within this virtual environment or another Python IDE of your choice.
4) In Spyder, you can create a new project and ensure the created project path or current path is in the root path (see https://docs.spyder-ide.org/3/projects.html).
5) In the IPython window (Spyder IDE) use 'pip install -r package_requirements.text' to install all the required packages.
6) Copy the python code into th Editor pane and edit the 
