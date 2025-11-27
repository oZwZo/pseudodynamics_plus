# Pseudodynamics+
Physics Informed Neural Network based method for solving the single-cell population dynamics.   
For each cell, we estiamte the dynamic parameter of the cell proliferation, differentiation and diffusion.

<img src="pdyn_logo.jpg" alt="pseudodynamics+" width="600"/>

# Getting started
Check the tutorial notebooks for instructions on preparing your data and downstream analysis.  



# Installation
```bash
git clone https://github.com/Gottgens-lab/pseudodynamics_plus.git
cd pseudodynamics_plus
pip install -e .
```

# Training
To train pseudodynamics+ on your data, make sure you store the population size information in `AnnData.uns['pop']` and saved in `h5ad` format. Configure the training setting in `config.yaml`. See examples here [example](url). Run the following command:

```bash
# with GPU 
python main_train.py --config_path config.yaml -G 0

# without GPU
python main_train.py --config_path config.yaml -G None
```




