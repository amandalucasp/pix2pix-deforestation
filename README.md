# Data augmentation for deforestation detection using Pix2Pix

1. Clone the repo
2. Pre-processing images T1, T2: 
    - Choose preprocessing parameters at file `config.yaml`
    - Run `python pre_processing.py`
3. Run pix2pix training with the extracted patches:
    - Choose training parameters at file `config.yaml`
    - Run `python pix2pix.py`
    - The script will firstly train the pix2pix network on the training pairs and generate synthetic data with the trained input.

## Options

A brief explanation of each parameter is given on the config.yaml file.

- `synthetic_input_mode`: defines the mode on which the script will generate the input (t1 + synthetic/random mask) to be provided to the trained pix2pix generator in order to generate t2 images
    - 0: does nothing
    - 1: get images with no deforestation; get masks with _only_ old deforestation and change the mask value from the _old deforestation pixels_ to the value corresponding to _new deforestation_.
    - 2: get images with forest and old deforestantion and its corresponding masks; apply dilatation to the mask and set the dilated pixels to the value corresponding to _new deforestation_ until a minimum % of new deforestation pixels is obtained.
    - 3: get images with all classes and its corresponding masks; apply dilation and set dilated pixels to _new deforestation_ until a minimum % of new deforestation pixels is obtained.