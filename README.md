Code for the paper ["Finding Collisions for Round-Reduced Romulus-H"](https://doi.org/10.46586/tosc.v2023.i1.67-88)
===================================================================================================================

This repository contains the implementations of our attacks presented in our paper:

```bibtex
@article{tosc/NagelerPE23,
    author = {Marcel Nageler and Felix Pallua and Maria Eichlseder},
    title = {Finding Collisions for Round-Reduced {Romulus}-H},
    journal = {{IACR} Trans. Symmetric Cryptol.},
    number = {1},
    volume = {2023},
    pages = {67--88},
    year = {2023},
    doi = {10.46586/tosc.v2023.i1.67-88},
}
```


## Usage

To run the Python code in the `sat_modelling` directory you first need to run the following command to compile the Cython extension modules.
```bash
cd sat_modelling
python3 setup.py build_ext -i
```

You may need the following dependencies
```bash
apt install cython3 python3-z3
pip install pycryptosat
```
