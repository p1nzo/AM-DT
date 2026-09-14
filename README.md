# AM-DT
## Reduced Order Model (ROM) builder for Direct Firing Furnace (DFF) configuration
***ROM builder for ArcelorMittal***, with code implemented in a *jupyter notebook* for easy use and reproducibility of the training. \
Hierarchical order of folders must be respected, otherwise the folder paths in the code must change accordingly.
```
main
 |
 ├── DATA ──── plane-xz-strip-middle_outlet/      # Simulation data
 |         |               └── species/     # Simulation data relative to species and temperature at the chimney outlet
 |         |               └── test_case/     # Simulation data relative to cases used as test
 |         |                       └── species/     # Simulation data relative to species and temperature at the chimney outlet in the test cases
 |         └── cases_parameters.csv      # .csv containing the operating conditions
 |
 └── SRC/       # Auxiliary codes (libraries for ROM construction and GPR training)
 └── rom_builder.ipynb
 └── config.json       # File to set the training parameters
 └── train.py       # File to train the GPR (alternative to jupyter notebook in .ipynb)
``

 
