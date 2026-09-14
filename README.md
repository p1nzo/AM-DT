# AM-DT
## Reduced Order Model (ROM) builder for Direct Firing Furnace (DFF) configuration
***ROM builder for ArcelorMittal***, with code implemented in a *jupyter notebook* for easy use and reproducibility of the training. \
Hierarchical order of folders must be respected, otherwise the folder paths in the code must change accordingly.
```
main
 |
 ├── DATA ──── plane-xz-strip-middle_outlet/      # Simulation data
 |         |               └── burners_bot_species/     # Simulation data relative to species and temperature at the flame section
 |         |               └── chimney_species/     # Simulation data relative to species and temperature at the chimney outlet
 |         └── cases_parameters.csv      # .csv containing the operating conditions
 |         └── avg_temp.csv      # .csv containing strip outlet temperature for reference
 |
 └── SRC/       # Auxiliary codes (libraries for ROM construction and GPR training)
 └── DT.ipynb
 └── config.json       # File to set the training parameters
 └── train.py       # File to train the GPR (alternative to jupyter notebook in .ipynb)
```

### Detailed information on how to use and what the code does is in ```DT.ipynb```.

