# Viscoelastic Constitutive Artificial Neural Networks (vCANNs)
This is the GitHub Repository for the paper "Viscoelastic Constitutive Artificial Neural Networks (vCANNs) – a framework for data-driven anisotropic nonlinear finite viscoelasticity", K.P. Abdolazizi, K. Linka, C.J. Cyron, Journal of Computational Physics: 499:112704, 2024.

Besides the source code, this repo contains a minimal working example that illustrates most of the features of vCANN using the uniaxial loading-unloading data of VHB 4910 also used in the paper. After training, the training and validation results should resemble the figure below. Note that the deviations in sub-figure (d) are due to inconsistencies in the experimental data, as discussed in detail in the paper.

<p align="center">
  <img src="https://github.com/user-attachments/assets/aa000380-d5a8-488d-b1df-b97d3c7c40d1" width=75%>
</p>

## Requirements (tested with)
```
python==3.10.13
matplotlib==3.8.2
numpy==1.26.0
scipy==1.11.3
scikit-learn==1.3.0
keras==2.10.0
tensorflow==2.10.0
tqdm==4.65.0
kormos==0.1.4
plotly==5.9.0
```

## Citation
```
@article{Abdolazizi2024,
title = {Viscoelastic constitutive artificial neural networks (vCANNs) - A framework for data-driven anisotropic nonlinear finite viscoelasticity},
journal = {Journal of Computational Physics},
volume = {499},
pages = {112704},
year = {2024},
issn = {0021-9991},
url = {https://www.sciencedirect.com/science/article/pii/S0021999123007994},
doi = {https://doi.org/10.1016/j.jcp.2023.112704},
author = {Kian P. Abdolazizi and Kevin Linka and Christian J. Cyron},
}
```

## Contact
If you have any questions, please contact kian.abdolazizi@tuhh.de.
