# Neural Network Representation of Exchange-Correlation Functionals

This repository contains a Python implementation of a neural network representation of the CFX exchange-correlation functional for density functional theory (DFT) applications.

The workflow combines DFT-inspired local descriptors with machine-learning preprocessing techniques. These fused input features are then passed through a trained PyTorch neural network to predict the local exchange-correlation energy density.

For a detailed description of the theoretical background, preprocessing strategy, neural-network architecture, and numerical results, please refer to the associated preprint.

## Project Structure

* `NN_CFX.py`
  Main execution file. It defines the `CF` class, which extends `ModelXC` and evaluates the neural-network exchange-correlation model on numerical DFT integration grids.

* `modelXC.py`
  Base class used to compute molecular DFT quantities, grid variables, densities, gradients, Laplacians, kinetic energy densities, and reference exchange-correlation contributions.

* `module.py`
  Preprocessing utilities. It contains the DFT-inspired feature construction and logarithmic transformations used before neural-network inference.

* `model.pth`
  Trained PyTorch neural-network parameters.

* `scaler.pkl`
  Fitted standardization scaler used to transform the fused input features before neural network prediction.

* `LICENSE.txt`
  MIT license file.

## Requirements

The code requires Python and the following main packages:

```bash
numpy
pandas
pyscf
scikit-learn
torch
```

## Important Notes

* The preprocessing column order must remain identical to the order used during training.
* The files `model.pth` and `scaler.pkl` must correspond to the same preprocessing pipeline.
* If the preprocessing functions are modified, the scaler and neural network may need to be regenerated or retrained.
* Some numerical differences may occur when using library versions different from those used in this work for PySCF, PyTorch, NumPy, pandas, and scikit-learn.*
* The current implementation expects an `MLP` class to be available from the import used in `NN_CFX.py`.

## Citation

If you use, adapt, or draw inspiration from this work, please cite:

> **Loutis, Mohamed**, **Ernzerhof, Matthias**, and **Roy, Pierre-Olivier** (2026).
> *Representing exchange-correlation functionals through neural networks*.
> Preprint, 2026.

### BibTeX

```bibtex
@article{loutis2026representing,
  title   = {Representing exchange-correlation functionals through neural networks},
  author  = {Loutis, Mohamed and Ernzerhof, Matthias and Roy, Pierre-Olivier},
  year    = {2026},
  note    = {Preprint}
}
```

## License

This project is licensed under the [MIT License](LICENSE.txt).