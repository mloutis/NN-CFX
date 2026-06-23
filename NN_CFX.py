import numpy as np
import pandas as pd
import pickle
import torch

from modelXC import ModelXC
from module import log_new_lap, log_new_zeta, quantum_scaling_noy
from script import MLP


class CF(ModelXC):
    """
    Correlation-factor neural-network exchange-correlation model.

    This class extends ModelXC and replaces the exchange-correlation energy
    density by a neural-network prediction based on preprocessed local DFT
    descriptors.

    The input features are built from PySCF grid quantities already computed
    in the parent ModelXC class.
    """

    def __init__(
        self,
        molecule,
        positions,
        spin,
        approx="pbe,pbe",
        basis="6-311+g2dp.nw",
        num_threads=1,
        ASE=False,
        charge=0,
        init_dm=None,
        batch_size=1000
    ):
        """
        Initialize the CF model.

        Parameters
        ----------
        molecule : object
            Molecular system passed to the parent ModelXC class.
        positions : array-like
            Atomic positions.
        spin : int
            Spin multiplicity or spin setting used by the parent class.
        approx : str, optional
            Exchange-correlation approximation used for the reference DFT run.
        basis : str, optional
            Basis set used in the electronic-structure calculation.
        num_threads : int, optional
            Number of CPU threads.
        ASE : bool, optional
            Whether the input structure comes from ASE.
        charge : int, optional
            Molecular charge.
        init_dm : array-like, optional
            Initial density matrix.
        batch_size : int, optional
            Number of grid points processed at once by the neural network.
        """

        # Initialize the parent ModelXC class.
        super().__init__(
            molecule,
            positions,
            spin,
            approx,
            basis,
            num_threads,
            ASE,
            charge,
            init_dm
        )

        self.batch_size = batch_size

    def calc_eps_xc_cf_batch(self, istart, iend):
        """
        Predict the exchange-correlation energy density on a grid batch.

        The grid is processed by slices in order to reduce memory usage.
        For each batch, two feature blocks are constructed:

        1. Machine-learning-preprocessed features.
        2. DFT-inspired quantum descriptors.

        These two blocks are then fused and passed through the trained neural
        network.

        Parameters
        ----------
        istart : int
            Starting grid index.
        iend : int
            Ending grid index.

        Returns
        -------
        list
            Neural-network predictions for the exchange-correlation energy
            density on the selected batch.
        """

        # ====================================================
        # 1. Extract local grid quantities for the current batch
        # ====================================================

        rho = self.rho_tot[istart:iend]
        zeta = self.zeta[istart:iend]
        GA = self.GA[istart:iend]
        GB = self.GB[istart:iend]
        GC = self.GC[istart:iend]
        lap_up = self.lap_up[istart:iend]
        lap_down = self.lap_down[istart:iend]
        tau_up = self.tau_up[istart:iend]
        tau_down = self.tau_down[istart:iend]

        # Keep the same input column names used during training.
        data = {
            "rho": rho,
            "zeta": zeta,
            "GA": GA,
            "GB": GB,
            "GC": GC,
            "lap_up": lap_up,
            "lap_down": lap_down,
            "tau_up": tau_up,
            "tau_down": tau_down
        }

        self.df_ml = pd.DataFrame(data)

        # A separate copy is used for the DFT-inspired preprocessing.
        self.df_qm = self.df_ml.copy()

        # ====================================================
        # 2. Machine-learning preprocessing block
        # ====================================================

        # Sign-preserving logarithmic transforms for quantities that may
        # be positive or negative.
        self.df_ml = log_new_lap(self.df_ml, "lap_up", "lap_up_prime")
        self.df_ml = log_new_lap(self.df_ml, "lap_down", "lap_down_prime")
        self.df_ml = log_new_zeta(self.df_ml, "zeta", "zeta_prime")

        # Keep the same order as in the original training pipeline.
        self.df_prime = self.df_ml[
            [
                "lap_up_prime",
                "lap_down_prime",
                "zeta_prime"
            ]
        ]

        # Positive or mostly positive variables transformed with log1p.
        # The column names and order are intentionally kept unchanged.
        self.df_ml = self.df_ml[
            [
                "rho",
                "GA",
                "GB",
                "GC",
                "tau_up",
                "tau_down"
            ]
        ]

        # Apply log(1 + x) element-wise.
        self.df_ml = np.log1p(self.df_ml)

        # Concatenate the log1p block with the sign-preserving log block.
        self.df_ml = pd.concat(
            [
                self.df_ml,
                self.df_prime
            ],
            axis=1
        )

        # ====================================================
        # 3. DFT-inspired preprocessing block
        # ====================================================

        # Build the quantum/DFT descriptors with the same output structure
        # used during training.
        self.df_qm = quantum_scaling_noy(self.df_qm)

        # ====================================================
        # 4. Feature fusion
        # ====================================================

        # The order is important:
        # first the ML-preprocessed block, then the DFT-inspired block.
        # This must match the scaler and the neural-network training setup.
        self.X = self.df_ml.join(self.df_qm)

        del self.df_ml, self.df_qm

        # ====================================================
        # 5. Apply the fitted scaler
        # ====================================================

        # Load the scaler used during training.
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Apply the same scaling to the new fused features.
        self.X = scaler.transform(self.X)

        # ====================================================
        # 6. Load the trained neural network
        # ====================================================

        self.model = MLP()

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load("model.pth"))
        else:
            self.model.load_state_dict(
                torch.load("model.pth", map_location="cpu")
            )

        self.model.eval()

        # ====================================================
        # 7. Neural-network prediction
        # ====================================================

        self.X_numpy = self.X

        del self.X

        # Convert the input features to a PyTorch tensor.
        self.X_tensor = torch.from_numpy(self.X_numpy).double()

        # Disable gradient tracking for inference.
        with torch.no_grad():
            self.y_pred = self.model(self.X_tensor)

        # Convert the tensor output to a Python list.
        self.y_pred = self.y_pred.tolist()

        return self.y_pred

    def calc_eps_xc_cf(self, batch_size):
        """
        Predict the exchange-correlation energy density on the full grid.

        The full integration grid is split into batches to avoid excessive
        memory usage.

        Parameters
        ----------
        batch_size : int
            Number of grid points processed per batch.

        Returns
        -------
        numpy.ndarray
            Predicted exchange-correlation energy density on all grid points.
        """

        eps_xc = np.empty([0])

        for istart in range(0, self.n_grid, batch_size):
            iend = min(istart + batch_size, self.n_grid)

            eps_xc = np.append(
                eps_xc,
                self.calc_eps_xc_cf_batch(istart, iend)
            )

        return eps_xc

    def calc_Exc_cf(self):
        """
        Compute the total exchange-correlation energy.

        The local neural-network prediction is integrated over the numerical
        grid using the DFT quadrature weights and the total electron density.

        Returns
        -------
        float
            Total exchange-correlation energy.
        """

        exc = self.calc_eps_xc_cf(self.batch_size)

        # Replace invalid numerical values by zero before integration.
        exc = np.nan_to_num(
            exc,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        Exc = np.sum(self.weights * self.rho_tot * exc)

        return Exc

    def calc_Etot_cf(self):
        """
        Compute the total energy corrected by the neural-network CF model.

        The reference DFT exchange-correlation contribution is removed and
        replaced by the neural-network CF exchange-correlation energy.

        Returns
        -------
        float
            Total corrected energy.
        """

        Exc = self.calc_Exc_cf()

        Etot = self.mf.e_tot - self.approx_Exc + Exc

        return Etot