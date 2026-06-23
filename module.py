import numpy as np
import pandas as pd


# ============================================================
# Numerical constant
# ============================================================

EPS_LOG = 1e-8      # Small offset used in log(x + eps)


# ============================================================
# 1. Sign-preserving logarithmic transformation
# ============================================================

def log_new_lap(dataframe, column_in, column_out):
    """
    Apply a sign-preserving logarithmic transformation.

    This transformation is used for quantities that can be positive,
    negative, or zero, such as Laplacians and cross-gradient terms.

    Formula:
        x' =  log(x^2 + 1), if x >= 0
        x' = -log(x^2 + 1), if x < 0

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.
    column_in : str
        Name of the input column.
    column_out : str
        Name of the output column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the transformed column added.
    """

    x = pd.to_numeric(dataframe[column_in], errors="coerce")

    dataframe[column_out] = np.where(
        x >= 0,
        np.log(x**2 + 1.0),
        -np.log(x**2 + 1.0)
    )

    return dataframe


# ============================================================
# 2. Sign-preserving logarithmic transformation for zeta
# ============================================================

def log_new_zeta(dataframe, column_in, column_out):
    """
    Apply the machine-learning logarithmic transformation to zeta.

    Important:
    This function preserves the sign of zeta. The absolute value of zeta
    is used only in the DFT-inspired descriptor zeta_red inside
    quantum_scaling_noy.

    Formula:
        zeta' =  log(zeta^2 + 1), if zeta >= 0
        zeta' = -log(zeta^2 + 1), if zeta < 0

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe.
    column_in : str
        Name of the zeta input column.
    column_out : str
        Name of the transformed zeta output column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with the transformed zeta column added.
    """

    x = pd.to_numeric(dataframe[column_in], errors="coerce")

    dataframe[column_out] = np.where(
        x >= 0,
        np.log(x**2 + 1.0),
        -np.log(x**2 + 1.0)
    )

    return dataframe


# ============================================================
# 3. DFT-inspired and machine-learning feature construction
# ============================================================

def quantum_scaling_noy(df):
    """
    Build the preprocessing dataframe used as input for the neural network.

    The function keeps the same public column names as the original
    preprocessing script.

    Final DFT-inspired columns, in the original order:
        rho_red, zeta_red, q_red, khi, s,
        delta_s, s_p, tau_ratio, delta_tau_ratio

    Final machine-learning-preprocessed columns:
        rho_log, zeta_log, GA_log, GB_log, GC_log,
        tau_up_log, tau_down_log, lap_up_log, lap_down_log

    Important:
    - abs(zeta) is used only for the DFT descriptor zeta_red.
    - signed zeta is preserved for the machine-learning descriptor zeta_log.
    - No standardization is applied here.
    - No NaN/inf cleanup is applied here.
    - Invalid numerical values are intentionally kept for later processing.

    Expected input columns:
        rho, zeta, GA, GB, GC, lap_up, lap_down, tau_up, tau_down

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the final preprocessing features in a fixed order.
    """

    df = df.copy()

    required_cols = [
        "rho", "zeta", "GA", "GB", "GC",
        "lap_up", "lap_down",
        "tau_up", "tau_down"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert required input columns to numeric values.
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")

    # ========================================================
    # 1. Density and spin variables
    # ========================================================

    # Signed spin polarization is used to reconstruct spin densities.
    zeta = df["zeta"]

    # Absolute spin polarization is used only for the DFT spin descriptor.
    zeta_abs = np.abs(zeta)

    # Spin-up and spin-down densities.
    # These use signed zeta, not abs(zeta).
    df["rho_up"] = 0.5 * df["rho"] * (1.0 + zeta)
    df["rho_down"] = 0.5 * df["rho"] * (1.0 - zeta)

    # rho_red = rho^(1/3)
    df["rho_red"] = np.cbrt(df["rho"])

    # zeta_red = 1/2 * [(1 + |zeta|)^(4/3) + (1 - |zeta|)^(4/3)]
    df["zeta_red"] = 0.5 * (
        np.cbrt(1.0 + zeta_abs) ** 4
        + np.cbrt(1.0 - zeta_abs) ** 4
    )

    # ========================================================
    # 2. Local Fermi wave vectors
    # ========================================================

    # Total-density Fermi wave vector:
    # kF = (3*pi^2*rho)^(1/3)
    df["kf"] = np.cbrt(3.0 * np.pi**2 * df["rho"])

    # Spin-resolved Fermi wave vectors:
    # kF_sigma = (6*pi^2*rho_sigma)^(1/3)
    df["kf_up"] = np.cbrt(6.0 * np.pi**2 * df["rho_up"])
    df["kf_down"] = np.cbrt(6.0 * np.pi**2 * df["rho_down"])

    # ========================================================
    # 3. Density-gradient contractions
    # ========================================================

    # Total squared density gradient:
    # |grad rho|^2 = GA + GB + 2 GC
    df["grad_total"] = df["GA"] + df["GB"] + 2.0 * df["GC"]

    # ========================================================
    # 4. Curvature variables q, q_red, and khi
    # ========================================================

    df["lap_total"] = df["lap_up"] + df["lap_down"]
    df["tau_r"] = df["tau_up"] + df["tau_down"]

    # q = 1/6 * [lap rho - 2 tau + 1/2 * |grad rho|^2 / rho]
    df["q"] = (1.0 / 6.0) * (
        df["lap_total"]
        - 2.0 * df["tau_r"]
        + 0.5 * (df["grad_total"] / df["rho"])
    )

    df["q_red"] = df["q"] / (df["rho"] * df["kf"].pow(2))

    # Spin-resolved curvature variables.
    df["q_up"] = (1.0 / 6.0) * (
        df["lap_up"]
        - 2.0 * df["tau_up"]
        + 0.5 * (df["GA"] / df["rho_up"])
    )

    df["q_down"] = (1.0 / 6.0) * (
        df["lap_down"]
        - 2.0 * df["tau_down"]
        + 0.5 * (df["GB"] / df["rho_down"])
    )

    df["q_red_up"] = df["q_up"] / (df["rho_up"] * df["kf_up"].pow(2))
    df["q_red_down"] = df["q_down"] / (df["rho_down"] * df["kf_down"].pow(2))

    # Same column name as in the original code: khi.
    df["khi"] = (
        np.abs(df["q_red_up"] - df["q_red_down"])
        / (df["q_red_up"] + df["q_red_down"])
    )

    # ========================================================
    # 5. Reduced-gradient variables
    # ========================================================

    df["s"] = (
        np.sqrt(df["grad_total"])
        / (2.0 * df["kf"] * df["rho"])
    )

    df["s_up"] = (
        np.sqrt(df["GA"])
        / (2.0 * df["kf_up"] * df["rho_up"])
    )

    df["s_down"] = (
        np.sqrt(df["GB"])
        / (2.0 * df["kf_down"] * df["rho_down"])
    )

    df["delta_s"] = (
        np.abs(df["s_up"] - df["s_down"])
        / (df["s_up"] + df["s_down"])
    )

    # Relative orientation of spin-density gradients.
    df["s_p"] = df["GC"] / np.sqrt(df["GA"] * df["GB"])

    # Keep the original convention for undefined same-direction cases.
    df["s_p"].fillna(1, inplace=True)

    # ========================================================
    # 6. Kinetic-energy-density variables
    # ========================================================

    df["tau_w"] = df["grad_total"] / (8.0 * df["rho"])
    df["tau_w_up"] = df["GA"] / (8.0 * df["rho_up"])
    df["tau_w_down"] = df["GB"] / (8.0 * df["rho_down"])

    df["tau_ratio"] = df["tau_w"] / df["tau_r"]
    df["tau_ratio_up"] = df["tau_w_up"] / df["tau_up"]
    df["tau_ratio_down"] = df["tau_w_down"] / df["tau_down"]

    df["delta_tau_ratio"] = (
        np.abs(df["tau_ratio_up"] - df["tau_ratio_down"])
        / (df["tau_ratio_up"] + df["tau_ratio_down"])
    )

    # ========================================================
    # 7. Machine-learning logarithmic preprocessing
    # ========================================================

    # Positive quantities: log(x + eps).
    df["rho_log"] = np.log(df["rho"] + EPS_LOG)
    df["GA_log"] = np.log(df["GA"] + EPS_LOG)
    df["GB_log"] = np.log(df["GB"] + EPS_LOG)
    df["tau_up_log"] = np.log(df["tau_up"] + EPS_LOG)
    df["tau_down_log"] = np.log(df["tau_down"] + EPS_LOG)

    # Signed quantities: sign-preserving log(x^2 + 1).
    # zeta_log keeps signed zeta. The absolute value is used only in zeta_red.
    df = log_new_zeta(df, "zeta", "zeta_log")
    df = log_new_lap(df, "GC", "GC_log")
    df = log_new_lap(df, "lap_up", "lap_up_log")
    df = log_new_lap(df, "lap_down", "lap_down_log")

    # ========================================================
    # 8. Final column order
    # ========================================================

    final_cols = [
        # DFT-inspired variables: same names and order as the original code
        "rho_red",
        "zeta_red",
        "q_red",
        "khi",
        "s",
        "delta_s",
        "s_p",
        "tau_ratio",
        "delta_tau_ratio",

        # Machine-learning-preprocessed variables
        "rho_log",
        "zeta_log",
        "GA_log",
        "GB_log",
        "GC_log",
        "tau_up_log",
        "tau_down_log",
        "lap_up_log",
        "lap_down_log"
    ]

    return df[final_cols]