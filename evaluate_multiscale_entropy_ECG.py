import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------
# Set global seed for reproducibility BEFORE starting multiprocessing
# ---------------------------------------------------------------------
SEED = 1001
np.random.seed(SEED)


# =========================
# Helper functions
# =========================
def coarse_grain(series, scale):
    """
    Standard Multiscale Entropy coarse-graining procedure:

    For a given scale factor 'scale', the time series is divided into
    non-overlapping blocks of size 'scale', and the mean of each block
    is computed.
    """
    n = len(series) // scale
    if n <= 0:
        return None

    cg = series[: n * scale].reshape(n, scale).mean(axis=1)
    return cg


# Function to compute microstate entropy
def Max_Entropy(x_rand, y_rand, Serie, StatsBlock, samples):
    Threshold = 0.0
    Frac = 10
    Frac2 = 4
    Increase_Thr = 1.0 / Frac
    Max_Threshold = 0.0
    S_Max = 0

    StatsM = np.zeros((2 ** (StatsBlock * StatsBlock)))
    pow_vec = np.array(
        [2**k for k in range(StatsBlock * StatsBlock)],
        dtype=np.int64
    )

    for i in range(0, Frac2):
        if i > 0:
            Threshold = Max_Threshold - Increase_Thr
            Increase_Thr = (2.0 * Increase_Thr) / Frac

        for j in range(0, Frac):
            Stats = np.zeros_like(StatsM)

            for count in range(len(x_rand)):
                Add = 0

                for count_y in range(StatsBlock):
                    for count_x in range(StatsBlock):
                        a = int(
                            abs(
                                Serie[x_rand[count] + count_x]
                                - Serie[y_rand[count] + count_y]
                            )
                            <= Threshold
                        )

                        Add += (
                            a
                            * pow_vec[count_x + count_y * StatsBlock]
                        )

                Stats[Add] += 1

            S = 0.0

            for Hist_S in Stats:
                if Hist_S > 0:
                    p = Hist_S / samples
                    S -= p * np.log(p)

            if S > S_Max:
                S_Max = S
                Max_Threshold = Threshold
                StatsM = Stats.copy()

            Threshold += Increase_Thr

    return Max_Threshold, S_Max, StatsM


# =========================
# Global parameters
# =========================

StatsBlock = 3
samples = 10000  # Maximum number of samples per scale
# max_scale = 7  # Maximum number of Multiscale Entropy scales


# =========================
# File loading
# =========================

with open("filename.dat", "r") as myfile:
    fn = myfile.read().splitlines()

with open("diagnostics.dat", "r") as myfile:
    di = myfile.read().splitlines()

with open("block_list.dat", "r") as myfile:
    block_list = myfile.read().splitlines()


# List of classes and name-to-index mapping
diag_list = np.array(
    ["SR", "SB", "AFIB", "ST", "SVT", "AF", "SI", "AT", "AVNRT", "AVRT", "SAAWR"]
)

diag_map = {label: idx for idx, label in enumerate(diag_list)}


# =========================
# File processing (multiscale)
# =========================

def process_file(i, max_scale):
    filename = fn[i]

    if filename in block_list:
        return None

    label_str = di[i]
    label_idx = diag_map[label_str]

    filepath = Path(
        f"../ecg-signal-classification-entropies/database/"
        f"ECGDataDenoised/{filename}.csv"
    )

    data = np.loadtxt(filepath, delimiter=",")  # shape: (time, leads)

    # Lists for storing ALL scales from ALL channels.
    #
    # Convention:
    # for each lead k and scale τ:
    #   first rec_en(lead k, scale τ),
    #   followed by eps(lead k, scale τ)
    rec_en_ms = []
    eps_ms = []

    Aux = []

    for k in range(data.shape[1]):
        Serie = data[:, k].astype(np.float64)

        if Serie.max() == Serie.min():
            # Constant signal → normalization and entropy calculation
            # are not meaningful.
            continue

        # Normalize to [0, 1]
        Serie = (
            (Serie - Serie.min())
            / (Serie.max() - Serie.min())
        )

        # For each scale τ
        # for scale in range(1, max_scale + 1):
        for scale in range(max_scale, max_scale + 1):

            Serie_cg = coarse_grain(Serie, scale)

            if Serie_cg is None:
                break  # Scale is too large for this series

            Size = len(Serie_cg)

            if Size <= StatsBlock + 1:
                # Coarse-grained series is too short to form blocks
                break

            # Dynamically adjust the number of samples
            # (cannot exceed the maximum number of available positions)
            max_pos = Size - StatsBlock - 1
            local_samples = min(samples, max_pos)

            if local_samples <= 0:
                break

            # Randomly select pairs (x_rand, y_rand)
            # from the coarse-grained series
            x_rand = np.random.choice(
                max_pos,
                local_samples,
                replace=False
            )

            y_rand = np.random.choice(
                max_pos,
                local_samples,
                replace=False
            )

            # Recurrence entropy at each scale
            Eps, S_max, Stats = Max_Entropy(
                x_rand,
                y_rand,
                Serie_cg,
                StatsBlock,
                local_samples
            )

            rec_en = S_max / (
                StatsBlock * StatsBlock * np.log(2)
            )

            rec_en_ms.append(rec_en)
            eps_ms.append(Eps)

    # Final feature vector
    #
    # Interleave rec_en and eps for each lead/scale:
    # [rec_en_1, eps_1, rec_en_2, eps_2, ...]
    Aux = [
        val
        for pair in zip(rec_en_ms, eps_ms)
        for val in pair
    ]

    return Aux, label_idx


# =========================
# Parallel execution
# =========================

def main():

    # Check for required arguments
    if len(sys.argv) < 2:
        print("Please select the scale.")
        print(
            "Usage: python evaluate_multiscale_entropy_ECG.py <scale>"
        )
        sys.exit(1)

    max_scale = int(sys.argv[1])

    print(f"Selected maximum scale: {max_scale}")

    diag_counts = np.zeros(
        len(diag_list),
        dtype=int
    )

    X = [None] * len(fn)
    Y = [None] * len(fn)

    print(
        "Computing multiscale microstate entropy in parallel (ordered)..."
    )

    with ProcessPoolExecutor() as executor:

        futures = {
            executor.submit(
                process_file,
                i,
                max_scale
            ): i
            for i in range(len(fn))
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Multiscale entropy"
        ):

            i = futures[future]
            result = future.result()

            if result is not None:

                Aux, label_idx = result

                X[i] = Aux
                Y[i] = label_idx

                diag_counts[label_idx] += 1

    # Keep only valid results
    X = np.array(
        [x for x in X if x is not None]
    )

    Y = np.array(
        [y for y in Y if y is not None]
    )

    print("\nRhythm distribution:")

    for label, count in zip(diag_list, diag_counts):
        print(f"{label:7} → {count} samples")

    # Save multiscale datasets
    np.save(
        f"Data_S_msrec_scale_{max_scale}.npy",
        X
    )

    np.save(
        f"Data_L_msrec_scale_{max_scale}.npy",
        Y
    )


if __name__ == "__main__":
    main()