import pickle as pkl
from pathlib import Path
import argparse
import time
import sys

from models.dynamic_springrank import dsr_offline, dynamic_springrank
from models.implement_springrank import static_sr

## System
recursion_limit = sys.getrecursionlimit()
if recursion_limit < 3000:
    sys.setrecursionlimit(3000)
    print("Changed recursion limit to 3000")

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["dsr", "dsr_off", "ssr"],
    help="Specified model to execute",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="A_synt_cos_beta2-0_c0-5",
    nargs="?",
    const="A_synt_cos_beta2-0_c0-5",
    choices=["A_synt_cos_beta2-0_c0-5", "A_synt_static_beta2-0_c0-5"],
    help="Specified dataset to be used",
)
parser.add_argument("-s", "--save", action="store_true", help="save results of model")
parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
args = parser.parse_args()

############
# Data
############
dataset = args.dataset
print("Dataset:", dataset)
file_path = Path(__file__).parent.parent.joinpath(f"data/input/synthetic/{dataset}.pkl")
#file_path = f"../data/input/synthetic/{dataset}.pkl"
with open(file_path, "rb") as file:
    A = pkl.load(file)
print('Dataset shape:', A.shape)
end_training = A.shape[0] // 2
end_testing = end_training + 4
T = A.shape[0]
print("Number of timesteps:", T)
encoder_path = None
test_size = 1
validation_size = 10

# Train and test split
A_train = A[:end_training, :, :]
A_test = A[end_training:, :, :]

############
# Models
############

### Dynamic SpringRank

# Online
if args.model == "dsr":
    print()
    print("Online Dynamic SpringRank:")
    print()

    tic = time.perf_counter()
    results_train = dynamic_springrank(
        A_train, validation_size=validation_size, verbose=args.verbose
    )
    results = dynamic_springrank(
        A_test,
        k0=results_train[0]["k0"],
        s0=results_train[0]["s_matrix"][-1, :],
        beta_a_opt=results_train[0]["beta_a_opt"],
        beta_L_opt=results_train[0]["beta_L_opt"],
        verbose=args.verbose,
    )
    toc = time.perf_counter()
    print(f"Accuracy: {results[0]['accuracy'] :.4f}")
    runtime = toc - tic
    print(f"Runtime: {runtime :.3f}s")
    print()

# Offline
if args.model == "dsr_off":
    print()
    print("Offline Dynamic SpringRank:")
    print()

    tic = time.perf_counter()
    results = dsr_offline(
        A,
        end_training=end_training,
        validation_size=validation_size,
        verbose=args.verbose,
    )
    toc = time.perf_counter()
    print(f"Accuracy: {results[0]['accuracy'] :.4f}")
    runtime = toc - tic
    print(f"Runtime: {runtime :.3f}s")
    print()

### Static SpringRank
if args.model == "ssr":
    print()
    print("Static SpringRank:")
    print()

    tic = time.perf_counter()
    results = static_sr(A, validation_size=validation_size, end_training=end_training)
    toc = time.perf_counter()
    print(f"Accuracy: {results[0]['accuracy']:.4f}")
    runtime = toc - tic
    print(f"Runtime: {runtime :.3f}s")
    print()

### Save Results
if args.save:
    output = {"results": results, "runtime": runtime}
    with open(f"../data/output/{args.model}_{args.dataset}_output.pkl", "wb") as file:
        pkl.dump(output, file)

    print("RESULTS SAVED")
