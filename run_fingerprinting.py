import argparse
import os

import numpy as np
from sktime.transformations.rocket import MultiRocketMultivariate

datasets = ['MSL', 'SMAP', 'SMD', 'SMAP_MSL']


def fit_multirocket(normalized_data_list, n_kernels, n_feature_per_kernel):

    groups = {}
    for d in normalized_data_list:
        groups.setdefault(d.shape[1], []).append(d)

    print(f"    Csatornaszám-csoport(ok): "
          f"{ {n_ch: len(g) for n_ch, g in groups.items()} }")

    transformers = {}
    for n_channels, group in groups.items():

        common_length = min(d.shape[0] for d in group)
        if any(d.shape[0] != common_length for d in group):
            print(f"    ({n_channels} csatornás csoport: fit-hez {common_length} "
                  f"hosszra vágva a legrövidebb fájl alapján -- a transform a "
                  f"teljes hosszon fut)")
        X_ref = np.stack([d[:common_length].T for d in group], axis=0)

        trf = MultiRocketMultivariate(
            num_kernels=n_kernels,
            n_features_per_kernel=n_feature_per_kernel,
            random_state=42,
            n_jobs=-1)

        print(f"    Fitting MultiRocket reference transformer "
              f"({n_channels} csatorna, {len(group)} fájl)...")
        trf.fit(X_ref)
        transformers[n_channels] = trf

    return transformers


def apply_multirocket(data, transformers):

    n_channels = data.shape[1]
    trf = transformers.get(n_channels)
    if trf is None:
        raise ValueError(
            f"Nincs fit-elt MultiRocket transzformer {n_channels} csatornás "
            f"fájlokhoz (elérhető csoportok: {sorted(transformers.keys())})."
        )

    X = data.T[np.newaxis, :, :]  # (T, n_channels) -> (1, n_channels, T)
    fingerprint = np.asarray(trf.transform(X))[0]
    return fingerprint


METHODS = {
    'MultiRocket': {
        'fit': fit_multirocket,
        'apply': apply_multirocket,
    },
}


def load_file(benchmark, folder_path, filename):
    if benchmark == 'SMD':
        return np.loadtxt(os.path.join(folder_path, 'test', filename), delimiter=',')
    return np.load(os.path.join(folder_path, 'test', filename))


def normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1
    return (data - mean) / std


def strip_extension(benchmark, filename):
    if benchmark == 'SMD':
        return filename.replace('.txt', '')
    return filename.replace('.npy', '')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run fingerprinting on different datasets.')
    parser.add_argument('--benchmark', type=str, required=True, choices=datasets,
                         help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--method', type=str, required=True, choices=list(METHODS.keys()),
                         help='Fingerprinting method to use')
    parser.add_argument('--n_kernels', type=int, default=84,
                         help='Number of kernels to use. Need to be a multiple of 84. If not, it will be rounded down')
    parser.add_argument('--n_feature_per_kernel', type=int, default=4,
                         help='Number of features per kernel')

    args = parser.parse_args()

    benchmark = args.benchmark
    method = args.method
    n_kernels = args.n_kernels
    n_feature_per_kernel = args.n_feature_per_kernel

    print(f"Started", method, "on ", benchmark, "dataset.")
    benchmark_path = f'benchmarks/{benchmark}'
    folder_path = f'{benchmark_path}/data'
    out_path = f'{benchmark_path}/fingerprint/{method}'

    os.makedirs(out_path, exist_ok=True)

    filename_list = os.listdir(f'{folder_path}/test')
    method_fns = METHODS[method]

    normalized_data_by_file = {}
    for filename in filename_list:
        data = load_file(benchmark, folder_path, filename)
        normalized_data_by_file[filename] = normalize(data)

    state = method_fns['fit'](
        list(normalized_data_by_file.values()),
        n_kernels=n_kernels,
        n_feature_per_kernel=n_feature_per_kernel,
    )

    for i, filename in enumerate(filename_list, start=1):
        print(f"[{i}/{len(filename_list)}] Generating fingerprint for: {filename}")
        normalized_data = normalized_data_by_file[filename]

        out_filename = strip_extension(benchmark, filename) + '_' + method

        fingerprint = method_fns['apply'](normalized_data, state)

        out_file = os.path.join(out_path, out_filename + '.csv')
        method_prefix = method.lower()
        header = ','.join(f'{method_prefix}_{j}' for j in range(fingerprint.shape[0]))

        np.savetxt(
            out_file,
            fingerprint.reshape(1, -1),
            delimiter=',',
            fmt="%.8f",
            header=header,
            comments=''
        )
        print(f"    Output feature vector size:", fingerprint.shape[0])
        print(f"    Saved fingerprint for:", out_filename)

    print(f"Saved all fingerprints to:", out_path)