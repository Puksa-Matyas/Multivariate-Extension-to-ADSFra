import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from time import time
import psutil
import argparse
import numpy as np


def AnomalyDetect_RollingZscore(data, window_size=100, eps=1e-6, process=None, peak_memory=0):
    def _update_peak(peak):
        if process is not None:
            return max(peak, process.memory_info().rss)
        return peak

    T, Ni = data.shape
    window_size = min(window_size, T - 1) if T > 1 else 1

    pad_cumsum = np.vstack([np.zeros((1, Ni)), np.cumsum(data, axis=0)])
    pad_cumsum_sq = np.vstack([np.zeros((1, Ni)), np.cumsum(data ** 2, axis=0)])
    peak_memory = _update_peak(peak_memory)

    mean = np.zeros((T, Ni))
    std = np.zeros((T, Ni))

    sums = pad_cumsum[window_size:T] - pad_cumsum[0:T - window_size]
    sums_sq = pad_cumsum_sq[window_size:T] - pad_cumsum_sq[0:T - window_size]

    mean[window_size:T] = sums / window_size
    var = sums_sq / window_size - mean[window_size:T] ** 2
    var = np.maximum(var, 0.0)
    std[window_size:T] = np.sqrt(var)
    peak_memory = _update_peak(peak_memory)

    z = (data - mean) / (std + eps)
    score = np.sum(z ** 2, axis=1)
    score[:window_size] = 0.0 
    peak_memory = _update_peak(peak_memory)

    return score, peak_memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rolling z-score anomaly detection on a specified benchmark.')
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--window_size', type=int, default=100, help='Rolling window size for mean/std estimation')

    args = parser.parse_args()

    benchmark = args.benchmark
    method = 'RZS'

    benchmark_path = f'benchmarks/{benchmark}'
    folder_path = f'{benchmark_path}/data'
    result_path = f'{benchmark_path}/results'

    os.makedirs(result_path, exist_ok=True)

    filename_list = os.listdir(f'{folder_path}/test_label')

    if len(filename_list) == 0:
        print(f"\nFIGYELEM: nem talalhato egyetlen feldolgozando fajl sem a '{benchmark}' "
              f"benchmarkhoz (a mappa letezik, de ures, vagy rossz az adatstruktura).\n"
              f"Ellenorizd, hogy a '{folder_path}' alatt megvannak-e a vart fajlok.\n")

    process = psutil.Process(os.getpid())

    for filename in filename_list:
        print(filename)

        dataset_start = time()
        memory_start = process.memory_info().rss
        peak_memory = memory_start

        if benchmark in ['SMD']:
            data_train = np.loadtxt(f'{folder_path}/train/{filename}', delimiter=',')
            data_test = np.loadtxt(f'{folder_path}/test/{filename}', delimiter=',')
        elif benchmark in ['SMAP', 'MSL','SMAP_MSL']:
            filename_npy = filename.replace('.txt', '.npy')
            data_train = np.load(f'{folder_path}/train/{filename_npy}')
            data_test = np.load(f'{folder_path}/test/{filename_npy}')

        data = np.concatenate((data_train, data_test), axis=0)
        print('data shape:', data.shape)

        train_length = data_train.shape[0]
        peak_memory = max(peak_memory, process.memory_info().rss)

        anomaly_score, peak_memory = AnomalyDetect_RollingZscore(
            data, window_size=args.window_size,
            process=process, peak_memory=peak_memory,
        )
        anomaly_score[:train_length] = 0.0

        filename_stem = os.path.splitext(filename)[0]
        test_score = anomaly_score[train_length:]
        np.savetxt(f'{result_path}/{filename_stem}_{method}.txt', test_score)

        dataset_time = time() - dataset_start
        peak_memory_gb = peak_memory / (1024 ** 3)

        with open(f'{result_path}/runtime_{method}.csv', 'a') as f:
            f.write(f"{benchmark},{filename_stem},{dataset_time:.4f},{peak_memory_gb:.4f}\n")

        print(f"Peak RAM usage: {peak_memory_gb:.2f} GB")
        print(f"Adathalmaz futasi ideje: {dataset_time:.4f} s")

    print(f"\nKesz. Az eredmenyek ide keruletek mentesre: {result_path}/*_{method}.txt")