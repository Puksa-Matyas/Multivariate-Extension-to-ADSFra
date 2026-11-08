import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from time import time
import psutil
from river import anomaly
import argparse
import numpy as np


def AnomalyDetect_HST(data, train_length, n_trees=25, height=15, window_size=250, seed=42,
                      process=None, peak_memory=0, mem_check_every=500):

    T1 = data.shape[0]
    Ni = data.shape[1]

    hst = anomaly.HalfSpaceTrees(
        n_trees=n_trees,
        height=height,
        window_size=window_size,
        seed=seed,
    )

    score = np.zeros(T1)

    for n in range(T1):
        x = {f'ch_{i}': float(data[n, i]) for i in range(Ni)}

        if n < train_length:
            hst.learn_one(x)
        else:
            score[n] = hst.score_one(x)

        # Peak RAM merese (nem minden lepesben, mert az rss lekerdezes dragabb)
        if process is not None and n % mem_check_every == 0:
            peak_memory = max(peak_memory, process.memory_info().rss)

    if process is not None:
        peak_memory = max(peak_memory, process.memory_info().rss)

    return score, peak_memory


if __name__ == "__main__":
    # Request the arguments
    parser = argparse.ArgumentParser(description='Run HalfSpaceTrees anomaly detection on a specified benchmark.')
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--n_trees', type=int, default=25, help='Number of trees in the HalfSpaceTrees ensemble')
    parser.add_argument('--height', type=int, default=15, help='Height of each tree')
    parser.add_argument('--window_size', type=int, default=250, help='Window size for the HalfSpaceTrees model')

    # Parse the arguments
    args = parser.parse_args()

    # Use command-line arguments
    benchmark = args.benchmark
    method = 'HST'

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

        channel_min = np.min(data, axis=0)
        channel_max = np.max(data, axis=0)
        normalized_data = (data - channel_min) / (channel_max - channel_min + 1e-10)

        anomaly_score, peak_memory = AnomalyDetect_HST(
            normalized_data, train_length,
            n_trees=args.n_trees, height=args.height, window_size=args.window_size,
            process=process, peak_memory=peak_memory,
        )

        filename_stem = os.path.splitext(filename)[0]
        # Only save the test portion of the anomaly score (train portion is always 0)
        test_score = anomaly_score[train_length:]
        np.savetxt(f'{result_path}/{filename_stem}_{method}.txt', test_score)

        dataset_time = time() - dataset_start
        peak_memory_gb = peak_memory / (1024 ** 3)

        with open(f'{result_path}/runtime_{method}.csv', 'a') as f:
            f.write(f"{benchmark},{filename_stem},{dataset_time:.4f},{peak_memory_gb:.4f}\n")

        print(f"Peak RAM usage: {peak_memory_gb:.2f} GB")
        print(f"Adathalmaz futasi ideje: {dataset_time:.4f} s")

    print(f"\nKesz. Az eredmenyek ide keruletek mentesre: {result_path}/*_{method}.txt")