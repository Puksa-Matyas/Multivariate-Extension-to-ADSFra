import ast
import os
import numpy as np
import pandas as pd

CSV_PATH = 'labeled_anomalies.csv'
BENCHMARKS_ROOT = 'benchmarks'


def generate_labels_for_spacecraft(df, spacecraft, benchmarks_root=BENCHMARKS_ROOT):
    data_dir = os.path.join(benchmarks_root, spacecraft, 'data')
    test_dir = os.path.join(data_dir, 'test')
    label_dir = os.path.join(data_dir, 'test_label')
    os.makedirs(label_dir, exist_ok=True)

    subset = df[df['spacecraft'] == spacecraft]
    n_written = 0
    n_mismatch = 0

    for _, row in subset.iterrows():
        chan_id = row['chan_id']
        anomaly_sequences = ast.literal_eval(row['anomaly_sequences'])
        num_values = int(row['num_values'])

        npy_path = os.path.join(test_dir, f'{chan_id}.npy')
        if os.path.exists(npy_path):
            actual_len = np.load(npy_path).shape[0]
            if actual_len != num_values:
                print(f"[{spacecraft}/{chan_id}] FIGYELEM: CSV num_values={num_values}, "
                      f"tenyleges test hossz={actual_len} -> a tenyleges hosszat hasznalom.")
                n_mismatch += 1
            length = actual_len
        else:
            print(f"[{spacecraft}/{chan_id}] FIGYELEM: nem talalhato {npy_path}, "
                  f"a CSV num_values={num_values} ertekkel dolgozom.")
            length = num_values

        label = np.zeros(length, dtype=int)
        for start, end in anomaly_sequences:
            start = max(0, start)
            end = min(length - 1, end)
            label[start:end + 1] = 1

        out_path = os.path.join(label_dir, f'{chan_id}.txt')
        np.savetxt(out_path, label, fmt='%d')
        n_written += 1

    print(f"{spacecraft}: {n_written} label fajl legeneralva ide: {label_dir} "
          f"({n_mismatch} hossz-eltérés volt a CSV es a tenyleges .npy kozott)")


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    for spacecraft in ['SMAP', 'MSL']:
        if spacecraft in df['spacecraft'].unique():
            generate_labels_for_spacecraft(df, spacecraft)
        else:
            print(f"Nincs '{spacecraft}' bejegyzes a CSV-ben, kihagyva.")