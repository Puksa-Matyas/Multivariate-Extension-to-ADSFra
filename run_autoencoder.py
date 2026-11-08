import os
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import layers, Model

# Set random seeds for reproducibility
SEED = 3
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

n_components = 4  # Bottleneck size


def build_autoencoder(input_dim, bottleneck_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(input_layer)
    encoded = layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(encoded)
    bottleneck = layers.Dense(bottleneck_dim, activation='linear', name='bottleneck', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(encoded)

    # Decoder
    decoded = layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(bottleneck)
    decoded = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(decoded)
    output_layer = layers.Dense(input_dim, activation='linear', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    return autoencoder, encoder


def load_fingerprints(input_folder, method):

    all_data_combined = pd.DataFrame()

    filenames = sorted(os.listdir(input_folder))
    print(f"Found {len(filenames)} file(s) in {input_folder}")

    method_suffix = f'_{method}'

    for filename in filenames:
        file_path = os.path.join(input_folder, filename)

        if not os.path.isfile(file_path):
            continue

        dataset = os.path.splitext(filename)[0]
        if dataset.endswith(method_suffix):
            dataset = dataset[: -len(method_suffix)]

        data = pd.read_csv(file_path)

        if data.shape[0] == 0:
            data = pd.read_csv(file_path, header=None)
            data.columns = [f'feature_{i}' for i in range(data.shape[1])]

        if 'Timestamp' in data.columns:
            data_clean = data.drop(columns=['Timestamp'])
        else:
            data_clean = data

        data_clean.insert(0, 'Dataset', dataset)
        all_data_combined = pd.concat([all_data_combined, data_clean], ignore_index=True)

        print(f"  loaded {filename}  (shape={data_clean.shape})")

    return all_data_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduce fingerprint files with an Autoencoder.')
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--method', type=str, required=True, help='Fingerprinting method / folder name to use (e.g., MultiRocket)')
    parser.add_argument('--n_components', type=int, default=n_components, help='Bottleneck dimension (default: 4)')

    args = parser.parse_args()

    benchmark = args.benchmark
    method = args.method
    n_components = args.n_components

    input_folder = f"benchmarks/{benchmark}/fingerprint/{method}"
    output_folder = f"benchmarks/{benchmark}/fingerprint/autoencoder"
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n=== Processing {method} fingerprints for {benchmark} ===")

    all_data_combined = load_fingerprints(input_folder, method)
    print('===========================================================')
    print(all_data_combined.isna().sum().sum())
    print(all_data_combined.shape)

    if all_data_combined.empty:
        print(f"No data found for {method} in {input_folder}. Exiting.")
        raise SystemExit(1)

    # Standardize
    scaler = StandardScaler()
    numeric_cols = all_data_combined.select_dtypes(include=[np.number]).columns
    data_scaled = scaler.fit_transform(all_data_combined[numeric_cols])

    # Train Autoencoder
    print(f"Training Autoencoder for {method} (input_dim={data_scaled.shape[1]}, samples={data_scaled.shape[0]})...")
    autoencoder, encoder = build_autoencoder(input_dim=data_scaled.shape[1], bottleneck_dim=n_components)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(data_scaled, data_scaled, epochs=1000, batch_size=32, shuffle=True, verbose=1)
    print("Training complete.")

    # Generate reduced features
    bottleneck_components = encoder.predict(data_scaled)

    autoencoder_df = pd.DataFrame(bottleneck_components, columns=[f'AE_{i + 1}' for i in range(n_components)])
    autoencoder_df.insert(0, 'Dataset', all_data_combined['Dataset'].values)

    # Save results
    output_file_path = os.path.join(output_folder, f'{method}_autoencoder_{n_components}.csv')
    autoencoder_df.to_csv(output_file_path, index=False)
    print(f"Saved reduced features to: {output_file_path}")