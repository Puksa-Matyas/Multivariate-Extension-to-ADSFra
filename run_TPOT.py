import argparse
import datetime
import os
import re
import statistics as stats
import time
import warnings


warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*Both generations and max_time_mins.*')
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

SPLITS = ['SMD', 'SMAP', 'MSL', 'SMAP_MSL']
ALGORITHMS = ['RZS','SPE','HST','TRAKR','TranAD']
DEFAULT_SEED = 50
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RESULTS_CSV = 'THESIS_DATA.csv'

TOPK_COL_PATTERN = re.compile(r'^Top(\d+)Targets$')


def find_topk_column(columns):
    for c in columns:
        m = TOPK_COL_PATTERN.match(c)
        if m:
            return c, int(m.group(1))
    return None, None


def append_result_row(path, row):
    df = pd.DataFrame([row])
    exists = os.path.exists(path)
    df.to_csv(path, mode='a' if exists else 'w', header=not exists, index=False)


def run_tpot_on_file(data_file_path, output_dir, generations, population_size, cv,
                      random_state=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE, quiet=False):
    if not quiet:
        print(f"\n=== Processing {data_file_path}  (seed={random_state}) ===")
    data = pd.read_csv(data_file_path)
    data = data.drop(columns=[c for c in ['Unnamed: 0'] if c in data.columns])

    dataset_names = data['Dataset'] if 'Dataset' in data.columns else None
    data = data.drop(columns=[c for c in ['Dataset'] if c in data.columns])

    topk_col, topk_k = find_topk_column(data.columns)
    topk_series = data[topk_col] if topk_col else None

    X = data.drop(columns=['target'])
    y = data['target']

    X = X.select_dtypes(exclude=['object'])
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e10, upper=1e10)

    combined = pd.concat([X, y], axis=1).dropna()
    X, y = combined.drop(columns=['target']), combined['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        random_state=random_state,
        cv=cv,
        verbose=1 if quiet else 2,
        max_time_mins=None,
        n_jobs=16,
        scorers=['balanced_accuracy'], scorers_weights=[1]
    )

    start = time.time()
    tpot.fit(X_train, y_train)
    duration = time.time() - start

    y_pred = tpot.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    if not quiet:
        print(f"Test Accuracy: {accuracy:.4f}  (balanced: {balanced_accuracy:.4f})")

    label_map = {i: name for i, name in enumerate(ALGORITHMS)}
    predictions_df = pd.DataFrame({'True_Label': y_test, 'Predicted_Label': y_pred}, index=X_test.index)

    if dataset_names is not None:
        predictions_df.insert(0, 'Dataset', dataset_names.loc[predictions_df.index])

    if pd.api.types.is_numeric_dtype(predictions_df['True_Label']):
        predictions_df['True_Detector'] = predictions_df['True_Label'].map(label_map)
        predictions_df['Predicted_Detector'] = predictions_df['Predicted_Label'].map(label_map)

    topk_accuracy = None
    if topk_series is not None:
        aligned_topk = topk_series.loc[predictions_df.index]

        def in_topk(topk_str, pred_label):
            try:
                targets = {int(x) for x in str(topk_str).split(',') if x != ''}
            except ValueError:
                return np.nan
            return pred_label in targets

        hit_mask = [in_topk(t, p) for t, p in zip(aligned_topk, predictions_df['Predicted_Label'])]
        predictions_df[f'In_Top{topk_k}'] = hit_mask
        topk_accuracy = float(np.nanmean(hit_mask))
        if not quiet:
            print(f"Top-{topk_k} accuracy (predikció a {topk_k} legjobb detektor közt van): {topk_accuracy:.4f}")

    if not quiet:
        print("\n=== Predictions on Test Set ===")
        print(predictions_df.to_string())

    filename = os.path.basename(data_file_path)
    suffix = f'_seed{random_state}' if random_state != DEFAULT_SEED else ''
    predictions_df.to_csv(os.path.join(
        output_dir, os.path.splitext(filename)[0] + suffix + '_predictions.csv'))

    export_path = os.path.join(output_dir, os.path.splitext(filename)[0] + suffix + '_pipeline.py')
    try:
        if hasattr(tpot, 'export'):
            tpot.export(export_path)
        elif getattr(tpot, 'fitted_pipeline_', None) is not None:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(repr(tpot.fitted_pipeline_))
    except Exception as e:
        print(f"Warning: pipeline export failed: {e}")

    return {
        'File': filename,
        'Seed': random_state,
        'Accuracy': accuracy,
        'BalancedAccuracy': balanced_accuracy,
        'TopKAccuracy': topk_accuracy,
        'TopK': topk_k,
        'Duration': duration,
    }


def run_multi_seed(data_file_path, output_dir, generations, population_size, cv, seeds, test_size):
    all_results = []
    for i, seed in enumerate(seeds, start=1):
        print(f"\n########## Seed {seed}  ({i}/{len(seeds)}) ##########")
        result = run_tpot_on_file(
            data_file_path, output_dir, generations, population_size, cv,
            random_state=seed, test_size=test_size, quiet=(len(seeds) > 1),
        )
        print(f"  -> accuracy={result['Accuracy']:.4f}  balanced={result['BalancedAccuracy']:.4f}"
              + (f"  top-{result['TopK']}={result['TopKAccuracy']:.4f}" if result['TopKAccuracy'] is not None else ""))
        all_results.append(result)
    return all_results


def summarize_results(all_results):
    summary = {}
    for key in ['Accuracy', 'BalancedAccuracy', 'TopKAccuracy', 'Duration']:
        vals = [r[key] for r in all_results if r.get(key) is not None]
        if not vals:
            summary[f'{key}_Mean'] = None
            summary[f'{key}_Std'] = None
        elif len(vals) == 1:
            summary[f'{key}_Mean'] = vals[0]
            summary[f'{key}_Std'] = 0.0
        else:
            summary[f'{key}_Mean'] = stats.mean(vals)
            summary[f'{key}_Std'] = stats.stdev(vals)
    summary['N_Seeds'] = len(all_results)
    summary['TopK'] = next((r['TopK'] for r in all_results if r.get('TopK') is not None), None)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, required=True, choices=SPLITS)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--n_components', type=int, default=None)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--population_size', type=int, default=60)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--n_kernels', type=int, default=None)
    parser.add_argument('--test_size', type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument('--results_csv', type=str, default=DEFAULT_RESULTS_CSV)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,help='Konkrét seed-lista, pl. --seeds 30 1 7 123')
    parser.add_argument('--n_seeds', type=int, default=1, help='Ha --seeds nincs megadva, ennyi egymást követő seedet generál DEFAULT_SEED-től.')
    args = parser.parse_args()

    DATA_DIR = f'benchmarks/{args.benchmark}/TPOT'
    OUTPUT_DIR = f'benchmarks/{args.benchmark}/TPOT/output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.raw:
        n_components_tag = 'raw'
        expected_filename = f'{args.method}_raw_labeled.csv'
    else:
        if args.n_components is None:
            raise SystemExit("--n_components kötelező, hacsak nem adod meg a --raw kapcsolót.")
        n_components_tag = args.n_components
        expected_filename = f'{args.method}_autoencoder_{args.n_components}_labeled.csv'

    data_file_path = os.path.join(DATA_DIR, expected_filename)
    if not os.path.exists(data_file_path):
        raise SystemExit(f"Nem található a várt bemeneti fájl: {data_file_path}")

    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [DEFAULT_SEED + i for i in range(args.n_seeds)]

    all_results = run_multi_seed(
        data_file_path, OUTPUT_DIR, args.generations, args.population_size, args.cv,
        seeds, args.test_size,
    )

    for result in all_results:
        append_result_row(args.results_csv, {
            'Timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'Benchmark': args.benchmark,
            'Method': args.method,
            'N_Kernels': args.n_kernels,
            'N_Components': n_components_tag,
            'Generations': args.generations,
            'Population_Size': args.population_size,
            'CV': args.cv,
            'TestSize': args.test_size,
            'Seed': result['Seed'],
            'File': result['File'],
            'Accuracy': result['Accuracy'],
            'BalancedAccuracy': result['BalancedAccuracy'],
            'TopK': result['TopK'],
            'TopKAccuracy': result['TopKAccuracy'],
            'Duration_Seconds': result['Duration'],
        })

    summary = summarize_results(all_results)

    print(f"\n{'=' * 60}")
    print(f"ÖSSZESÍTÉS ({summary['N_Seeds']} seed alapján)")
    print(f"{'=' * 60}")
    print(f"Accuracy:          {summary['Accuracy_Mean']:.4f} ± {summary['Accuracy_Std']:.4f}")
    print(f"Balanced accuracy: {summary['BalancedAccuracy_Mean']:.4f} ± {summary['BalancedAccuracy_Std']:.4f}")
    if summary['TopKAccuracy_Mean'] is not None:
        print(f"Top-{summary['TopK']} accuracy:      {summary['TopKAccuracy_Mean']:.4f} ± {summary['TopKAccuracy_Std']:.4f}")
    print(f"Átlagos futásidő:  {summary['Duration_Mean']:.1f}s ± {summary['Duration_Std']:.1f}s")

    append_result_row(args.results_csv, {
        'Timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'Benchmark': args.benchmark,
        'Method': args.method,
        'N_Kernels': args.n_kernels,
        'N_Components': n_components_tag,
        'Generations': args.generations,
        'Population_Size': args.population_size,
        'CV': args.cv,
        'TestSize': args.test_size,
        'Seed': 'SUMMARY',
        'File': os.path.basename(data_file_path),
        'Accuracy': summary['Accuracy_Mean'],
        'Accuracy_Std': summary['Accuracy_Std'],
        'BalancedAccuracy': summary['BalancedAccuracy_Mean'],
        'BalancedAccuracy_Std': summary['BalancedAccuracy_Std'],
        'TopK': summary['TopK'],
        'TopKAccuracy': summary['TopKAccuracy_Mean'],
        'TopKAccuracy_Std': summary['TopKAccuracy_Std'],
        'N_Seeds': summary['N_Seeds'],
        'Duration_Seconds': summary['Duration_Mean'],
    })
    print(f"\n[OK] Eredmények (seedenkénti sorok + SUMMARY sor) mentve ide: {args.results_csv}")