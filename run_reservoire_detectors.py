import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
import copy
import threading
import psutil


class PeakMemoryMonitor:

    def __init__(self, process, interval=0.05):
        self.process = process
        self.interval = interval
        self.peak = 0
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self._stop_event.wait(self.interval)

    def start(self):
        self.peak = self.process.memory_info().rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return self.peak


def generate_Wr(Nx, alpha_r, beta_r):
    np.random.seed(0)
    Wr = np.zeros(Nx * Nx)
    Wr[0:int(Nx * Nx * beta_r)] = np.random.normal(0, 1 / np.sqrt(beta_r * Nx), int(Nx * Nx * beta_r))
    np.random.shuffle(Wr)
    Wr = Wr.reshape((Nx, Nx))
    Wr *= alpha_r
    return Wr

def generate_Wb(Nx, Ny, alpha_b, beta_b):
    np.random.seed(0)
    Wb = np.zeros(Nx * Ny)
    Wb[0:int(Nx * Ny * beta_b)] = np.random.uniform(-1, 1, int(Nx * Ny * beta_b))  # beta_b = non-zero rate
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nx, Ny))
    Wb *= alpha_b
    return Wb

def generate_Wi(Nx, Ni, alpha_i, beta_i):
    np.random.seed(0)
    Wi = np.zeros(Nx * Ni)
    Wi[0:int(Nx * Ni * beta_i)] = np.random.uniform(-1, 1, int(Nx * Ni * beta_i))  # beta_i = non-zero rate
    np.random.shuffle(Wi)
    Wi = Wi.reshape((Nx, Ni))
    Wi *= alpha_i
    return Wi

def fx(x):
    return np.tanh(x)

def f_kernel(x):
    return x


class Reservoir:
    def __init__(self, param):
        self.T1 = param['T1']
        self.train_length = param['train_length']
        self.Nx = param['Nx']
        self.Ny = param['Ny']
        self.Ni = param['Ni']
        self.N_P = param['N_P']

        self.tau = param['tau']
        self.dt = param['dt']
        self.sigma_noise = param['sigma_noise']
        self.alpha = param['alpha']

        self.alpha_r = param['alpha_r']
        self.beta_r = param['beta_r']
        self.alpha_i = param['alpha_i']
        self.beta_i = param['beta_i']
        self.alpha_b = param['alpha_b']
        self.beta_b = param['beta_b']

        ## Prepare precision matrix P for each label
        self.P = np.identity(self.N_P) / self.alpha
        self.P_mod = np.zeros_like(self.P)
        self.mu = np.zeros(self.N_P)

    def generate_weight_matrices(self):
        self.Wr = generate_Wr(self.Nx, self.alpha_r, self.beta_r)
        self.Wb = generate_Wb(self.Nx, self.Ny, self.alpha_b, self.beta_b)
        self.Wi = generate_Wi(self.Nx, self.Ni, self.alpha_i, self.beta_i)
        self.Wo = np.zeros((self.Ny, self.N_P))

    ## Reset reservoir to initial value at the start of learning or inference
    def reset(self, x_init=0.2):
        self.X = np.zeros((self.T1, self.Nx))
        self.Y = np.zeros((self.T1, self.Ny))
        self.X[0, :] = x_init
        self.Y[0, :] = np.zeros(self.Ny)
        self.TRAKR_errors = np.zeros(self.T1)
        self.current_x = None  # Hold current state (for MDRS)

    def freerun(self):
        self.reset()
        self.T_free = 10000
        self.X_ = np.zeros((self.T_free, self.Nx))
        self.Y_ = np.zeros((self.T_free, self.Ny))
        self.X_[0] = self.X[0]
        for n in range(self.T_free):
            self.run_one_step(n, np.zeros(self.Ni), 'freerun', online=False)
        self.x_init = self.X_[self.T_free - 1]

    def run_one_step(self, n, input, mode, online=False):
        if mode == 'freerun':
            x = self.X_[n, :]  # Aktualis allapot
            y = self.Y_[n, :]
        else:
            x = self.X[n, :]
            y = self.Y[n, :]

        sum = np.zeros(self.Nx)  # Jϕ(x(t))
        sum += self.Wr @ fx(x)

        if self.Ni == 1:
            sum += (self.Wi * input)[:, 0]
        else:
            sum += self.Wi @ input

        dx = (self.dt / self.tau) * (- x + sum)
        x += dx

        y = self.Wo @ f_kernel(x[:self.N_P])

        if online == True:
            q = self.P @ f_kernel(x)[:self.N_P]
            c = 1 / (1 + f_kernel(x)[:self.N_P].T @ q)
        error = y - input

        if mode == 'train' and online == True:
            self.mu = (n * self.mu + f_kernel(x)[:self.N_P]) / (n + 1)  # average
            self.P -= np.outer(q, q) * c
            self.Wo[:, :self.N_P] -= np.outer(error, q) * c

        if mode == 'test' and online == True:
            self.TRAKR_errors[n] = np.linalg.norm(np.outer(error, q) * c) ** 2

        if mode == 'freerun':
            self.Y_[n, :] = y  # Save as output at time n (for reconstruction)
            if n < self.T_free - 1:
                self.X_[n + 1, :] = x
        else:
            self.Y[n, :] = y  # Save as output at time n (for reconstruction)
            self.current_x = x  # Hold updated state (for MDRS)
            if n < self.T1 - 1:
                self.X[n + 1, :] = x


def AnomalyDetect_TRAKR(data, reservoir, continual_learning, online):
    ## Initialize precision matrix
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha

    reservoir.reset(reservoir.x_init)
    train_end_time = None
    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                reservoir.run_one_step(n, data[n], 'train', online=online)

                if n == reservoir.train_length - 1:
                    if online == False:
                        ## Train precision matrix in batch
                        x_mat = f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P])
                        cov_mat = x_mat.T @ x_mat + reservoir.alpha * np.identity(reservoir.N_P)  # with regularization
                        reservoir.P = np.linalg.inv(cov_mat)

                    train_end_time = time.time()
            else:
                reservoir.run_one_step(n, data[n], 'test', online=online)

        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)

    TRAKR = reservoir.TRAKR_errors

    return TRAKR, train_end_time

def AnomalyDetect_SPE(data, reservoir, continual_learning=False, online=False):
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha
    reservoir.P_mod = np.zeros_like(reservoir.P)
    reservoir.mu = np.zeros(reservoir.N_P)
 
    reservoir.reset(reservoir.x_init)
 
    train_end_time = None
    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                reservoir.run_one_step(n, data[n], 'train', online=online)
 
                if n == reservoir.train_length - 1:
                    if online == True:
                        ## Correct precision matrix with average
                        P_mu = reservoir.P @ reservoir.mu
                        tmp_mat = (reservoir.train_length / (
                                    1 - reservoir.train_length * reservoir.mu.T @ reservoir.P @ reservoir.mu)) * np.outer(P_mu,
                                                                                                                          P_mu)
                        reservoir.P_mod = reservoir.train_length * (reservoir.P + tmp_mat)  # precision matrix
 
                    elif online == False:
                        ## Train readout in batch (Reconstruction)
                        # Train to reconstruct data(t) from state X(t)
                        X_list = f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P])
                        Y_list = data[:reservoir.train_length]  # Target data at the same time (for reconstruction)
                        reservoir.Wo = Y_list.T @ np.linalg.pinv(X_list.T)  # solve Y_list.T = Wo @ X_list.T
 
                    train_end_time = time.time()
 
            else:
                reservoir.run_one_step(n, data[n], 'test', online=False)
 
 
        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)
 
    if reservoir.Ni == 1:
        SPE = (data - reservoir.Y[:, 0]) ** 2
    else:
        SPE = (data - reservoir.Y) ** 2
        SPE = np.sum(SPE, axis=1)
    SPE[:reservoir.train_length] = 0
 
    return SPE, train_end_time

def AnomalyDetect_MDRS(data, reservoir, precision_mode, continual_learning=False, online=False):
    ## Initialize precision matrix
    reservoir.P = np.identity(reservoir.N_P) / reservoir.alpha
    reservoir.P_mod = np.zeros_like(reservoir.P)
    reservoir.mu = np.zeros(reservoir.N_P)

    MDRS = np.zeros(reservoir.T1)
    reservoir.reset(reservoir.x_init)
    train_end_time = None
    for n in range(reservoir.T1):
        if continual_learning == False:
            if n < reservoir.train_length:
                reservoir.run_one_step(n, data[n], 'train', online=online)

                if n == reservoir.train_length - 1:
                    if online == True:
                        ## Correct precision matrix with average
                        P_mu = reservoir.P @ reservoir.mu
                        tmp_mat = (reservoir.train_length / (
                                    1 - reservoir.train_length * reservoir.mu.T @ reservoir.P @ reservoir.mu)) * np.outer(P_mu,
                                                                                                                          P_mu)
                        reservoir.P_mod = reservoir.train_length * (reservoir.P + tmp_mat)  # precision matrix

                    elif online == False:
                        ## Train precision matrix in batch
                        x_mat = f_kernel(reservoir.X[:reservoir.train_length, :reservoir.N_P])
                        if precision_mode == 'zero':
                            cov_mat = x_mat.T @ x_mat + reservoir.alpha * np.identity(reservoir.N_P)  # with regularization
                            reservoir.P = np.linalg.inv(cov_mat)
                        elif precision_mode == 'mean':
                            reservoir.mu = np.mean(x_mat)  # average
                            cov_mat = (x_mat - reservoir.mu).T @ (x_mat - reservoir.mu) + reservoir.alpha * np.identity(
                                reservoir.N_P)  # corrected with average, with regularization
                            reservoir.P_mod = np.linalg.inv(cov_mat)

                    train_end_time = time.time()

            else:
                reservoir.run_one_step(n, data[n], 'test', online=False)
                x = reservoir.current_x[:reservoir.N_P]  # Use updated state (for reconstruction)
                if precision_mode == 'zero':
                    MDRS[n] = f_kernel(x).T @ reservoir.P @ f_kernel(x)
                elif precision_mode == 'mean':
                    MDRS[n] = (f_kernel(x) - reservoir.mu).T @ reservoir.P_mod @ (
                            f_kernel(x) - reservoir.mu)  # corrected with average

        elif continual_learning == True:
            reservoir.run_one_step(n, data[n], 'train', online=True)
            if n >= reservoir.train_length:
                x = reservoir.X[n, :reservoir.N_P]
                MDRS[n] = f_kernel(x).T @ reservoir.P @ f_kernel(x)

    return MDRS, train_end_time


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run MDRS or TRAKR anomaly detection on a specified benchmark.')
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--method', type=str, default='MDRS', choices=['MDRS', 'TRAKR', 'SPE'], help='Method to use for anomaly detection (MDRS or TRAKR , SPE)')

    # Parse the arguments
    args = parser.parse_args()

    # Use command-line arguments
    benchmark = args.benchmark
    method = args.method

    MEASURE_METHODS = ['TRAKR', 'SPE']
    measure = method in MEASURE_METHODS

    param = {
        'T1': 10000,  # time size
        'Nx': 500,  # number of network nodes
        'N_P': 200,  # size of precision matrix
        'dt': 0.01,  # time step
        'sigma_noise': 0,  # standard deviation of noise
        'alpha': 1e-7,  # regularization coefficient
        'alpha_r': 1.3,  # recurrent strength
        'beta_r': 1.0,  # recurrent non-zero rate
        'alpha_i': 1.0,  # input vector strength
        'beta_i': 0.25,  # input vector non-zero rate
        'alpha_b': 1.0,  # bias strength
        'beta_b': 0.25,  # bias non-zero rate
        'train_length': 5000,  # default training data length (will be overwritten for each dataset later)
        'Ni': 1,  # number of input dimensions (will be overwritten for each dataset later)
        'Ny': 1,  # number of output dimensions (will be overwritten for each dataset later)
    }

    # Setting time constants
    mix_ratio = 0.9
    param['tau'] = np.zeros(param['Nx'])
    param['tau'][:int(param['Nx'] * (1 - mix_ratio))] = 0.01
    param['tau'][int(param['Nx'] * (1 - mix_ratio)):] = 0.025

    # Setting random seed
    np.random.seed(0)

    benchmark_path = f'benchmarks/{benchmark}'

    folder_path = f'{benchmark_path}/data'
    result_path = f'{benchmark_path}/results'
    plot_path = f'{benchmark_path}/plots'
    analysis_path = f'{benchmark_path}/analysis'

    # Create directories if they do not exist
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)

    continual_learning = False
    online = True if method == 'TRAKR' else False

    train_time_list = []
    test_time_list = []
    total_time_list = []

    filename_list = os.listdir(f'{folder_path}/test_label')

    if len(filename_list) == 0:
        print(f"\nFIGYELEM: nem talalhato egyetlen feldolgozando fajl sem a '{benchmark}' "
              f"benchmarkhoz (a mappa letezik, de ures, vagy rossz az adatstruktura).\n"
              f"Ellenorizd, hogy a '{folder_path}' alatt megvannak-e a vart fajlok.\n")

    ## Generate reservoir (only once at the beginning)
    base_reservoir = Reservoir(param)
    base_reservoir.generate_weight_matrices()
    ## Set initial value of reservoir
    base_reservoir.freerun()

    process = psutil.Process(os.getpid())

    for filename in filename_list:
        print(filename)

        dataset_start = time.time()
        if measure:
            mem_monitor = PeakMemoryMonitor(process)
            mem_monitor.start()

        if benchmark in ['SMD']:
            data_train = np.loadtxt(f'{folder_path}/train/{filename}', delimiter=',')
            data_test = np.loadtxt(f'{folder_path}/test/{filename}', delimiter=',')
            anomaly_label = np.loadtxt(f'{folder_path}/test_label/{filename}')
        elif benchmark in ['SMAP', 'MSL','SMAP_MSL']:
            filename_npy = filename.replace('.txt', '.npy')
            data_train = np.load(f'{folder_path}/train/{filename_npy}')
            data_test = np.load(f'{folder_path}/test/{filename_npy}')
            anomaly_label = np.loadtxt(f'{folder_path}/test_label/{filename}')

        data = np.concatenate((data_train, data_test), axis=0)
        print('data shape:', data.shape)

        param['train_length'] = data_train.shape[0]
        param['T1'] = data.shape[0]
        param['Ni'] = data.shape[1]
        param['Ny'] = data.shape[1]

        start_end = np.where(np.diff(anomaly_label) != 0)[0] + 1
        if anomaly_label[0] == 1:  # if first element is 1, add start point to list
            start_end = np.insert(start_end, 0, 0)
        if anomaly_label[-1] == 1:  # if last element is 1, add end point to list
            start_end = np.append(start_end, len(anomaly_label))

        ## Normalize by column (by channel)
        channel_min = np.min(data, axis=0)
        channel_max = np.max(data, axis=0)
        normalized_data = (data - channel_min) / (channel_max - channel_min + 1e-10)  # avoid division by zero

        start_time = time.time()

        # Create and use a copy of base_reservoir
        reservoir = copy.deepcopy(base_reservoir)

        # Reflect updated parameters for each dataset to reservoir
        reservoir.T1 = param['T1']
        reservoir.train_length = param['train_length']
        reservoir.Ni = param['Ni']
        reservoir.Ny = param['Ny']

        # Regenerate weight matrices if Ni or Ny has changed
        if reservoir.Ni != 1 or reservoir.Ny != 1:
            reservoir.Wb = generate_Wb(reservoir.Nx, reservoir.Ny, reservoir.alpha_b, reservoir.beta_b)
            reservoir.Wi = generate_Wi(reservoir.Nx, reservoir.Ni, reservoir.alpha_i, reservoir.beta_i)
            reservoir.Wo = np.zeros((reservoir.Ny, reservoir.N_P))

        normalized_data = 2.0 * normalized_data + 2.0
        ## Normalize by number of channels (necessary)
        normalized_data /= param['Ni']


        # Run the alogirthm on data
        if method == 'MDRS':
            anomaly_score, train_end_time = AnomalyDetect_MDRS(normalized_data, reservoir, precision_mode='mean',continual_learning=continual_learning, online=online)
        elif method == 'TRAKR':
            anomaly_score, train_end_time = AnomalyDetect_TRAKR( normalized_data, reservoir, continual_learning=continual_learning, online=online)
        elif method == 'SPE':
            anomaly_score, train_end_time = AnomalyDetect_SPE(normalized_data, reservoir, continual_learning=continual_learning, online=online)

            
        end_time = time.time()

        # Calculate time
        train_time = train_end_time - start_time if train_end_time else 0
        test_time = end_time - train_end_time if train_end_time else end_time - start_time
        total_time = end_time - start_time

        filename = os.path.splitext(filename)[0]
        # Only save the test portion of the anomaly score (train portion is always 0)
        test_score = anomaly_score[reservoir.train_length:]
        np.savetxt(f'{result_path}/{filename}_{method}.txt', test_score)

        if measure:
            peak_memory = mem_monitor.stop()
            dataset_time = time.time() - dataset_start
            peak_memory_gb = peak_memory / (1024 ** 3)

            with open(f'{result_path}/runtime_{method}.csv', 'a') as f:
                f.write(f"{benchmark},{filename},{dataset_time:.4f},{peak_memory_gb:.4f}\n")

            print(f"Peak RAM usage: {peak_memory_gb:.2f} GB")
            print(f"Adathalmaz futasi ideje: {dataset_time:.4f} s")

        print(f'train_time: {train_time:.4f}s, test_time: {test_time:.4f}s, total_time: {total_time:.4f}s')
        train_time_list.append(train_time)
        test_time_list.append(test_time)
        total_time_list.append(total_time)


        fig = plt.figure()
        ax = fig.add_subplot(211)

        for i in range(0, len(start_end), 2):
            plt.axvspan(reservoir.train_length + start_end[i], reservoir.train_length + start_end[i + 1], color='tab:orange', alpha=0.5, linewidth=0)
        ax.plot(normalized_data, linewidth=0.8)
        plt.title(filename)

        ax = fig.add_subplot(212, sharex=ax)
  
        for i in range(0, len(start_end), 2):
            plt.axvspan(reservoir.train_length + start_end[i], reservoir.train_length + start_end[i + 1], color='tab:orange', alpha=0.5, linewidth=0)

        ax.plot(anomaly_score, linewidth=0.8)

        plt.savefig(f'{plot_path}/{filename}_{method}.png')
        plt.close()

    ## save runtime
    time_path = f'{benchmark_path}/analysis/runtime_{method}.csv'
    runtime_df = pd.DataFrame({
        'Filename': filename_list,
        'Train_Time': train_time_list,
        'Test_Time': test_time_list,
        'Total_Time': total_time_list
    })
    runtime_df = runtime_df.sort_values(by='Filename') if len(runtime_df) > 0 else runtime_df
    runtime_df.to_csv(time_path, index=False)

    print(f"\nKesz. Az eredmenyek ide keruletek mentesre: {result_path}/*_{method}.txt")