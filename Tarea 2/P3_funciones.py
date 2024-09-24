from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random
import numpy as np
from os import getcwd
cwd = getcwd()

#------------------------------------------------------------------------------------#
# the first site is 0, insead of 1 as in classes.
def Nbr_periodic(L):
    N = L * L
    Nbr_dict = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                     (i // L) * L + (i - 1) % L, (i - L) % N)
                     for i in range(N)}
    return Nbr_dict
#------------------------------------------------------------------------------------#
def energy_ising(S, Nbr_dict):
    E = 0
    for site in Nbr_dict.keys():
        for neighbor in Nbr_dict[site]:
            E -= S[site] * S[neighbor]
    return E / 2
#------------------------------------------------------------------------------------#
def markov_ising(L, S, T):
    beta = 1 / T
    E = energy_ising(S, Nbr_periodic(L))
    N = L * L
    k = random.randint(0, N - 1)
    h = 0
    neigbors_list = Nbr_periodic(L)[k]
    
    for neighbor in neigbors_list:
        h += S[neighbor]

    d_E = 2 * h * S[k]
    U = np.exp(-beta * d_E)
    if random.uniform(0,1) < U:
        S[k] *= -1
        E += d_E
    return S, E
#------------------------------------------------------------------------------------#
def thermo_markov_ising(L, T, N_samples=1e7):
    beta = 1 / T
    N = L * L
    S = [random.choice([-1, 1]) for _ in range(N)]
    E_avg = 0
    E_sqrd_avg = 0
    M_avg = 0
    E = energy_ising(S, Nbr_periodic(L))

    for _ in range(int(N_samples)):
        S, E = markov_ising(L, S, T)
        E_avg += E 
        E_sqrd_avg += E**2
        M_avg += np.abs(sum(S))
    E_avg /= N_samples
    E_sqrd_avg /= N_samples
    M_avg /= N_samples

    c_V = beta ** 2 * (E_sqrd_avg - E_avg ** 2) / float(N)
    e_avg = E_avg / N
    m_avg = M_avg / N

    return e_avg, c_V, m_avg
#------------------------------------------------------------------------------------#
def markov_chain(L, T, N_samples):
    N = L * L
    S = [random.choice([-1, 1]) for _ in range(N)]
    E_avg = 0
    E_sqrd_avg = 0
    M_avg = 0
    E = energy_ising(S, Nbr_periodic(L))

    for _ in range(int(N_samples)):
        S, E = markov_ising(L, S, T)
        E_avg += E 
        E_sqrd_avg += E**2
        M_avg += np.abs(sum(S))
    
    return E_avg, E_sqrd_avg, M_avg
#------------------------------------------------------------------------------------#
# Combine results from multiple chains
def combine_results(results, N_samples):
    E_avg = sum(r[0] for r in results) / N_samples
    E_sqrd_avg = sum(r[1] for r in results) / N_samples
    M_avg = sum(r[2] for r in results) / N_samples
    
    return E_avg, E_sqrd_avg, M_avg
#------------------------------------------------------------------------------------#
# Parallelized version using Joblib
def parallel_thermo_markov_ising(L, T, n_chains=8, N_samples=1e6):
    N_samples_per_chain = int(N_samples / n_chains)
    
    results = Parallel(n_jobs=n_chains)(delayed(markov_chain)(L, T, N_samples_per_chain) for _ in range(n_chains))
    
    # Combine results
    E_avg, E_sqrd_avg, M_avg = combine_results(results, N_samples)

    # Compute final thermodynamic properties
    beta = 1 / T
    c_V = beta ** 2 * (E_sqrd_avg - E_avg ** 2) / float(L * L)
    e_avg = E_avg / (L * L)
    m_avg = M_avg / (L * L)

    return e_avg, c_V, m_avg
#------------------------------------------------------------------------------------#
def abs_mag_for_T(L, T_values, n_chains=8, N_samples=1e6):
    mag_list = []
    for T in T_values:
        _, _, mag_value = parallel_thermo_markov_ising(L, T, n_chains,N_samples) 
        mag_list.append(mag_value)
    return mag_list
#------------------------------------------------------------------------------------#
def plot_absolute_magnetization(L_values=None, T_steps = 15, fig_name=None, n_chains=8, N_samples_factor=1e5):
    T_values = np.linspace(0.5, 5, T_steps)
    if not L_values:
        L_values = [4, 8, 16, 32]
    for L in L_values:
        mag_LxL = abs_mag_for_T(L, T_values,n_chains=n_chains, N_samples = L * N_samples_factor)
        plt.plot(T_values, mag_LxL, "o-", label=rf"{L}$\times${L}")

    plt.ylabel(r'Absolute mean magnetization $\left\langle|m|\right\rangle$')
    plt.xlabel(r'Temperature $T$')
    plt.legend()
    plt.xlim(min(T_values), max(T_values))
    plt.title("Mean absolute magnetization for a grid with PBC")
    if fig_name != None:
        file_path = cwd + "/P3/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
