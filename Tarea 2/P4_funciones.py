from joblib import Parallel, delayed
import random
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
cwd = getcwd()
#------------------------------------------------------------------------#
# the first site is 0, insead of 1 as in classes.
def Nbr_periodic(L):
    N = L * L
    Nbr_dict = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                     (i // L) * L + (i - 1) % L, (i - L) % N)
                     for i in range(N)}
    return Nbr_dict
#------------------------------------------------------------------------#
def energy_ising(S, Nbr_dict):
    E = 0
    for i in range(len(S)):
        for j in Nbr_dict[i]:
            E -= S[i] * S[j]
    return E / 2
#------------------------------------------------------------------------#
def calculate_binder_cumulant(magnetizations):
    M2 = np.mean(np.square(magnetizations))
    M4 = np.mean(np.power(magnetizations, 4))
    U_L = (3 - (M4 / (M2**2))) / 2
    return U_L
# Function to process a single temperature
def process_temperature(T, N, S, Nbr_dict, N_spins_to_flip):
    p_magic  = 1.0 - np.exp(-2.0 / T)
    E_tot = 0.0
    E_tot_sq = 0.0
    N_steps = 0
    magnetizations = []
    N_flipped_spins = 0

    while N_flipped_spins < N_spins_to_flip:
        # Definition of Pocket and Cluster
        j = random.randint(0, N - 1)
        P, C = [j], [j]
        while P:
            k = random.choice(P)
            for l in Nbr_dict[k]:
                if l not in C and S[l] == S[k] and random.uniform(0.0, 1.0) < p_magic:
                    P.append(l)
                    C.append(l)
            P.remove(k)
        for k in C:
            S[k] *= -1

        # Energy and magnetization updates
        E = energy_ising(S, Nbr_dict) / N  # Normalize energy per spin
        E_tot += E
        E_tot_sq += E * E
        M = sum(S) / N  # Magnetization per spin
        magnetizations.append(M)
        N_flipped_spins += len(C)
        N_steps += 1

    E_avg = E_tot / N_steps  # <E> per spin
    E_avg_sq = E_tot_sq / N_steps  # <E^2> per spin
    c_V = (E_avg_sq - E_avg**2) * N / (T**2)

    U_L = calculate_binder_cumulant(magnetizations)

    return T, E_avg, c_V, magnetizations, U_L

# Modified cluster_ising function
def cluster_ising(L, T_values):
    N = L * L
    Nbr_dict = Nbr_periodic(L)
    S = [random.choice([1, -1]) for _ in range(N)]
    N_spins_to_flip = N * 100000

    # Use Joblib to parallelize temperature processing
    results = Parallel(n_jobs=-1)(
        delayed(process_temperature)(T, N, S.copy(), Nbr_dict, N_spins_to_flip) for T in T_values
    )

    # Collect the results
    temps, energies, specific_heats, magnetization_histograms, binder_cumulants = zip(*[
        (res[0], res[1], res[2], res[3], res[4]) for res in results
    ])

    return list(zip(temps, energies, specific_heats)), magnetization_histograms, binder_cumulants
#------------------------------------------------------------------------#
# Plotting function for magnetization histograms and Binder cumulants
def hist_magnetization(L, T_values, magnetization_histograms):
    # Plot magnetization histograms
    plt.figure(figsize=(10, 5))
    for i, T in enumerate(T_values):
        plt.hist(magnetization_histograms[i], bins=50, alpha=0.5, label=f"T = {T:.1f}")
    plt.xlabel("Magnetization per spin")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.title(rf"Magnetization Histogram for a {L}$\times${L} lattice")
    plt.savefig(cwd + "/P4/" + f"hist_P4_magnetization_L{L}.png")
    plt.tight_layout()
    plt.show()
#------------------------------------------------------------------------#
def plot_Binder_Cumulants(T_values, binder_dict):
    for L in binder_dict.keys():
        plt.plot(T_values, binder_dict[L], 'o-', label=rf"{L}$\times${L}")
    plt.xlabel("Temperature")
    plt.ylabel("Binder Cumulant")
    plt.title(rf"Binder Cumulant for different lattices")
    plt.savefig(cwd + "/P4/" + f"plot_P4_BC.png")
    plt.xlim(min(T_values), max(T_values))
    plt.axvline(x = 2/np.log(1 + np.sqrt(2)), linestyle="--", color="gray", zorder=-1, label=r"$T_c$")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()