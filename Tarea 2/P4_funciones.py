import random
import numpy as np
import os
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------#
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
#------------------------------------------------------------------------#
def cluster_ising(L, T_values):
    N = L * L
    # Neighbor directory
    Nbr_dict = Nbr_periodic(L)
    # random spin list and number of spins to flip
    S = [random.choice([1, -1]) for _ in range(N)]
    N_spins_to_flip = N * 100000
    results = []
    binder_cumulants = []
    magnetization_histograms = []

    for T in T_values:
        # Definition of variables for each temperature
        p_magic  = 1.0 - np.exp(-2.0 / T)
        E_tot = 0.0
        E_tot_sq = 0.0
        N_steps = 0
        magnetizations = []

        N_flipped_spins = 0
        while N_flipped_spins < N_spins_to_flip:
            # def. of Pocket and Cluster
            j = random.randint(0, N - 1)
            P, C = [j], [j]
            while P != []:
                k = random.choice(P)
                for l in Nbr_dict[k]:
                    if l not in C and S[l] == S[k] and random.uniform(0.0, 1.0) < p_magic:
                        P.append(l)
                        C.append(l)
                P.remove(k)
            for k in C:
                S[k] *= -1

            # actualization of the energy
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

        magnetization_histograms.append(magnetizations)
        U_L = calculate_binder_cumulant(magnetizations)
        binder_cumulants.append(U_L)

        results.append((T, E_avg, c_V))
    return results, magnetization_histograms, binder_cumulants
#------------------------------------------------------------------------#
# Plotting function for magnetization histograms and Binder cumulants
def plot_results(T_values, magnetization_histograms, binder_cumulants):
    # Plot magnetization histograms
    plt.figure(figsize=(10, 5))
    for i, T in enumerate(T_values):
        plt.hist(magnetization_histograms[i], bins=50, alpha=0.6, label=f"T = {T:.1f}")
    plt.xlabel("Magnetization per spin")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Magnetization Histograms")
    plt.show()

    # Plot Binder cumulant vs temperature
    plt.figure()
    plt.plot(T_values, binder_cumulants, 'o-')
    plt.xlabel("Temperature")
    plt.ylabel("Binder Cumulant")
    plt.title("Binder Cumulant as a Function of Temperature")
    plt.grid(True)
    plt.show()