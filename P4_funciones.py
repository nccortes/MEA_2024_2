import random
import numpy as np
import os
import matplotlib.pyplot as plt

#------------------------------------------------------------------------#
def energy_ising(S, Nbr_dict):
    E = 0
    for i in range(len(S)):
        for j in Nbr_dict[i]:
            E -= S[i] * S[j]  # Each interaction is counted twice
    return E / 2  # Correct for double counting
#------------------------------------------------------------------------#
def cluster_ising(L, T_values):
    N = L * L
    # generación del directorio de vecinos
    Nbr_dict = {i : ((i // L) * L + (i + 1) % L, (i + L) % N,
                     (i // L) * L + (i - 1) % L, (i - L) % N)
                     for i in range(N)}
    # lista de spins y magnetización como la suma de los spines
    S = [random.choice([1, -1]) for _ in range(N)]
    N_spins_to_flip = N * 100000
    results = []

    for T in T_values:
        # def de variables para cada temperatura
        p_magic  = 1.0 - np.exp(-2.0 / T)
        M_tot_abs = 0.0
        E_tot = 0.0
        E_tot_sq = 0.0
        N_steps = 0
    
        N_flipped_spins = 0
        while N_flipped_spins < N_spins_to_flip:
            # definición de conjuntos iniciales para Pocket y Cluster
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

            # actualización de la energía y magnetización
            E = energy_ising(S, Nbr_dict) / N  # Normalize energy per spin
            E_tot += E
            E_tot_sq += E * E
            N_flipped_spins += len(C)
            N_steps += 1

        # calcular los valores medios
        E_avg = E_tot / N_steps  # <E> por spin
        E_avg_sq = E_tot_sq / N_steps  # <E^2> por spin
        c_V = (E_avg_sq - E_avg**2) * N / (T**2)

        results.append((T, E_avg, c_V)) 
    return results