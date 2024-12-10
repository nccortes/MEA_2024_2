import matplotlib.pyplot as plt
import numpy as np
import os
cwd = os.getcwd()
#------------------------------------------------------------------------------------#
# Utilized to show the different configurations and spin flips for a certain list of N spins
def gray_show_binary(N):
    k = 1
    g = [[0], [1]]
    while k < N:
        g = [[0] + gg for gg in g] + [[1] + gg for gg in g[::-1]]
        k += 1
    return g
#------------------------------------------------------------------------------------#
# Gray flip func to be utilized inside the enumerate-ising algorithym
def gray_flip(tau):
    k = tau[0]
    tau[k - 1] = tau[k]
    tau[k] = k + 1
    if k != 1:
        tau[0] = 1
    return tau, k
#------------------------------------------------------------------------------------#
# Nearest neighbors of a given site. Note that sites start from 1 up to N=L*L, which 
# doesn't match python's indexing convention. There are two separate funtions for non-
# PBC and PBC lattices.
def Nbr(site, L, direction):
    x = (site - 1) % L
    y = (site - 1) // L 
    if direction == 1:  # first (right)
        x += 1
    elif direction == 2:  # second (up)
        y += 1
    elif direction == 3:  # third (left)
        x -= 1
    elif direction == 4:  # fourth (down)
        y -= 1
    elif direction >= 5 or direction<=0:
        return 0
    # Check for out-of-bounds conditions
    if x < 0 or x >= L or y < 0 or y >= L:
        return 0
    # Convert coordinates back to site number (1-indexed)
    neighbor_site = y*L + (x + 1)
    return neighbor_site
def Nbr_periodic(site, L, direction):
    if site >= 1:
        x = (site - 1) % L
        y = (site - 1) // L 
        if direction == 1:  # first (right)
            x = (x + 1) % L  # wrap around on x-axis (right)
        elif direction == 2:  # second (up)
            y = (y + 1) % L  # wrap around on y-axis (up)
        elif direction == 3:  # third (left)
            x = (x - 1) % L  # wrap around on x-axis (left)
        elif direction == 4:  # fourth (down)
            y = (y - 1) % L  # wrap around on y-axis (down)
        elif direction >= 5 or direction <= 0:
            return 0
        neighbor_site = y * L + (x + 1)
        return neighbor_site
    else:
        exit("Wrong site number!")  
#------------------------------------------------------------------------------------#
def energy_ising(spin_list, periodicity=True):
    E = 0
    N = len(spin_list)
    L = int(np.sqrt(N))
    for site in range(1, N + 1):
        site_index = site - 1
        if periodicity == False:
            neighbors = [Nbr(site, direction, L) for direction in range(1,5)]
        if periodicity == True:
            neighbors = [Nbr_periodic(site, direction, L) for direction in range(1,5)]
        for neighbor in neighbors:
            neighbor_index = neighbor - 1
            E -= spin_list[site_index]*spin_list[neighbor_index]
    return E/2
#------------------------------------------------------------------------------------#
def enumerate_ising(L, periodicity=True):
    N = L * L
    S = [-1] * N
    E = -2 * N
    #M = -1 * N
    tau = list(range(1, N + 2))

    dos = {}
    dos[E] = 1

    for _ in range(1, 2**N):
        tau, k = gray_flip(tau)
        if periodicity == False:
            neighbor_list = [Nbr(k, L, direction) for direction in range(1,5)]
        elif periodicity == True:
            neighbor_list = [Nbr_periodic(k, L, direction) for direction in range(1,5)]
        else:
            exit("Wrong periodicity parameter!")
        
        h = 0
        for neighbor in neighbor_list:
            h += S[neighbor - 1]
        E += 2 * h * S[k - 1]
        S[k - 1] *= -1
        
        if E in dos:
            dos[E] += 1
        else:
            dos[E] = 1
    return dos  
#------------------------------------------------------------------------------------#
def plot_dos_ising(L, dos_dict, periodicity=False, fig_name=None):
    # unpack the dictionary
    energy_list = np.array(list(dos_dict.keys()))
    dos = np.array(list(dos_dict.values()))
    dos_normalized = dos/sum(dos)

    plt.bar(energy_list, dos_normalized, color="green")
    plt.xticks(energy_list, energy_list) 
    plt.xlabel("Energy")
    plt.ylabel(r"Density of States  $\mathcal{N}(E)$")
    if periodicity:
        plt.title(f"Density of states for a {L}x{L} grid with PBC")
    else:
        plt.title(f"Density of states for a {L}x{L} grid without PBC")
    if fig_name != None:
        file_path = cwd + "/Plots/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def enumerate_ising_magnetization(L, periodicity=True):
    N = L * L
    S = [-1] * N
    E = -2 * N
    M = -1 * N
    tau = list(range(1, N + 2))
    
    # inicialización del diccionario para M y E iniciales
    N_M_E_dict = {}
    N_M_E_dict[M] = {E: 1}
    
    N_E_M_dict = {}
    N_E_M_dict[E] = {M: 1}

    for _ in range(1, 2**N):
        tau, k = gray_flip(tau)
        
        if periodicity == False:
            neighbor_list = [Nbr(k, L, direction) for direction in range(1,5)]

        elif periodicity == True:
            neighbor_list = [Nbr_periodic(k, L, direction) for direction in range(1,5)]

        h = 0
        for neighbor in neighbor_list:
            h += S[neighbor - 1]
        E  += 2 * S[k - 1] * h
        S[k - 1] *= -1
        M += 2 * S[k - 1] 

        if M in N_M_E_dict:
            if E in N_M_E_dict[M]:
                N_M_E_dict[M][E] += 1
            else:
                N_M_E_dict[M][E] = 1
        else:
            N_M_E_dict[M] = {E: 1}

        if E in N_E_M_dict:
            if M in N_E_M_dict[E]:
                N_E_M_dict[E][M] += 1
            else:
                N_E_M_dict[E][M] = 1
        else:
            N_E_M_dict[E] = {M: 1}

    return N_M_E_dict, N_E_M_dict
#------------------------------------------------------------------------------------#
def check_e_sum(N_E_M_dict):
    for e_key, M_dicts in zip(N_E_M_dict.keys(), N_E_M_dict.values()):
        count_for_e = 0
        for count in M_dicts.values():
            count_for_e += count
        print(e_key, count_for_e) 
#------------------------------------------------------------------------------------#
def magnetization_probability(N_M_E_dict, T):
    kB = 1  # Boltzmann constant
    beta = 1 / (kB * T)
    
    Z = 0.0
    pi_M = {}

    # Calculate total partition function Z
    for M in N_M_E_dict:  # Loop over all magnetizations
        for E, count in N_M_E_dict[M].items():  # Loop over energies for each magnetization
            weight = count * np.exp(-beta * E)
            Z += weight
    
    # Calculate the probability for each magnetization
    for M in N_M_E_dict:
        sum_over_E = 0.0
        for E, count in N_M_E_dict[M].items():
            weight = count * np.exp(-beta * E)
            sum_over_E += weight
        pi_M[M] = sum_over_E / Z  # Normalize by the partition function
    return pi_M
#------------------------------------------------------------------------------------#
def plot_magnetization_probability(L, T_list=None, periodicity=True, bar_width=0.8, fig_name=None):
    if not T_list:
        T_list = [2.5, 5.0]
    Tc = 2/np.log(1+np.sqrt(2))
    T_list.append(Tc)
    T_list = sorted(T_list)
    # get the N_M_E directory
    dos = enumerate_ising_magnetization(L, periodicity)[0]
    
    for T in T_list:
        # get magnetization probability directory
        pi_M = magnetization_probability(dos, T)
        M_values = sorted(pi_M.keys())
        pi_values = [pi_M[M] for M in M_values]
        if T == Tc:
            label = r"T = $T_c$"
        else:
            label = f'T = {T}'
        plt.bar(M_values, pi_values, width=bar_width, label=label)
    plt.xticks(list(M_values), list(M_values))
    plt.xlabel(r'Total Magnetization $M$')
    plt.ylabel(r'Probability $\pi_M$')
    plt.legend()
    plt.xlim(min(dos.keys())-.5, max(dos.keys())+.5)
    if periodicity:
        plt.title(rf"Probability $\pi_M$ for a {L}$\times${L} grid with PBC")
    else:
        plt.title(rf"Probability $\pi_M$ for a {L}$\times${L} grid without PBC")
    if fig_name != None:
        file_path = cwd + "/P2/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def Binder_cumulant(N_M_E_dict, T):
    pi_M = magnetization_probability(N_M_E_dict, T)
    avg_M2 = 0
    avg_M4 = 0
    for M in pi_M.keys():
        avg_M2 += (M**2) * pi_M[M]
        avg_M4 += (M**4) * pi_M[M]
    result = (1 / 2) * (3 - avg_M4 / avg_M2**2) 
    return result
#------------------------------------------------------------------------------------#
def plot_Binder_cumulant(T_values=None, L_list=None, periodicity=True, fig_name=None):
    Tc = 2 / np.log(1 + np.sqrt(2))

    if not T_values:
        T_values = np.linspace(0.5, 5, 1000)
    
    if not L_list:
        L_list = list(range(2,6))

    for L in L_list:
        BC = Binder_cumulant(enumerate_ising_magnetization(L, periodicity=periodicity)[0], T_values)
        plt.plot(T_values, BC, label= f"{L}x{L} grid")

    plt.axvline(x=(2 / np.log(1 + np.sqrt(2))),color="gray",linestyle="dashed", label = rf"$T_c$ = {np.round(Tc, 3)}")
    plt.xlabel("Temperature")
    plt.ylabel("Binder cumulant")
    plt.xlim(min(T_values), max(T_values))
    plt.legend(loc="best")
    if periodicity:
        plt.title(rf"Binder cumulant for grids with PBC")
    else:
        plt.title(rf"Binder cumulant for grids without PBC")
    if fig_name != None:
        file_path = cwd + "/P2/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#