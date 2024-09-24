import random
import numpy as np
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()

#------------------------------------------------------------------------------------#
def direct_pi(N):
    N_hits = 0
    for _ in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if (x**2 + y**2 < 1):
            N_hits += 1
    return N_hits/N
#------------------------------------------------------------------------------------#
def plot_direct_pi(n_runs = 20, max_power = 8, fig_name=None):
    N_list = []
    results_list = []
    for power in range(1, max_power + 1):
        N = 10**power
        result_for_N = 0
        for _ in range(n_runs):
            result = direct_pi(N)
            result_for_N += result
        result_for_N /= n_runs
        
        N_list.append(N)
        results_list.append(result_for_N)

    N_list = np.array(N_list)
    results_list = np.array(results_list)
    
    plt.scatter(N_list, results_list, label="Results", color="red", zorder=2)
    plt.plot(N_list, results_list, "--", zorder= 1)
    plt.xscale('log')
    plt.axhline(y = np.pi/4, linestyle="dashed", color="gray", label=r"$\pi / 4$", zorder=-1)
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$N_\text{hits}/ N$")
    plt.legend(loc="best")
    plt.xlim(min(N_list), max(N_list))
    plt.title(rf'direct-pi results for $N = 10^1 \dots 10^{max_power}$')
    plt.tight_layout()
    if fig_name != None:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def plot_direct_pi_msqrt_dev(n_runs = 20, max_power = 8, fig_name=None):
    N_list = []
    sigma_list = []

    for power in range(1, max_power + 1):
        N = 10**power
        sum = 0.0
        for _ in range(n_runs):
            value = direct_pi(N)
            sum += (value - np.pi/4)**2
        sigma_list.append(np.sqrt(sum / N))
        N_list.append(N)

    plt.plot(N_list, sigma_list, "--", zorder=1)
    plt.scatter(N_list, sigma_list, label="Results", zorder=2, color="red")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$\left\langle (N_{\text{hits}/N}-\pi/4)^2\right\rangle$ (log)")
    plt.title(r'direct-pi sampling of $\pi$: root mean square deviation vs. N')
    plt.legend(loc='upper right')
    plt.xlim(min(N_list), max(N_list))
    plt.tight_layout()
    # if we choose so to save the image as a .png file.
    if fig_name != None:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
# Estimation of pi/4 utilizing a Markov-chain of N-"throws" or tries.
def markov_pi(N, delta):
    x, y = 1.0, 1.0
    n_hits = 0
    for _ in range(N):
        del_x, del_y = random.uniform(-delta, delta), random.uniform(-delta, delta)
        if abs(x + del_x) < 1.0 and abs(y + del_y) < 1.0:
            x, y = x + del_x, y + del_y
        if x**2 + y**2 < 1.0:
            n_hits += 1
    return n_hits/N
#------------------------------------------------------------------------------------#
def plot_markov_pi(n_runs, delta, fig_name=None):
    N_list = []
    results_list = []

    for i in np.linspace(1, 7, 10):
        N = 10**i
        result_for_N = 0
        for _ in range(n_runs):
            result = markov_pi(int(N), delta) 
            result_for_N += result
        result_for_N /= n_runs

        results_list.append(result_for_N)
        N_list.append(N)
    # pasamos a array
    N_list = np.array(N_list)
    results_list = np.array(results_list)
    
    plt.scatter(N_list, results_list, label="Results", color="red", zorder=2)
    plt.plot(N_list, results_list, "--", zorder= 1)
    plt.xscale('log')
    plt.axhline(y = np.pi/4, linestyle="dashed", color="gray", label=r"$\pi / 4$", zorder=-1)
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$N_\text{hits}/ N$")
    plt.legend(loc="best")
    plt.xlim(min(N_list), max(N_list))
    plt.title(r'$\mathtt{Markov\text{-}pi}$ results for $N = 10^1, \dots , 10^6$ and $\delta = $'+f"{delta}")
    plt.tight_layout()
    if fig_name != None:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
# Plotting of the mean sqrt of the deviation vs the number of trials for a given number
# of runs with each trial number and arbitrary list for the values of delta to test with.
def plot_markov_pi_msqrt_dev_delta_list(n_runs, delta_list, fig_name=None):
    for delta in delta_list:
        N_list = []
        sigma_list = []

        # We increase the number of trials from 2^4 up to 2^(12)
        for power in range(4, 13):
            N = 2**power
            sum = 0.0

            # we run markov_pi for n_runs-times for each number of trials 
            for _ in range(n_runs):
                value = markov_pi(N, delta)
                sum += (value - np.pi / 4)**2

            sigma_list.append(np.sqrt(sum / N))
            N_list.append(N)
        # plotting the result for each N_trials
        plt.plot(N_list, sigma_list, "--")
        plt.scatter(N_list, sigma_list, label = r'$\delta = $' + str(delta))
    
    # rest of the plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$\left\langle (N_{\text{hits}/N}-\pi/4)^2\right\rangle$ (log)")
    plt.title(r'Markov-chain sampling of $\pi$: root mean square deviation vs. N')
    plt.legend(loc='upper right')
    plt.tight_layout()
    # if we choose so to save the image as a .png file.
    if fig_name != None:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def plot_markov_pi_rejected(n_runs, delta_list, fig_name=None):
    last_reject_sum = 0
    for delta in delta_list:
        N_list = []
        reject_list = []

        for poweroftwo in range(4, 13):
            N = 2 ** poweroftwo
            sum_per_power = 0.0
            for _ in range(n_runs):
                value = markov_pi(N, delta)
                reject_rate_per_run = 100*(1 - value)
                sum_per_power += reject_rate_per_run
            mean_reject_rate_per_power = sum_per_power/n_runs
            reject_list.append(mean_reject_rate_per_power)
            N_list.append(N)
        last_reject_sum += reject_list[-1] 
        plt.plot(N_list, reject_list, "--")
        plt.scatter(N_list, reject_list, label = r'$\delta = $' + str(delta))

    plt.xscale('log')
    plt.xlabel('Number of trials (log)')
    plt.ylabel('Reject rate (%)')
    plt.title(r'Markov-chain sampling of $\pi$: reject rate vs. N')
    plt.axhline(y = last_reject_sum/len(delta_list), linestyle="--", color="gray",zorder=-1, label=f"{np.round(last_reject_sum/len(delta_list),2)} %")
    plt.legend(loc='upper right')
    plt.ylim(0,100)
    if fig_name != None:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()