import random
import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

cwd = os.getcwd()

#------------------------------------------------------------------------------------#
def direct_pi(N):
    N_hits = 0
    for _ in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if (x**2 + y**2 < 1):
            N_hits += 1
    return N_hits / N
#------------------------------------------------------------------------------------#
def plot_direct_pi(n_runs=20, max_power=8, fig_name=None):
    N_list = []
    results_list = []

    # Parallelized the runs
    for power in range(1, max_power + 1):
        N = 10**power

        # Use Joblib to parallelize over multiple runs
        result_for_N = np.mean(Parallel(n_jobs=-1)(
            delayed(direct_pi)(N) for _ in range(n_runs)
        ))
        
        N_list.append(N)
        results_list.append(result_for_N)

    N_list = np.array(N_list)
    results_list = np.array(results_list)
    
    plt.plot(N_list, results_list, "o-", zorder=1, label="Results")
    plt.xscale('log')
    plt.axhline(y=np.pi/4, linestyle="dashed", color="gray", label=r"$\pi / 4$", zorder=-1)
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$N_\text{hits}/ N$")
    plt.legend(loc="best")
    plt.xlim(min(N_list), max(N_list))
    plt.title(rf'direct-pi results for $N = 10^1 \dots 10^{max_power}$')
    plt.tight_layout()
    if fig_name:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def plot_direct_pi_msqrt_dev(n_runs=20, max_power=8, fig_name=None):
    N_list = []
    sigma_list = []

    for power in range(1, max_power + 1):
        N = 10**power

        # Parallelized the sigma calculation
        deviations = Parallel(n_jobs=-1)(
            delayed(lambda: (direct_pi(N) - np.pi/4)**2)() for _ in range(n_runs)
        )
        sigma_list.append(np.sqrt(np.mean(deviations)))
        N_list.append(N)

    plt.plot(N_list, sigma_list, "o-", zorder=1, label="Results")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$\left\langle (N_{\text{hits}/N}-\pi/4)^2\right\rangle$ (log)")
    plt.title(r'direct-pi sampling of $\pi$: root mean square deviation vs. N')
    plt.legend(loc='upper right')
    plt.xlim(min(N_list), max(N_list))
    plt.tight_layout()
    if fig_name:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def markov_pi(N, delta):
    x, y = 1.0, 1.0
    n_hits = 0
    for _ in range(N):
        del_x, del_y = random.uniform(-delta, delta), random.uniform(-delta, delta)
        if abs(x + del_x) < 1.0 and abs(y + del_y) < 1.0:
            x, y = x + del_x, y + del_y
        if x**2 + y**2 < 1.0:
            n_hits += 1
    return n_hits / N
#------------------------------------------------------------------------------------#
def plot_markov_pi(n_runs, delta, fig_name=None):
    N_list = []
    results_list = []

    for i in np.linspace(1, 7, 10):
        N = 10**i

        # Parallelized the runs for Markov Pi estimation
        result_for_N = np.mean(Parallel(n_jobs=-1)(
            delayed(markov_pi)(int(N), delta) for _ in range(n_runs)
        ))

        results_list.append(result_for_N)
        N_list.append(N)

    N_list = np.array(N_list)
    results_list = np.array(results_list)
    
    plt.plot(N_list, results_list, "o-", zorder=1, label="Results")
    plt.xscale('log')
    plt.axhline(y=np.pi/4, linestyle="dashed", color="gray", label=r"$\pi / 4$", zorder=-1)
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$N_\text{hits}/ N$")
    plt.legend(loc="best")
    plt.xlim(min(N_list), max(N_list))
    plt.title(r'$\mathtt{Markov\text{-}pi}$ results for $N = 10^1, \dots , 10^7 $ and $\delta = $'+f"{delta}")
    plt.tight_layout()
    if fig_name:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
#------------------------------------------------------------------------------------#
def plot_markov_pi_msqrt_dev_delta_list(n_runs, delta_list, fig_name=None):
    for delta in delta_list:
        N_list = []
        sigma_list = []

        for power in range(4, 13):
            N = 2**power

            deviations = Parallel(n_jobs=-1)(
                delayed(lambda: (markov_pi(N, delta) - np.pi / 4)**2)() for _ in range(n_runs)
            )
            sigma_list.append(np.sqrt(np.mean(deviations)))
            N_list.append(N)

        plt.plot(N_list, sigma_list, "o-", label=r'$\delta = $' + str(delta))

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of trials (log)')
    plt.ylabel(r"$\left\langle (N_{\text{hits}/N}-\pi/4)^2\right\rangle$ (log)")
    plt.title(r'Markov-chain sampling of $\pi$: mean square deviation vs. N')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if fig_name:
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

            reject_rates = Parallel(n_jobs=-1)(
                delayed(lambda: 100 * (1 - markov_pi(N, delta)))() for _ in range(n_runs)
            )
            reject_list.append(np.mean(reject_rates))
            N_list.append(N)

        last_reject_sum += reject_list[-1]
        plt.plot(N_list, reject_list, "o-", label=r'$\delta = $' + str(delta))

    plt.xscale('log')
    plt.xlabel('Number of trials (log)')
    plt.ylabel('Reject rate (%)')
    plt.title(r'Markov-chain sampling of $\pi$: reject rate vs. N')
    plt.axhline(y=last_reject_sum / len(delta_list), linestyle="--", color="gray", zorder=-1, label=f"{np.round(last_reject_sum/len(delta_list), 2)} %")
    plt.legend(loc='upper right')
    plt.ylim(0, 100)
    if fig_name:
        file_path = cwd + "/P1/" + f"{fig_name}.png"
        plt.savefig(file_path)
    plt.show()
