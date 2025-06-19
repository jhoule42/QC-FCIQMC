import numpy as np
import matplotlib.pyplot as plt


def plot_results(total_walkers, shift, estimated_energies, dt, N_W_target, exact_gs_energy, shift_update_time):
    """
    Plots the key metrics from an FCIQMC or QC-FCIQMC simulation.
    """
    num_steps = len(total_walkers)
    time_axis = np.arange(num_steps) * dt
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Walker Population vs. Time ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_axis, total_walkers, label='Total Walkers |N|', color='dodgerblue')
    ax1.axhline(y=N_W_target, color='r', linestyle='--', label=f'Target Walkers ({N_W_target})')
    ax1.set_xlabel("Imaginary Time (τ)")
    ax1.set_ylabel("Number of Walkers")
    ax1.set_title("Walker Population vs. Time")
    ax1.legend()
    ax1.set_yscale('log')

    # --- Shift Energy vs. Time ---
    ax2 = fig.add_subplot(gs[0, 1])
    shift_time_axis = np.arange(len(shift)) * dt
    ax2.plot(shift_time_axis, shift, label='Shift S', color='darkorange')
    ax2.axhline(y=exact_gs_energy, color='r', linestyle='--', label=f'Exact Energy ({exact_gs_energy:.4f})')
    ax2.set_xlabel("Imaginary Time (τ)")
    ax2.set_ylabel("Shift Energy (S)")
    ax2.set_title("Shift Energy vs. Time")
    ax2.legend()

    # --- Projected Energy vs. Time ---
    ax3 = fig.add_subplot(gs[1, 0])
    energy_time_axis = []
    if estimated_energies:
        start_step = num_steps - len(estimated_energies) * shift_update_time
        num_energy_points = len(estimated_energies)
        energy_time_axis = np.linspace(max(0, start_step * dt), (num_steps - 1) * dt, num_energy_points)
        ax3.plot(energy_time_axis, estimated_energies, label='Projected Energy E_proj', color='green', marker='.', markersize=2, linestyle='-')

    ax3.axhline(y=exact_gs_energy, color='r', linestyle='--', label=f'Exact Energy ({exact_gs_energy:.4f})')
    ax3.set_xlabel("Imaginary Time (τ)")
    ax3.set_ylabel("Projected Energy")
    ax3.set_title("Projected Energy vs. Time")
    ax3.legend()

    # --- Final Energy Convergence ---
    ax4 = fig.add_subplot(gs[1, 1])
    if len(estimated_energies) > 100:
        last_portion = estimated_energies[-100:]
        time_last_start = energy_time_axis[-100] if len(energy_time_axis) > 100 else 0
        time_last_end = energy_time_axis[-1] if len(energy_time_axis) > 0 else 0
        time_last_axis = np.linspace(time_last_start, time_last_end, len(last_portion))
        ax4.plot(time_last_axis, last_portion, label='QC-FCIQMC Energy (Final)', linewidth=2, color='darkblue', marker='o', markersize=3)
        ax4.axhline(y=exact_gs_energy, color='red', linestyle='--', label=f'Exact Energy ({exact_gs_energy:.4f})', linewidth=2)
        ax4.set_xlabel("Imaginary Time (τ)")
        ax4.set_ylabel("Estimated Energy")
        ax4.set_title("Energy Convergence (Final Steps)")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Not enough data\nfor convergence plot', transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title("Energy Convergence (Final Steps)")

    plt.tight_layout()
    plt.show()

    # --- Print Final Summary ---
    if len(estimated_energies) > 100:
        final_energy_avg = np.mean(estimated_energies[-100:])
        final_energy_std = np.std(estimated_energies[-100:])
        energy_error = abs(final_energy_avg - exact_gs_energy)
        print("\n============ SIMULATION SUMMARY ============")
        print(f"Final Avg. Energy (last 100): {final_energy_avg:.8f} ± {final_energy_std:.8f}")
        print(f"Exact Ground State Energy:      {exact_gs_energy:.8f}")
        print(f"Error:                          {energy_error:.8f}")
        print("============================================")
