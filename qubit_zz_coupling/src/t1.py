def solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=False):
    from qutip import mesolve
    from scipy.optimize import curve_fit

    result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
    pop = make_population(result.expect[0])
    fit_par, _ = curve_fit(exp_decay, tlist, pop, p0=[1.0, 20, 0])

    if ret_pop:
        return fit_par, pop
    
    return fit_par

def plot_t1(H, psi0, tlist, c_ops, e_ops, plable_Qbit, system_params):
    import matplotlib.pyplot as plt

    fit_par, pop = solve_t1(H, psi0, tlist, c_ops, e_ops, ret_pop=True)
    
    fig, ax = plt.subplots()
    ax.plot(tlist, pop, 'bo', alpha=0.5, label='Data')
    ax.plot(tlist, exp_decay(tlist, *fit_par), 'r-', 
            label=f'Fit: T1 = {fit_par[1]:.2f} μs')
    ax.set_title(f'T1 - {plable_Qbit} (Jzztls = {system_params["JTLS"]}, Jzz = {system_params["Jzz"]}, Jxx = {system_params["Jxx"]})')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Population |1⟩')
    ax.legend()
    ax.grid(True)
    return ax 

def make_population(expect):
    return (1 + expect) / 2  

def exp_decay(t, a, T1, c):
    return a * np.exp(-t / T1) + c