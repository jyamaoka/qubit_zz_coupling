import numpy as np

def make_population(expect):
    pop = (1 + expect) / 2  
    return pop

def exp_decay(t, a, T1, c):
    return a * np.exp(-t / T1) + c

def ramsey(t, A, T2_star, f, phi, C):
    return A * np.exp(-t/T2_star) * np.cos(2*np.pi*f*t + phi) + C