import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
from collections import deque  
import random as rand

sats = [(1, 5), (2, 6), (3, 7), (4, 8), (0, 8), (1, 9), (0, 7), (1, 8), (2, 9), (1, 2),
            (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 3), (1, 4), (2, 5), (3, 6),
            (4, 7), (5, 8), (0, 2), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (0, 5), (1, 6),
            (2, 7), (3, 8), (4, 9), (3, 9), (0, 6), (1, 7), (3, 9)]
g1tap = [2,9]
g2tap = [1,2,5,7,8,9]

def getCode(satsNum):
    
    g1 = deque(1 for i in range(10))
    g2 = deque(1 for i in range(10))
    
    # result
    g = []
    
    # Generating 1023 chips(One C/A sequence)
    for i in range(1023):
        val = (g1[9] + g2[satsNum[0]] + g2[satsNum[1]]) % 2
        g.append(val)
        
        #shift g1
        g1[9] = sum(g1[i] for i in g1tap) % 2
        g1.rotate()
        
        #shift g2
        g2[9] = sum(g2[i] for i in g2tap) % 2
        g2.rotate()
    # 0 => -1
    for n,i in enumerate(g):
            if i==0:
                g[n]=-1
        
    return g


def generate_signal(code_delay, doppler_freq, svNumber, code_freq=1.023e6, oversampling_factor=255750, signal_length=1023):
    # Generate C/A code
    ca_code = np.array(getCode(sats[svNumber]))
    
    # Apply code delay
    delayed_code = np.roll(ca_code, code_delay)
    
    # Generate oversampled signal
    t = np.arange(signal_length * 4)
    code_phase = 2 * np.pi * code_freq * t / oversampling_factor
    oversampled_signal = np.exp(1j * (code_phase + 2 * np.pi * doppler_freq * t / oversampling_factor))
    
    # Multiply with delayed C/A code
    signal = delayed_code * oversampled_signal[:signal_length]
    
    return signal


def acquisition(signal, code_delay_range, doppler_freq_range, svNumber, doppler_freq_step=500, oversampling_factor=255750, signal_length=1023):
    max_corr = 0
    max_code_delay = 0
    max_doppler_freq = 0
    
    corr_matrix = np.zeros((len(code_delay_range), len(doppler_freq_range)))
    """
    수정 전
    for i, code_delay in enumerate(code_delay_range):
        for j, doppler_freq in enumerate(doppler_freq_range):
            # Generate the reference signal with given code delay and Doppler frequency
            reference_signal = generate_signal(code_delay, doppler_freq,svNumber, oversampling_factor=oversampling_factor, signal_length=signal_length)
            
            # Compute correlation between received signal and reference signal
            correlation = np.abs(correlate(signal, reference_signal, mode='valid'))
            
            # Find maximum correlation
            peak_corr = np.max(correlation)
            corr_matrix[i, j] = peak_corr
            
            if peak_corr > max_corr:
                max_corr = peak_corr
                max_code_delay = code_delay
                max_doppler_freq = doppler_freq
                """
    for i, doppler_freq in enumerate(doppler_freq_range):
        reference_signal = generate_signal(code_delay_range[0], doppler_freq,svNumber, oversampling_factor=oversampling_factor, signal_length=signal_length)
        for j in range(len(code_delay_range)):
            correlation = np.abs(correlate(signal, reference_signal, mode='valid'))
            
            # Find maximum correlation
            peak_corr = np.max(correlation)
            corr_matrix[j, i] = peak_corr
            
            if peak_corr > max_corr:
                max_corr = peak_corr
                max_code_delay = code_delay_range[j]
                max_doppler_freq = doppler_freq
            reference_signal = np.roll(reference_signal,1) # 이거 왜 1만큼 움직여야 되는거지???
    return max_corr, max_code_delay, max_doppler_freq, corr_matrix


# Parameters
code_delay_range = range(-200, 201)  # Range of code delay in chips
doppler_freq_range = np.linspace(-5000, 5000, 500)  # Adjusted range of Doppler frequency in Hz
code_freq = 1.023e6  # Code frequency in Hz
svNumber = rand.randint(0,36)
# Generate received signal
true_code_delay = rand.randint(-200,200)
true_doppler_freq = (rand.randint(-10,10))*500
received_signal = generate_signal(true_code_delay, true_doppler_freq, svNumber, code_freq=code_freq)

# Perform acquisition
max_corr, estimated_code_delay, estimated_doppler_freq, corr_matrix = acquisition(received_signal, code_delay_range, doppler_freq_range, svNumber)

print("Target SV Number:", svNumber)
print("Maximum correlation:", max_corr)
print("Estimated code delay:", estimated_code_delay)
print("Estimated Doppler frequency:", estimated_doppler_freq)


# Plotting
Code_Delay, Doppler_Freq = np.meshgrid(code_delay_range, doppler_freq_range)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Code_Delay, Doppler_Freq, corr_matrix.T, cmap='viridis')
ax.set_xlabel('Code Delay')
ax.set_ylabel('Doppler Frequency (Hz)')
ax.set_zlabel('Correlation')
ax.set_title('Correlation vs Code Delay and Doppler Frequency')
plt.show()

def phase_find_pll(init_sine, sin_map, cos_map, n):
    # Constants
    sample_rate = 16000
    start_idx = (sample_rate * (n - 1))
    end_idx = sample_rate * n

    # Extract relevant portion of init_sine, sin_map, and cos_map
    init_sine_portion = init_sine[start_idx:end_idx]
    sin_map_portion = sin_map[start_idx:end_idx]
    cos_map_portion = cos_map[start_idx:end_idx]

    # Quadrature and Inphase components
    Q = init_sine_portion * cos_map_portion
    I = init_sine_portion * sin_map_portion

    # Calculate Quadrature and Inphase sums
    Qps = np.sum(Q) / len(Q)
    Ips = np.sum(I) / len(I)

    # Calculate phase using arctan2
    calc_phase = np.arctan2(Qps, Ips)

    return calc_phase

# Main function
if __name__ == "__main__":
    
    sample_rate = 16000
    # Parameters
    code_delay_range = range(-200, 201)  # Range of code delay in chips
    doppler_freq_range = np.linspace(-5000, 5000, 500)  # Adjusted range of Doppler frequency in Hz
    code_freq = 1.023e6  # Code frequency in Hz

    # Generate received signal
    true_code_delay = 50
    true_doppler_freq = 1000
    received_signal = generate_signal(true_code_delay, true_doppler_freq, code_freq=code_freq)

    # Perform acquisition
    max_corr, estimated_code_delay, estimated_doppler_freq, corr_matrix = acquisition(received_signal, code_delay_range, doppler_freq_range)

    print("Maximum correlation:", max_corr)
    print("Estimated code delay:", estimated_code_delay)
    print("Estimated Doppler frequency:", estimated_doppler_freq)

    # Generate Sin and Cos Maps
    sin_map = np.sin(2 * np.pi * doppler_freq_range / sample_rate)
    cos_map = np.cos(2 * np.pi * doppler_freq_range / sample_rate)

    # Perform PLL
    init_sine = received_signal[:sample_rate]  # Take initial portion of the received signal
    n = 1  # Time index
    estimated_phase = phase_find_pll(init_sine, sin_map, cos_map, n)
    print("Estimated phase:", estimated_phase)