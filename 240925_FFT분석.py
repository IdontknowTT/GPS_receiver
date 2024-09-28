import numpy as np
import matplotlib.pyplot as plt


f = open('live_record(1.567G, 50MHz, 25dB, 5M)_230513_2013_102s_SDR_검증.bin', 'rb')
data = np.fromfile(f, dtype=np.int16, count = int(50e6*10*2))
f.close()
# 측정값 오류로 인한 값들 양옆 중간값으로 치환
for i in range(0,len(data),250001):
    data[i] = (data[i-2]+ data[i+2]) / 2
    data[i+1] = (data[i-1] + data[i+3]) / 2

real = data[::2]
imag = data[1::2]
signal = real + 1j*imag

# 샘플링 주파수 및 필요한 샘플 수 설정
fs = 50e6  # 샘플링 주파수 (50 MHz)
desired_time = 10  # 원하는 신호 길이 (초)
total_samples = int(fs * desired_time)


# 신호 길이 확인 및 자르기
if len(signal) < total_samples:
    print('신호 길이가 충분하지 않습니다. desired_time 변수를 줄여주세요.')
    total_samples = len(signal)
    
signal = signal[:total_samples]
print(total_samples)
# FFT 포인트 수 설정
N_fft = 2 ** int(np.floor(np.log2(len(signal))))

# FFT 계산
signal_fft = np.fft.fftshift(np.fft.fft(signal, N_fft))
freq_axis = np.linspace(-fs/2, fs/2, N_fft)

# 스펙트럼 크기 계산 (dB 단위)
spectrum_magnitude = 20 * np.log10(np.abs(signal_fft))

# 스펙트럼 플로팅
plt.figure(figsize=(10, 6))
plt.plot(freq_axis / 1e6, spectrum_magnitude)
plt.xlabel('주파수 (MHz)')
plt.ylabel('스펙트럼 크기 (dB)')
plt.title('원신호의 FFT 스펙트럼')
plt.grid(True)
plt.show()
