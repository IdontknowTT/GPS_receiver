

import numpy as np
import matplotlib.pyplot as plt
from collections import deque  

sats = [(1, 5), (2, 6), (3, 7), (4, 8), (0, 8), (1, 9), (0, 7), (1, 8), (2, 9), (1, 2),
            (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 3), (1, 4), (2, 5), (3, 6),
            (4, 7), (5, 8), (0, 2), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (0, 5), (1, 6),
            (2, 7), (3, 8), (4, 9), (3, 9), (0, 6), (1, 7), (3, 9)]
g1tap = [2,9]
g2tap = [1,2,5,7,8,9]
# -5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
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

# 코드 미리 만들어두기(시간복잡도 줄이려고)
codes = []
for i in range(37):
    codes.append(getCode(sats[i]))
    

# PRN 번호 선택 (예: PRN 1)
prn = 1
prn_sequence = codes[prn]

# 주파수 설정
chip_frequency = 1.023e6  # 칩 주파수 (Hz)

# 사각 펄스 신호 생성
t = np.arange(0, len(prn_sequence) / chip_frequency, 1 / chip_frequency)
duty_cycle = 0.5  # 듀티 사이클 (0과 1 사이)
square_wave = np.where(np.mod(t, 1 / chip_frequency) < duty_cycle, 1, -1)

# C/A 코드 생성
ca_code = np.tile(prn_sequence, int(len(t) / len(prn_sequence)))

plt.plot(t[:20], ca_code[:20])
plt.title('First 20 Samples of Generated C/A Code using Square Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


        
    
    
