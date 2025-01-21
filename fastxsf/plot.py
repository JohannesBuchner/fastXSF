from xraystan import load_pha
import numpy as np
import matplotlib.pyplot as plt
import sys

data = load_pha(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
plt.imshow(data['RMF'], cmap='viridis')
plt.colorbar()
plt.xlabel('Energy')
plt.ylabel('Energy channel')
plt.savefig(sys.argv[1] + '_rmf.pdf', bbox_inches='tight')
plt.close()

plt.title('exposure: %d area: %f' % (data['src_expo'], data['src_expoarea'] / data['src_expo']))
plt.plot(data['ARF'])
plt.ylabel('Sensitive area')
plt.xlabel('Energy channel')
plt.savefig(sys.argv[1] + '_arf.pdf', bbox_inches='tight')
plt.close()
