import torch
import pickle
import matplotlib.pyplot as plt
import io


original_load = torch.load

def cpu_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return original_load(*args, **kwargs)

torch.load = cpu_load
try:
    with open('./outputs/output.pkl', 'rb') as f:
            content = pickle.load(f)
finally:
    # Restore original torch.load
    torch.load = original_load


loss_list = content['val_loss_list']
flat_list = [item for sublist in loss_list for item in sublist]

plt.plot(flat_list)

plt.show()