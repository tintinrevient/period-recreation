# CycleGAN

## Issues

Why is my CUDA GPU-Util ~70% when there are “No running processes found”?

To correct this, enabling Persistence Mode:
```bash
sudo nvidia-smi -pm 1
```

To find which process is using GPU:

```bash
sudo fuser -v /dev/nvidia*
```

You can use numba library to release all the gpu memory:

```bash
pip install numba
```

This will release all the memory

```python
from numba import cuda 
device = cuda.get_current_device()
device.reset()
```

## References
* https://keras.io/examples/generative/cyclegan/