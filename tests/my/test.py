import egl_probe
valid_gpu_devices = egl_probe.get_available_devices()
print(valid_gpu_devices)