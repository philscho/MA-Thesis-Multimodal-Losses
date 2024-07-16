import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    # Print the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print(f"CUDA is available with {num_cuda_devices} device(s)")

    # Print the current CUDA device
    current_cuda_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_cuda_device}")

    # Print the name of the current CUDA device
    current_cuda_device_name = torch.cuda.get_device_name(current_cuda_device)
    print(f"Current CUDA device name: {current_cuda_device_name}")

    # Print the compute capability of the current CUDA device
    compute_capability = torch.cuda.get_device_capability(current_cuda_device)
    print(f"Compute capability: {compute_capability}")
else:
    print("CUDA is not available on your system.")
