
import torch

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("BF16 supported:", torch.cuda.is_bf16_supported())

print(
    "GPU compute capability:",
    torch.cuda.get_device_capability(0)
)

print(
    "Total GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    ),
    "GB"
)
