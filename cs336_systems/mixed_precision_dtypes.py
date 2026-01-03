import json
import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def main(output_path: str = "mixed_precision_dtypes.json") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script for mixed-precision profiling.")

    device = torch.device("cuda")
    dtype = torch.float16

    model = ToyModel(in_features=4, out_features=16).to(device)

    # Capture intermediate activations via forward hooks
    activations = {}

    def save_activation(name):
        def hook(module, inp, out):
            activations[name] = out
        return hook

    h_fc1 = model.fc1.register_forward_hook(save_activation("fc1"))
    h_ln = model.ln.register_forward_hook(save_activation("ln"))

    x = torch.randn(2, 4, device=device)

    # Forward under autocast to FP16
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(x)

    # Loss and backward in FP32 (standard pattern)
    loss = logits.sum()
    loss.backward()

    # Collect dtypes of interest
    param_dtypes = {name: str(param.dtype) for name, param in model.named_parameters()}

    results = {
        "fc1_weight_dtype": str(model.fc1.weight.dtype),
        "fc1_out_dtype": str(activations["fc1"].dtype),
        "ln_out_dtype": str(activations["ln"].dtype),
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "fc1_grad_dtype": str(model.fc1.weight.grad.dtype),
        "all_param_dtypes": param_dtypes,
    }

    # Clean up hooks
    h_fc1.remove()
    h_ln.remove()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved dtype information to {output_path}")


if __name__ == "__main__":
    main()
