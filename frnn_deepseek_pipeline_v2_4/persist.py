
import torch, os

def save_state(core, path="frnn_state.pt"):
    torch.save({
        "M": core.frnn.M.detach().cpu(),
        "bank_keys": core.frnn.bank_keys.cpu(),
        "bank_vals": core.frnn.bank_vals.cpu(),
        "bank_used": core.frnn.bank_used.cpu(),
        "bank_ptr": core.frnn.bank_ptr.cpu(),
    }, path)

def load_state(core, path="frnn_state.pt"):
    if not os.path.exists(path): return False
    s=torch.load(path, map_location="cpu")
    core.frnn.M.data.copy_(s["M"])
    core.frnn.bank_keys.copy_(s["bank_keys"])
    core.frnn.bank_vals.copy_(s["bank_vals"])
    core.frnn.bank_used.copy_(s["bank_used"])
    core.frnn.bank_ptr.copy_(s["bank_ptr"])
    return True
