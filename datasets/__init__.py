# datasets/__init__.py
from .mvtec import MVTecDataset
from .msdd import MSDDDataset
from .ospd import OSPDDataset
from .csdd import CSDDDataset

def get_dataset(name, root="data", **kwargs):
    if name == "mvtec":
        return MVTecDataset(root=os.path.join(root, "mvtec"), **kwargs)
    elif name == "msdd":
        return MSDDDataset(root=os.path.join(root, "msdd"), **kwargs)
    elif name == "ospd":
        return OSPDDataset(root=os.path.join(root, "ospd"), **kwargs)
    elif name == "csdd":
        return CSDDDataset(root=os.path.join(root, "csdd"), **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
