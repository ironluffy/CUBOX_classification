import os

data_config = {
    "cubox": {
        "none2all": {
            "train": ["none"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"]
        },
        "none2none": {
            "train": ["none"],
            "val": ["none"],
            "test": ["none"]
        },
        "all2all": {
            "train": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"]
        },
        "semi2all": {
            "train": ["semitransparent"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
        },
        "dense2all": {
            "train": ["wiredense"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
        },
        "medium2all": {
            "train": ["wiremedium"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
        },
        "loose2all": {
            "train": ["wireloose"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
        },
    },
}
