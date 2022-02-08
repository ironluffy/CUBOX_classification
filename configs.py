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
        "synth2all": { # CAUTION! use synth aug!!
            "train": ["none"],
            "val": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"],
            "test": ["none", "semitransparent", "wiredense", "wiremedium", "wireloose"]
        },  
        # "all2all": {
	    # "train": ["None", "SemiTransparent", "WireDense", "WireMedium", "WireLoose"],
	    # "val": ["None", "SemiTransparent", "WireDense", "WireMedium", "WireLoose"],
	    # "test": ["None", "SemiTransparent", "WireDense", "WireMedium", "WireLoose"]
        # },

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
        "none2loose": {
            "train": ["none"],
            "val": ["wireloose"],
            "test": ["wireloose"],
        },
        "none2med": {
            "train": ["none"],
            "val": ["wiremedium"],
            "test": ["wiremedium"],
        },
        "none2dense": {
            "train": ["none"],
            "val": ["wiredense"],
            "test": ["wiredense"],
        },
    },
}

data_config["cubox_singlewire"] = data_config["cubox"]