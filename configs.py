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


wire_cfg = {
    '1': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/1x.jpg',
        'threshold': 100,
    },
    '2': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/2x.jpg',
        'threshold': 100,
    },
    '3': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/3x.jpg',
        'threshold': 100,
    },
    '4': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/4x.jpg',
        'threshold': 100,
    },
    '8': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/8x.jpg',
        'threshold': 155,
    },
    '9': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/9x.jpg',
        'threshold': 155,
    },
    '10': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/10x.jpg',
        'threshold': 210, # bad threshold..
    },
    '11': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/11x.jpg',
        'threshold': 125,
    },
    '13': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/13x.jpg',
        'threshold': 125,
    },
    '16': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/16x.jpg',
        'threshold': 125,
    },
    '17': {
        'wire_path': '/mnt/disk1/cubox_dataset/patterns/17x.jpg',
        'threshold': 125,
    },
    '100': {
        'wire_path': None,
        'threshold': None,
    },
}


transform_config = {
    'box': {
        
    }

}