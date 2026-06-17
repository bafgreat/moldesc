from importlib.resources import files
from mofstructure import filetyper
import h5py

def load_data(name: str):
    resource = files("moldesc.data").joinpath(name)
    return filetyper.load_data(resource)

import h5py

def read_group(group):
    data = {}

    # Read attributes
    if group.attrs:
        data.update(dict(group.attrs))

    # Read datasets/groups
    for key in group:
        item = group[key]

        if isinstance(item, h5py.Dataset):
            value = item[()]
            if isinstance(value, bytes):
                value = value.decode()

            data[key] = value

        elif isinstance(item, h5py.Group):
            data[key] = read_group(item)

    return data

def load_hdf5(filename):
    with h5py.File(filename, "r") as f:
        return read_group(f)