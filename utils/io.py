import tables


def load_hdf5(fname):
    with tables.open_file(fname) as hdf5_:
        data = hdf5_.root.image[:]
    return data


def save_hdf5(data, ofname, compress=False):
    with tables.open_file(ofname, 'w') as hdf5_:
        atom = tables.Atom.from_dtype(data.dtype)
        shape = data.shape
        if (compress):
            filters = tables.Filters(complevel=5, complib='zlib')
            ca = hdf5_.create_carray(hdf5_.root, 'image', atom, shape, filters=filters)
        else:
            ca = hdf5_.create_carray(hdf5_.root, 'image', atom, shape)

        ca[:] = data[:]

