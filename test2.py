import h5py

f = h5py.File('solveDataRepeats2.hdf5','r')

print(f["faces"][0])
print(f["moves"][0])

f.close()