from cuhnsw import CuHNSWIndex
import numpy as np
import h5py

def main():
  res_file = h5py.File("../res/glove-50-angular.hdf5", "r")
  data = res_file["train"][:, :].astype(np.float32)
  test = res_file["test"][:, :].astype(np.float32)
  res_file.close()
  ch0 = CuHNSWIndex("example_data/", "example_config.json")
  ch0.set_data(data)
  ch0.search(test, topk=30, ef_search=300)

if __name__ == "__main__":
  main()