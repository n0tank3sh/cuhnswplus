from cuhnswplus import CuHNSWIndex
import numpy as np
import time
import h5py
from cuhnswplus import aux
LOGGER = aux.get_logger()

def main():
  res_file = h5py.File("../res/mnist-784-euclidean.hdf5", "r")
  data = res_file["train"][:, :].astype(np.float32)
  test = res_file["test"][:, :].astype(np.float32)
  #data /= np.linalg.norm(data, axis=1)[:, None]
  #test /= np.linalg.norm(test, axis=1)[:, None]
  ch0 = CuHNSWIndex("example_data/", "example_config.json")
  ch0.set_data(data)
  ef_search=300
  topk = 100
  start = time.time()
  neighbors = res_file["neighbors"][:, :topk].astype(np.int32)
  res_file.close()
  pred_nn, _, _ = ch0.search(test, topk, ef_search)
  el0 = time.time() - start
  LOGGER.info("elapsed for inferencing %d queries of top@%d (ef_search: %d): "
              "%.4e sec", test, topk, ef_search, el0)
  accs = []
  for _pred_nn, _gt_nn in zip(pred_nn, neighbors):
    intersection = set(_pred_nn) & set(_gt_nn)
    acc = len(intersection) / float(topk)
    accs.append(acc)
  LOGGER.info("accuracy mean: %.4e, std: %.4e", np.mean(accs), np.std(accs))

if __name__ == "__main__":
  main()