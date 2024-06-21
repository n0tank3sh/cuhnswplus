from cuhnsw import CuHNSWIndex
import numpy as np

def main():
  data = np.random.rand(100, 30)
  ch0 = CuHNSWIndex("example_data/", "example_config.json")
  ch0.set_data(data)
  return 0

if __name__ == "__main__":
  main()