#include <cuhnswplus.hpp>
#include <gtest/gtest.h>
#include <random>

TEST(Index_test, Index_SetData) {
  std::mt19937 rnd;
  std::uniform_real_distribution distrib(0.0f, 1.0f);
  cuhnswplus::Index index("./testdata/", "TestConfig.json");
  int dim = 32 * 32 * 3;
  std::vector<float> test_data;
  for(int i = 0; i < 3000; i++) {
    for(int j = 0; j < dim; j++) {
      test_data.push_back(distrib(rnd));
    }
  }
  index.SetData(&test_data[0], test_data.size()/dim, dim);
  std::vector<int> nns(4 * 10);
  std::vector<float> distances(4 * 10);
  std::vector<int> found_cnt(4);
  index.Search(&test_data[60 * dim], 1, 10, 10, nns.data(), 
    distances.data(), found_cnt.data());
  for(int i = 0; i < found_cnt[0]; i++) {
    std::cout << distances[i] << ":" << nns[i] << ';';
  }
  std::cout << std::endl;
}