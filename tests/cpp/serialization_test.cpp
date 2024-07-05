#include <cuhnswplus.hpp>
#include <gtest/gtest.h>

TEST(Serialization_test, Serialization) {
    cuhnswplus::CuHNSW client;
    EXPECT_TRUE(client.Init("TestConfig.json"));
    std::mt19937 gen;
    std::uniform_real_distribution distrib(0.0, 1.0);
    std::vector<float> data;  
    std::vector<int> levels(100);
    int dim = 8;
    for(int i = 0; i < 100; i++) {
        for(int j = 0; j < dim; j++) {
            data.push_back(distrib(gen) * 10.0f);
        }
        levels[i] = (int)(-log(distrib(gen)) * 0.4);
    }
    std::cout << std::endl << std::endl;
    client.SetData(&data[0], data.size() / dim, dim);
    client.SetRandomLevels(levels.data());
    client.BuildGraph();
    std::vector<int> nns(100, 0);
    std::vector<float> dist(100, INFINITY);
    std::vector<int> found_cnt(5);
    client.SearchGraph(&data[0], 5, 20, 30, nns.data(), dist.data(), found_cnt.data());
    client.SaveIndex("test.index");
    client.LoadIndex("test.index");
    std::vector<int> nns1(100, 0);
    std::vector<float> dist1(100, INFINITY);
    std::vector<int> found_cnt1(5);
    client.SearchGraph(&data[0], 5, 20, 30, nns1.data(), dist1.data(), found_cnt1.data());
    for(int i = 0; i < nns.size(); i++) {
      EXPECT_EQ(nns[i], nns1[i]);
    }
}