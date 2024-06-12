#include <gtest/gtest.h>
#include <cuhnsw.hpp>
#include <random>

TEST(AddPoint_test, AddPoint) {
    cuhnsw::CuHNSW client;
    EXPECT_TRUE(client.Init("TestConfig.json"));
    std::mt19937 gen;
    std::uniform_real_distribution distrib(0.0, 1.0);
    std::vector<float> data;
    client.SelectGPU(0);
    client.SetDims(2);
    for(int i = 0; i < 50; i++) {
        for(int j = 0; j < 2; j++) {
            data.push_back(distrib(gen));
        }
        client.AddPoint(&data[i * 2], -1, -1);
    }
    data.clear();
    for(int i = 0; i < 20; i++) {
        for(int j = 0; j < 2; j++) {
            data.push_back(distrib(gen));
        }
    }
    std::vector<int> levels(20, -1);
    std::vector<int> labels(20, -1);
    client.AddPoints(&data[0], &levels[0], &labels[0], 20);
    client.BuildGraph();
    std::vector<int> nns(100, 0);
    std::vector<float> dist(100, INFINITY);
    std::vector<int> found_cnt(5);
    client.SearchGraph(&data[14], 5, 20, 30, nns.data(), dist.data(), found_cnt.data());
    for(auto e: dist) {
        EXPECT_NE(e, INFINITY);
    }
}