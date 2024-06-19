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
    client.BuildGraph();
    std::vector<int> nns(100, 0);
    std::vector<float> dist(100, INFINITY);
    std::vector<int> found_cnt(5);
    client.SearchGraph(&data[14], 5, 20, 30, nns.data(), dist.data(), found_cnt.data());
    for(auto e: dist) {
        EXPECT_NE(e, INFINITY);
    }
}

TEST(SetData_test, SetData) {  
    cuhnsw::CuHNSW client;
    EXPECT_TRUE(client.Init("TestConfig.json"));
    std::mt19937 gen;
    std::uniform_real_distribution distrib(0.0, 1.0);
    std::vector<float> data;
    client.SelectGPU(0);    
    std::vector<int> levels(100);
    int dim = 8;
    for(int i = 0; i < 100; i++) {
        for(int j = 0; j < dim; j++) {
            data.push_back(distrib(gen));
        }
        levels[i] = (int)(-log(distrib(gen)) * 0.5);
    }
    std::vector<int> labels(100, -1);
    client.SetData(&data[0], data.size() / dim, dim);
    client.SetRandomLevels(levels.data());
    client.BuildGraph();
    std::vector<int> nns(100, 0);
    std::vector<float> dist(100, INFINITY);
    std::vector<int> found_cnt(5);
    client.SearchGraph(&data[14], 5, 20, 30, nns.data(), dist.data(), found_cnt.data());
    for(auto e: dist) {
        EXPECT_NE(e, INFINITY);
    }
}
