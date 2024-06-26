// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include <cuhnswplus.hpp>
#include <fstream>
#include <thread>
#include <log.hpp>
#include <fstream>
#include <filesystem>
#include <random>
#include <json/json.h>
#include <thread_pool.hpp>

namespace cuhnswplus {
  constexpr uint32_t BLOCK_SIZE = 30000000;
  Index::Index(std::string storage_prefix, std::string config_file)
    :
    storage_prefix(storage_prefix), 
    config_file(config_file)
    {
      std::ifstream in(config_file.c_str());
      if (not in.is_open()) throw;

      std::string str((std::istreambuf_iterator<char>(in)),
          std::istreambuf_iterator<char>());
      std::string err_cmt;
      auto _opt = json11::Json::parse(str, err_cmt);
      if (not err_cmt.empty()) throw;
      level_mult = _opt["level_mult"].int_value();
    }

  void Index::SetData(const float* data, int num_data, int dims) {
    int offset = 0;
    shard_size = BLOCK_SIZE / (dims * sizeof(float));
    clusters = (num_data + shard_size - 1)/shard_size;
    std::cout << shard_size << ',' << clusters << std::endl;
    for(int i = 0; i < clusters; i++) {
      std::string file_name = "data_" + std::to_string(i) + ".bin";
      if(!std::filesystem::exists(storage_prefix)) {
        std::filesystem::create_directory(storage_prefix);
      }
      file_name = storage_prefix + "/" + file_name;
      std::ofstream ofs(file_name, std::ios_base::binary | std::iostream::trunc);
      int size = std::min(shard_size, num_data - offset/dims);
      ofs.write(reinterpret_cast<const char*>(&size), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&dims), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&data[offset]), sizeof(float) * size * dims);
      std::cout << std::min(shard_size, num_data - offset/dims) << std::endl;
      offset += shard_size * dims;
      ofs.close();
    }
  }

  void Index::Search(const float* qdata, const int num_queries, const int topk, const int ef_search,
  int* nns, float* distances, int* found_cnt) {
    std::mt19937 level_generator;
    std::uniform_real_distribution distrib(0.0, 1.0);
    std::vector<std::pair<float, int>> m_store;
    std::vector<std::priority_queue<std::pair<float,int>, std::vector<std::pair<float, int>>, 
    std::greater<std::pair<float, int>>>> pq_list(num_queries);
    auto& pool = ThreadPoolSingleton::get_instance();

    for(int cluster = 0; cluster < clusters; cluster++) {
      std::vector<float> cluster_dist(num_queries * topk);
      std::vector<int> cluster_nns(num_queries * topk);
      std::vector<int> cluster_found_cnt(num_queries);
      std::string file_name = "data_" + std::to_string(cluster) + ".bin";
      if(!std::filesystem::exists(storage_prefix)) {
        throw std::runtime_error("");
      }
      file_name = storage_prefix + "/" + file_name;
      std::ifstream ifs(file_name, std::ios_base::binary);
      int num_data, dims;
      std::vector<float> data;
      ifs.read(reinterpret_cast<char*>(&num_data), sizeof(int));
      ifs.read(reinterpret_cast<char*>(&dims), sizeof(int));
      data.resize(num_data * dims);
      ifs.read(reinterpret_cast<char*>(data.data()), sizeof(float) * dims * num_data);
      cuhnswplus::CuHNSW client;
      client.Init(config_file);
      client.SetData(data.data(), num_data, dims);
      std::vector<int> levels(num_data);
      int num_threads = pool.get_thread_count();
      int chunk_size = (num_data + num_threads - 1)/num_threads;
      for(int i = 0; i < num_threads; i++) {
        int start = i * chunk_size;
        int end = std::min((i + 1) * chunk_size, num_data);
        auto fut = pool.submit_task([&]{
          for(int i = start; i < end; i++) {
            levels[i] = (int)(-log(distrib(level_generator)) * level_mult);
          }
        });
      }
      pool.wait();
      client.SetRandomLevels(levels.data());
      client.BuildGraph();
      client.SearchGraph(qdata, num_queries, topk, ef_search, 
        cluster_nns.data(), cluster_dist.data(), cluster_found_cnt.data());

      for(int q = 0; q < num_queries; q++) {
        for(int j = 0; j <  cluster_found_cnt[q]; j++) {
          found_cnt[q] += cluster_found_cnt[q];
          found_cnt[q] = std::min(topk, found_cnt[q]);
          int global_id = cluster_nns[topk * q + j] + (cluster * shard_size);
          float dist = cluster_dist[topk * q + j];
          pq_list[q].push(std::make_pair(std::abs(dist), m_store.size()));
          m_store.emplace_back(dist, global_id);
        }
      }
      ifs.close();
    }
    for(int i = 0; i < num_queries; i++) {
      int j = 0;
      while(j < topk) {
        auto p = m_store[pq_list[i].top().second];
        pq_list[i].pop();
        nns[i * topk + j] = p.second;
        distances[i * topk + j] = p.first;
        j++;
      }
    }
  }
}