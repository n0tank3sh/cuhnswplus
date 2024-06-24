#include <cuhnsw.hpp>
#include <fstream>
#include <thread>
#include <log.hpp>
#include <fstream>
#include <filesystem>
#include <random>

namespace cuhnsw {
  constexpr uint32_t BLOCK_SIZE = 30000000;
  Index::Index(std::string storage_prefix, std::string config_file)
    :
    storage_prefix(storage_prefix), 
    config_file(config_file)
    {}

  void Index::SetData(const float* data, int num_data, int dims) {
    int offset = 0;
    shard_size = BLOCK_SIZE / (dims * sizeof(float));
    clusters = (num_data + shard_size - 1)/shard_size;
    for(int i = 0; i < clusters; i++) {
      std::string file_name = "data_" + std::to_string(i) + ".bin";
      if(!std::filesystem::exists(storage_prefix)) {
        std::filesystem::create_directory(storage_prefix);
      }
      file_name = storage_prefix + "/" + file_name;
      std::ofstream ofs(file_name, std::ios_base::binary | std::iostream::trunc);
      ofs.write(reinterpret_cast<const char*>(&shard_size), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&dims), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&data[offset]), sizeof(float) * (std::min(shard_size, num_data - offset/dims) * dims));
      offset += shard_size * dims;
      ofs.close();
    }
  }

  void Index::Search(const float* qdata, const int num_queries, const int topk, const int ef_search,
  int* nns, float* distances, int* found_cnt) {
    std::mt19937 level_generator;
    std::uniform_real_distribution distrib;
    std::vector<std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, 
    std::greater<std::pair<float, int>>>> pq_list(num_queries);

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
      float* data;
      ifs.read(reinterpret_cast<char*>(&num_data), sizeof(int));
      ifs.read(reinterpret_cast<char*>(&dims), sizeof(int));
      data = new float[num_data * dims];
      ifs.read(reinterpret_cast<char*>(data), sizeof(float) * dims * num_data);
      cuhnsw::CuHNSW client;
      client.Init(config_file);
      client.SetData(data, num_data, dims);
      std::vector<int> levels(num_data);
      for(int i = 0; i < num_data; i++) {
        levels[i] = (int)(-log(distrib(level_generator)) * 0.5);
      }
      client.SetRandomLevels(levels.data());
      client.BuildGraph();
      client.SearchGraph(qdata, num_queries, topk, ef_search, 
        cluster_nns.data(), cluster_dist.data(), cluster_found_cnt.data());

      for(int q = 0; q < num_queries; q++) {
        for(int j = 0; j <  cluster_found_cnt[q]; j++) {
          found_cnt[q] += cluster_found_cnt[q];
          found_cnt[q] = std::min(topk, found_cnt[q]);
          int global_id = cluster_nns[topk * q + j] + (cluster * shard_size);
          pq_list[q].push(std::make_pair(cluster_dist[topk * q + j], global_id));
        }
      }
      ifs.close();
    }
    for(int i = 0; i < num_queries; i++) {
      int j = 0;
      while(j < topk) {
        auto p = pq_list[i].top();
        std::cout << p.first << ':' << p.second << ';';
        pq_list[i].pop();
        nns[i * topk + j] = p.second;
        distances[i * topk + j] = p.first;
        j++;
      }
      std::cout << std::endl;
    }
  }
}