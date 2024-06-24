// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <set>
#include <unordered_set>
#include <random>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <queue>
#include <functional>
#include <vector>
#include <unordered_map>
#include "thread_pool.hpp"

#include "log.hpp"

namespace cuhnsw {

class LevelGraph {
 public:
  explicit LevelGraph(int max_elements = 0) {
    logger_ = CuHNSWLogger().get_logger();
    nodes_idmap_.resize(max_elements , -1);
  }

  ~LevelGraph() {}

  void SetNodes(std::vector<int>& nodes, int num_data, int ef_construction) {
    nodes_ = nodes;
    num_nodes_ = nodes_.size();
    neighbors_.clear();
    neighbors_.resize(num_nodes_);
    for (int i = 0; i < num_nodes_; ++i)
      nodes_idmap_[nodes[i]] = i;
  }

  void AddNode(int node) {
    num_nodes_++;
    nodes_.push_back(node);
    neighbors_.resize(nodes_.size());
    nodes_idmap_[node] = nodes_.size() - 1;
  }

  const std::vector<std::pair<float, int>>& GetNeighbors(int node) const  {
    int nodeid = GetNodeId(node);
    return neighbors_[nodeid];
  }

  const std::vector<int>& GetNodes() const {
    return nodes_;
  }

  void ClearEdges(int node) {
    neighbors_[GetNodeId(node)].clear();
  }

  void AddEdge(int src, int dst, float dist) {
    if (src == dst) return;
    int srcid = GetNodeId(src);
    neighbors_[srcid].emplace_back(dist, dst);
  }

  inline int GetNodeId(int node) const {
    int nodeid = nodes_idmap_.at(node);
    if (not(nodeid >= 0 and nodeid < num_nodes_)) {
      throw std::runtime_error(
          fmt::format("[{}:{}] invalid nodeid: {}, node: {}, num_nodes: {}",
            __FILE__, __LINE__, nodeid, node, num_nodes_));
    }
    return nodeid;
  }
  
  void LoadGraphVec(std::vector<int>& graph_vec, std::vector<int>& deg, int max_m) {
    const std::vector<int>& nodes = GetNodes();
    auto loader = [&](int start, int end) {
      if(start >= nodes.size()) return;
      for(int i = start; i < end; i++) {
        auto& neighbors = GetNeighbors(nodes[i]);
        deg[i] = neighbors.size();
        for(int j = 0; j < deg[i]; j++) {
          graph_vec[i * max_m + j] = GetNodeId(neighbors[j].second);
        }
      }
    };
    auto& thread_pool = ThreadPoolSingleton::get_instance();
    int num_thread = 12;
    int chunk_size = (nodes.size() + num_thread - 1)/num_thread;
    for(int i = 0; i < num_thread; i++) {
      int start = i * chunk_size;
      int end = std::min((int)nodes.size(), (i + 1) * chunk_size);
      auto fut = thread_pool.submit_task([loader, start, end]{loader(start, end);});
    }
    thread_pool.wait();
  }

  void UnLoadGraphVec(const std::vector<int>& graph_vec, const std::vector<int>& deg, const std::vector<float>& dist, int max_m) {
    const std::vector<int>& nodes = GetNodes();
    auto unloader = [&](int start, int end) {
      for(int i = start; i < end; i++) {
        ClearEdges(nodes[i]);
        for(int j = 0; j < deg[i]; j++) {
          AddEdge(nodes[i], nodes[graph_vec[i * max_m +j]], dist[i * max_m + j]);
        }
      }
    };  
    auto& thread_pool = ThreadPoolSingleton::get_instance();
    int num_thread = 12;
    int chunk_size = (nodes.size() + num_thread - 1)/num_thread;
    for(int i = 0; i < num_thread; i++) {
      int start = i * chunk_size;
      int end = std::min((int)nodes.size(), (i + 1) * chunk_size);
      auto fut = thread_pool.submit_task([unloader, start, end]{unloader(start, end);});
    }
    thread_pool.wait();
  }

  void ShowGraph() {
    for (int i = 0; i < num_nodes_; ++i) {
      std::cout << std::string(50, '=') << std::endl;
      printf("nodeid %d: %d\n", i, nodes_[i]);
      for (auto& nb: GetNeighbors(nodes_[i])) {
        printf("neighbor id: %d, dist: %f\n",
            nb.second, nb.first);
      }
      std::cout << std::string(50, '=') << std::endl;
    }
  }

 private:
  std::shared_ptr<spdlog::logger> logger_;
  std::vector<int> nodes_;
  std::vector<std::vector<std::pair<float, int>>> neighbors_;
  int num_nodes_ = 0;
  int max_elements = 0;
  std::vector<int> nodes_idmap_;
};  // class LevelGraph

} // namespace cuhnsw
