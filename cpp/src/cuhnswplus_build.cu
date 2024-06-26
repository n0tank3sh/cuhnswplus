// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

#include "cuhnswplus.hpp"
#include "cuda_build_kernels.cuh"

namespace cuhnswplus {

void CuHNSW::GetDeviceInfo() {
  CHECK_CUDA(cudaGetDevice(&devId_));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, devId_));
  mp_cnt_ = prop.multiProcessorCount;
  major_ = prop.major;
  minor_ = prop.minor;
  cores_ = -1;
}
inline int GetCores(int major, int minor, int mp_cnt) {      
  int cores = -1;
  switch (major){
    case 2: // Fermi
      if (minor == 1) 
        cores = mp_cnt * 48;
      else 
        cores = mp_cnt * 32;
      break;
    case 3: // Kepler
      cores = mp_cnt * 192;
      break;
    case 5: // Maxwell
      cores = mp_cnt * 128;
      break;
    case 6: // Pascal
      if (minor == 1 or minor == 2) 
        cores = mp_cnt * 128;
      else if (minor == 0) 
        cores = mp_cnt * 64;
      else 
      //  DEBUG0("Unknown device type");
      break;
    case 7: // Volta and Turing
      if (minor == 0 or minor == 5) 
        cores = mp_cnt * 64;
      else 
     //   DEBUG0("Unknown device type");
      break;
    case 8: // Ampere
      if (minor == 0) 
        cores = mp_cnt * 64;
      else if (minor == 6) 
        cores = mp_cnt * 128;
      else 
    //    DEBUG0("Unknown device type");
      break;
    default:
   //   DEBUG0("Unknown device type"); 
      break;
  }
  if (cores == -1) cores = mp_cnt * 128;
  return cores;
}
void CuHNSW::SelectGPU(int gpu_id) {
  if(gpus.size() == 0) {
    int dcount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dcount));
    int device;
    gpus.resize(dcount);
    cudaDeviceProp prop;
    for(int i = 0; i < dcount; i++) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaGetDevice(&device));
      CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
      int mp_cnt = prop.multiProcessorCount;
      int major = prop.major;
      int minor = prop.minor; 
      int block_cnt = opt_["hyper_threads"].number_value() * (GetCores(major, minor, mp_cnt) / block_dim_);
      gpus[i] = block_cnt;
    }
  }
  assert(gpu_id < gpus.size());
  CHECK_CUDA(cudaSetDevice(gpu_id));
  block_cnt_ = gpus[gpu_id];
}

void CuHNSW::SetDims(int dims) {
  num_dims_ = dims;
}


void CuHNSW::BuildGraph() {
  visited_ = new bool[batch_size_ * num_data_];
  //for (int level = max_level_; level >= 0; --level) {
  //  DEBUG("build graph of level: {}", level);
  //  BuildLevelGraph(level);
  //}
  BuildLevelGraph(0);
}

void CuHNSW::BuildLevelGraph(int level) {
  
  int max_size = level_graphs_[0].GetNodes().size();
  thrust::device_vector<int> device_graph(max_m0_ * max_size);
  thrust::device_vector<float> device_distances(max_m0_ * max_size);
  thrust::device_vector<int> device_deg(max_size);
  thrust::device_vector<int> device_nodes(max_size);
  thrust::device_vector<int> device_visited_table(visited_table_size_ * block_cnt_, -1);
  thrust::device_vector<int> device_visited_list(visited_list_size_ * block_cnt_);
  thrust::device_vector<int> device_mutex(max_size, 0);
  thrust::device_vector<int64_t> device_acc_visited_cnt(block_cnt_, 0);
  thrust::device_vector<Neighbor> device_neighbors(ef_construction_ * block_cnt_);
  thrust::device_vector<int> device_cand_nodes(ef_construction_ * block_cnt_);
  thrust::device_vector<cuda_scalar> device_cand_distances(ef_construction_ * block_cnt_);
  thrust::device_vector<int> device_backup_neighbors(max_m0_ * block_cnt_);
  thrust::device_vector<cuda_scalar> device_backup_distances(max_m0_ * block_cnt_);
  thrust::device_vector<bool> device_went_through_heuristic(max_size, false);

  for(int l = max_level_; l >= 0; l--) {
    std::set<int> upper_nodes;
    std::vector<int> new_nodes;
    LevelGraph& graph = level_graphs_[l];
    const std::vector<int>& nodes = graph.GetNodes();
    int size = nodes.size();  
    int max_m = l > 0? max_m_: max_m0_;

    std::vector<int> graph_vec(size * max_m, 0);
    std::vector<int> deg(size, 0);
    if (l < max_level_) {
      LevelGraph& upper_graph = level_graphs_[l + 1];
      upper_graph.LoadGraphVec(graph_vec, deg, max_m);

    }

    for (auto& node: graph.GetNodes()) {
      if (upper_nodes.count(node)) continue;
      new_nodes.push_back(node);
    }
  
    // initialize entries
    std::vector<int> entries(new_nodes.size(), enter_point_);

    GetEntryPoints(new_nodes, entries, l, false);
    for (int i = 0; i < new_nodes.size(); ++i) {
      int srcid = graph.GetNodeId(new_nodes[i]);
      int dstid = graph.GetNodeId(entries[i]);
      graph_vec[max_m * srcid] = dstid;
      deg[srcid] = 1;
    }

    thrust::copy(graph_vec.begin(), graph_vec.end(), device_graph.begin());
    thrust::copy(deg.begin(), deg.end(), device_deg.begin());
    thrust::copy(nodes.begin(), nodes.end(), device_nodes.begin());

    BuildLevelGraphKernel<<<block_cnt_, block_dim_>>>(
      thrust::raw_pointer_cast(device_data_.data()),
      thrust::raw_pointer_cast(device_nodes.data()),
      num_dims_, size, max_m, dist_type_, save_remains_,
      ef_construction_,
      thrust::raw_pointer_cast(device_graph.data()),
      thrust::raw_pointer_cast(device_distances.data()),
      thrust::raw_pointer_cast(device_deg.data()),
      thrust::raw_pointer_cast(device_visited_table.data()),
      thrust::raw_pointer_cast(device_visited_list.data()),
      visited_table_size_, visited_list_size_,
      thrust::raw_pointer_cast(device_mutex.data()),
      thrust::raw_pointer_cast(device_acc_visited_cnt.data()),
      reverse_cand_,
      thrust::raw_pointer_cast(device_neighbors.data()),
      thrust::raw_pointer_cast(device_cand_nodes.data()),
      thrust::raw_pointer_cast(device_cand_distances.data()),
      heuristic_coef_,
      thrust::raw_pointer_cast(device_backup_neighbors.data()),
      thrust::raw_pointer_cast(device_backup_distances.data()),
      thrust::raw_pointer_cast(device_went_through_heuristic.data())
      );
    CHECK_CUDA(cudaDeviceSynchronize());
    thrust::copy(device_deg.begin(), device_deg.begin() + deg.size(), deg.begin());
    thrust::copy(device_graph.begin(), device_graph.begin() + graph_vec.size(), graph_vec.begin());
    std::vector<float> distances(max_m * size);
    thrust::copy(device_distances.begin(), device_distances.begin() + distances.size(), distances.begin());

    std::vector<int64_t> acc_visited_cnt(block_cnt_);
    thrust::copy(device_acc_visited_cnt.begin(), device_acc_visited_cnt.end(), acc_visited_cnt.begin());
    CHECK_CUDA(cudaDeviceSynchronize());
    int64_t full_visited_cnt = std::accumulate(acc_visited_cnt.begin(), acc_visited_cnt.end(), 0LL);
    DEBUG("full number of visited nodes: {}", full_visited_cnt);

    graph.UnLoadGraphVec(graph_vec, deg, distances, max_m);
  }
}

void CuHNSW::AddPoint(const float* qdata, int level, int label) {  
  #ifdef HALF_PRECISION
    // DEBUG0("fp16")
    std::vector<cuda_scalar> hdata(num_dims_);
    for (int i = 0; i <  num_dims_; ++i) {
      hdata[i] = conversion(qdata[i]);
      // DEBUG("hdata i: {}, scalar: {}", i, out_scalar(hdata[i]));
    }
    device_data_.insert(device_data_.end(), hdata.begin(), hdata.end());
  #else
    // DEBUG0("fp32")
    device_data_.insert(device_data_.end(), qdata, qdata + num_dims_);
  #endif
  data_.insert(data_.end(), qdata, qdata + num_dims_);
  if(level == -1) {
    level = (int)(-std::log(std::uniform_real_distribution(0.0, 1.0)(level_generator)) * level_mult_);
  }
  levels_.resize(num_data_ + 1);
  levels_[num_data_] = level;
  if(level > max_level_) {
    enter_point_ = num_data_;
    max_level_ = level;
    level_graphs_.resize(level + 1, LevelGraph(max_elements_));
  } 
  for(int i = 0; i <= level; i++) {
    level_graphs_[i].AddNode(num_data_);
  }
  if(labelled_) {
    labels_.push_back(label);
  }
  num_data_++;
}
void CuHNSW::AddPoints(const float* qdata, int* levels, int* labels, int num_points) {
  for(int i = 0; i < num_points; i++) {
    AddPoint(qdata + i * num_dims_, levels[i], labels[i]);
  }
}

} // namespace cuhnswplus
