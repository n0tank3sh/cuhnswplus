// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once 

#include <BS_thread_pool.hpp>
namespace cuhnswplus{
class ThreadPoolSingleton {
  public:
    static BS::thread_pool& get_instance() {
      static BS::thread_pool instance;
      return instance;
    }
    ThreadPoolSingleton(const ThreadPoolSingleton&) = delete;
    ThreadPoolSingleton& operator=(const ThreadPoolSingleton&) = delete;
  private:
    ThreadPoolSingleton() = delete;
};
}