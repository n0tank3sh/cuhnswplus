#pragma once 

#include <BS_thread_pool.hpp>
namespace cuhnsw{
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