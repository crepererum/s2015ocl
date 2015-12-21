#pragma once

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

using shared_mutex_t = std::shared_ptr<std::mutex>;

template <typename T>
using shared_mem_t = std::shared_ptr<std::vector<T>>;

template <typename T>
using shared_atomic_t = std::shared_ptr<std::atomic<T>>;

class MyException : public std::exception {
    public:
        MyException(const std::string& msg) : msg(msg) {}
        virtual const char* what() const throw() override {
            return msg.c_str();
        }

    private:
        std::string msg;
};

inline void myassert(bool test, const std::string& msg) {
    if (!test) {
        throw MyException(msg);
    }
}
