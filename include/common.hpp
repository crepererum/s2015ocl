#pragma once

#include <exception>
#include <string>

#include <spdlog/spdlog.h>

class MyException : public std::exception {
    public:
        MyException(const std::string& msg) : msg(msg) {}
        virtual const char* what() const throw() override {
            return msg.c_str();
        }

    private:
        std::string msg;
};
