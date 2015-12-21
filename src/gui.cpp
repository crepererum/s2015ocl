#include <memory>

#include <nanogui/screen.h>

#include "common.hpp"
#include "gui.hpp"


class MyScreen : public nanogui::Screen {
    public:
        MyScreen(const std::shared_ptr<spdlog::logger>& logger) : Screen(Eigen::Vector2i(800, 600), "s2015ocl"), log(logger) {}

        virtual void drawAll() override {
            log->debug() << "draw screen";
            Screen::drawAll();
        }

    private:
        std::shared_ptr<spdlog::logger> log;
};

void main_gui() {
    auto log_gui = spdlog::stdout_logger_mt("gui");
    log_gui->info() << "hello world";

    log_gui->info() << "set up";
    nanogui::init();

    // inner part, where nanogui (and glfw) are initialized
    {
        MyScreen screen(log_gui);
        screen.drawAll();
        screen.setVisible(true);

        log_gui->info() << "start main loop";
        nanogui::mainloop();
    }

    log_gui->info() << "shutdown";
    nanogui::shutdown();

    log_gui->info() << "goodbye!";
}
