#include <memory>

#include <GLFW/glfw3.h>

#include <nanovg.h>
#include <nanogui/nanogui.h>

#include "common.hpp"
#include "gui.hpp"


class MyScreen : public nanogui::Screen {
    public:
        MyScreen(
            const std::shared_ptr<spdlog::logger>& logger,
            std::size_t n,
            std::size_t m,
            const shared_mutex_t& mGlobal,
            const shared_mem_t<unsigned char>& hTexture,
            const shared_atomic_t<bool>& shutdown
        ) : Screen(Eigen::Vector2i(800, 600), "s2015ocl"),
            log(logger),
            n(n),
            m(m),
            mGlobal(mGlobal),
            hTexture(hTexture),
            shutdown(shutdown){
            mainwindow = new nanogui::Window(this, "s2015ocl");
            mainwindow->setPosition(Eigen::Vector2i(100, 100));
            mainwindow->setLayout(new nanogui::GroupLayout());

            new nanogui::Label(mainwindow, "foo bar", "sans-bold");

            visualization = new nanogui::ImageView(mainwindow);
            visualization->setPolicy(nanogui::ImageView::SizePolicy::Expand);
            visualization->setFixedSize(Eigen::Vector2i(300, 300));

            performLayout();
        }

        virtual void drawAll() override {
            log->debug() << "draw screen";

            // update visualization
            if (visualization->image() != 0) {
                nvgDeleteImage(this->nvgContext(), visualization->image());
            }
            {
                std::lock_guard<std::mutex> guard(*mGlobal);
                visualization->setImage(nvgCreateImageRGBA(
                    this->nvgContext(),
                    static_cast<int>(n),
                    static_cast<int>(n),
                    0,
                    hTexture->data()
                ));
                myassert(visualization->image() != 0, "image data should be loaded by nanovg");
            }

            Screen::drawAll();

            if (*shutdown) {
                glfwSetWindowShouldClose(this->glfwWindow(), 1);
            }
        }

    private:
        std::shared_ptr<spdlog::logger> log;
        std::size_t n;
        std::size_t m;
        shared_mutex_t mGlobal;
        shared_mem_t<unsigned char> hTexture;
        shared_atomic_t<bool> shutdown;

        // widgets, ref-counted by nanogui
        nanogui::Window* mainwindow;
        nanogui::ImageView* visualization;
};

void main_gui(std::size_t n, std::size_t m, shared_mutex_t mGlobal, shared_mem_t<unsigned char> hTexture, shared_atomic_t<bool> shutdown) {
    auto log_gui = spdlog::stdout_logger_mt("gui");
    log_gui->info() << "hello world";

    log_gui->info() << "set up";
    nanogui::init();

    // inner part, where nanogui (and glfw) are initialized
    try {
        MyScreen screen(log_gui, n, m, mGlobal, hTexture, shutdown);
        screen.drawAll();
        screen.setVisible(true);

        log_gui->info() << "start main loop";
        nanogui::mainloop();
    } catch (const std::exception& e) {
        log_gui->error() << e.what();
    } catch (...) {
        log_gui->error() << "unkown error";
    }

    log_gui->info() << "shutdown";
    nanogui::shutdown();

    *shutdown = true;
    log_gui->info() << "goodbye!";
}
