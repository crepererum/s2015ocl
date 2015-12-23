#include <chrono>
#include <thread>

#include <pulse/error.h>
#include <pulse/simple.h>

#include "common.hpp"

class PulseWrapper {
    public:
        PulseWrapper(std::size_t samplerate) {
            pa_sample_spec ss = {
                .format = PA_SAMPLE_FLOAT32LE,
                .rate = static_cast<std::uint32_t>(samplerate),
                .channels = 1
            };
            int err;

            s = pa_simple_new(
                nullptr,
                "s2015ocl",
                PA_STREAM_PLAYBACK,
                nullptr,
                "playback",
                &ss,
                nullptr,
                nullptr,
                &err
            );

            if (!s) {
                throw_exception(err, "unknown error (pa_simple_new)");
            }
        }

        ~PulseWrapper() {
            pa_simple_free(s);
        }

        void write(const std::vector<float>& data) {
            int err;
            if (!data.empty()) {
                if (pa_simple_write(s, data.data(), data.size(), &err) < 0) {
                    throw_exception(err, "unknown error (pa_simple_write)");
                }
                something_played = true;
            }
        }

        std::size_t latency() {
            if (something_played) {
                int err;
                std::size_t l = pa_simple_get_latency(s, &err);
                if (err != PA_OK) {
                    // TODO: why is the pulse error handling so weird?!
                    //throw_exception(err, "unknown error (pa_simple_get_latency)");
                    return 0;
                }
                return l;
            } else {
                return 0;
            }
        }

    private:
        pa_simple* s;
        bool something_played = false;

        void throw_exception(int err, const char* fallback_what) {
            auto estring = pa_strerror(err);
            if (estring) {
                throw MyException(estring);
            } else {
                throw MyException(fallback_what);
            }
        }
};

void main_audio(std::size_t samplerate, shared_mutex_t mGlobal, shared_atomic_t<bool> shutdown, shared_buffer_t<float> audiobuffer) {
    auto log = spdlog::stdout_logger_mt("audio");
    log->info() << "hello world";

    try {
        log->info() << "connect to pulseaudio";
        PulseWrapper pulse(samplerate);
        bool warned_about_underflow = true; // start with 'warned' state because of startup latency

        while (!(*shutdown)) {
            if (pulse.latency() < 1000000) {
                std::vector<float> chunk;
                {
                    std::lock_guard<std::mutex> guard(*mGlobal);
                    if (!audiobuffer->empty()) {
                        chunk = audiobuffer->front();
                        audiobuffer->pop();
                    }
                }

                if (!chunk.empty()) {
                    warned_about_underflow = false;
                    log->debug() << "write to pulse";
                    pulse.write(chunk);
                } else {
                    if (!warned_about_underflow) {
                        log->warn() << "audio queue does not contain content";
                        warned_about_underflow = true;
                    }
                }
            } else {
                log->debug() << "pulse latency high -> sleep";
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        log->info() << "disconnect";
    } catch (const std::exception& e) {
        log->error() << e.what();
    } catch (...) {
        log->error() << "unkown error";
    }

    *shutdown = true;
    log->info() << "goodbye";
}
