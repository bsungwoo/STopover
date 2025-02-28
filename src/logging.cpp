#include "logging.h"

std::mutex log_mutex;

void log_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ofstream log_file("parallelize_debug.log", std::ios_base::app);
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    log_file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
             << " - " << message << std::endl;
} 