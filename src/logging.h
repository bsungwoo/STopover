#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>

// Define the mutex as inline to avoid multiple definitions
inline std::mutex& get_log_mutex() {
    static std::mutex log_mutex;
    return log_mutex;
}

// Define the log_message function as inline to avoid multiple definitions
inline void log_message(const std::string& message) {
    std::lock_guard<std::mutex> lock(get_log_mutex());
    std::ofstream log_file("parallelize_debug.log", std::ios_base::app);
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    log_file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
             << " - " << message << std::endl;
}

#endif // LOGGING_H 