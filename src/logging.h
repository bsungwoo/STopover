#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>

extern std::mutex log_mutex;

void log_message(const std::string& message);

#endif // LOGGING_H 