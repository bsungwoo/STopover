#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <mutex>

// Declare the mutex
extern std::mutex log_mutex;

// Declare the log_message function
void log_message(const std::string& message);

#endif // LOGGING_H 