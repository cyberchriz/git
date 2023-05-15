// requires C++17 or higher

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

enum LogLevel {
    LOG_LEVEL_NONE,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG
};

class Log {
public:
    static void error(const std::string& message) {
        log(LOG_LEVEL_ERROR, message);
    }

    static void warning(const std::string& message) {
        log(LOG_LEVEL_WARNING, message);
    }

    static void info(const std::string& message) {
        log(LOG_LEVEL_INFO, message);
    }

    static void debug(const std::string& message) {
        log(LOG_LEVEL_DEBUG, message);
    }

    static void set_level(LogLevel level) {
        log_level = level;
    }

    static void set_filepath(const std::string& filepath) {
        log_filepath = filepath;
        if (!log_filepath.empty() && log_filepath.back() != '/') {
            log_filepath += '/';
        }
        log_filepath += "log.txt";
    }

    static void enable_to_console(bool active=true){
        log_to_console = active;
    }

    static void enable_to_file(bool active=true){
        log_to_file = active;
    }    

    template <typename... Args>
    static void log(int level, Args&&... args) {
        if (level > LogLevel::LOG_LEVEL_NONE &&
            level <= log_level) {
            const char* const levelStrings[] = {
                "ERROR", "WARNING", "INFO", "DEBUG"
            };
            std::stringstream stream;
            (stream << ... << std::forward<Args>(args));
            std::string log_message = "[" + std::string(levelStrings[level]) + "]: " + stream.str() + "\n";


            if (log_to_console) {
                std::cout << log_message << std::endl;
            }

            if (log_to_file) {
                std::ofstream file_stream(log_filepath, std::ios_base::app);
                if (file_stream.good()) {
                    file_stream << log_message << std::endl;
                }
            }
        }
    }

private:
    static LogLevel log_level;
    static bool log_to_console;
    static bool log_to_file;
    static std::string log_filepath;
};

LogLevel Log::log_level = LOG_LEVEL_NONE;
bool Log::log_to_console = true;
bool Log::log_to_file = false;
std::string Log::log_filepath = "log.txt";