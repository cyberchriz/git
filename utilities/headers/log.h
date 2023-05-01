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
    void error(const std::string& message) {
        log(LOG_LEVEL_ERROR, message);
    }

    void warning(const std::string& message) {
        log(LOG_LEVEL_WARNING, message);
    }

    void info(const std::string& message) {
        log(LOG_LEVEL_INFO, message);
    }

    void debug(const std::string& message) {
        log(LOG_LEVEL_DEBUG, message);
    }

    void set_level(LogLevel level) {
        logLevel_ = level;
    }

    void set_filepath(const std::string& filepath) {
        logFilepath_ = filepath + "log.txt";
    }

    void enable_to_console(bool active=true){
        logToConsole_ = active;
    }

    void enable_to_file(bool active=true){
        logToFile_ = active;
    }    

    template <typename... Args>
    void log(int level, Args&&... args) {
        if (level > LogLevel::LOG_LEVEL_NONE &&
            level <= logLevel_) {
            const char* const levelStrings[] = {
                "ERROR", "WARNING", "INFO", "DEBUG"
            };
            std::stringstream stream;
            (stream << ... << std::forward<Args>(args));
            std::string logMessage = "[" + std::string(levelStrings[level]) + "]: " + stream.str();

            if (logToConsole_) {
                std::cout << logMessage << std::endl;
            }

            if (logToFile_) {
                std::ofstream fileStream(logFilepath_, std::ios_base::app);
                if (fileStream.good()) {
                    fileStream << logMessage << std::endl;
                }
            }
        }
    }

private:
    LogLevel logLevel_ = LOG_LEVEL_NONE;
    bool logToConsole_ = true;
    bool logToFile_ = false;
    std::string logFilepath_ = "log.txt";
};
