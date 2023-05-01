#include <iostream>
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
        logLevel_ = level;
    }

private:
    static void log(int level, const std::string& message) {
        if (level >LogLevel::LOG_LEVEL_NONE &&
            level <= logLevel_) {
            static const char* const levelStrings[] = {
                "ERROR", "WARNING", "INFO", "DEBUG"
            };
            std::cout << "[" << levelStrings[level] << "]: " << message << std::endl;
        }
    }

    static LogLevel logLevel_;
};

/* usage example
int main() {
    Log::set_level(LOG_LEVEL_DEBUG);
    Log::error("An error occurred.");
    Log::warning("This is a warning message.");
    Log::info("This is an informational message.");
    Log::debug("Debugging information here.");

    Log::set_level(LOG_LEVEL_WARNING);
    Log::error("This should still be logged.");
    Log::warning("This should be logged.");
    Log::info("This should not be logged.");
    Log::debug("This should not be logged.");

    return 0;
}
*/
