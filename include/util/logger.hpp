#ifndef SGPROCESSINGMANAGER_LOGGER_HPP
#define SGPROCESSINGMANAGER_LOGGER_HPP

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#if defined( ANDROID )
#include <spdlog/sinks/android_sink.h>
#endif

namespace sgns::sgprocmanager
{
    using Logger = std::shared_ptr<spdlog::logger>;

    /**
   * Provide logger object
   * @param tag - tagging name for identifying logger
   * @return logger object
   */
    Logger createLogger( const std::string &tag, const std::string &basepath = "" );
} // namespace sgns::sgprocmanager

#endif // SGPROCESSINGMANAGER_LOGGER_HPP
