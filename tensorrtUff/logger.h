#pragma once
#ifndef LOGGER_H
#define LOGGER_H

#include "logging.h"

extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

constexpr long long int operator"" _GiB(long long unsigned int val)
{
	return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
	return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
	return val * (1 << 10);
}

#endif // LOGGER_H
