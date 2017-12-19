#pragma once

#include <cstring>
#include <string>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif

    #include <windows.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <unistd.h>
#endif

namespace dtl
{
    std::string GetComputerName()
    {
        std::string name;
        char buffer[1024];

    #ifdef _WIN32
        WSADATA wsadata;
        if (WSAStartup(MAKEWORD(2, 2), &wsadata) != 0)
            return "";
    #endif

        memset(buffer, 0, sizeof(buffer));
        if (gethostname(buffer, 1024) < 0)
            name = "";
        
        name = std::string(buffer);

    #ifdef _WIN32
        WSACleanup();
    #endif

        return name;
    }
}
