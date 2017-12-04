#pragma once

#include <string>

namespace dtl
{
    class Topic
    {
    public:
        Topic();
        ~Topic();

        

    private:
        size_t id;
        std::string friendly_name;

    };
}
