#pragma once

namespace dtl
{
    enum class TaskLocation
    {
        Host, GPU
    };

    class Task
    {
    public:
        Task();
        ~Task();
    private:
        TaskLocation location;
    };
}
