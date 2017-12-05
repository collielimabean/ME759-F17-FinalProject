#pragma once

#include <thread>
#include <functional>
#include <memory>
#include <vector>

namespace dtl
{
    enum class TaskLocation
    {
        Host, GPU
    };

    enum class TaskStatus
    {
        Init, Running, Waiting, Done
    };

    class Task
    {
    public:
        static std::shared_ptr<Task> Run(
            std::shared_ptr<Task> parent, 
            const std::function<void(std::shared_ptr<Task>)>& func, 
            TaskLocation loc
        )
        {
            std::shared_ptr<Task> task(new Task());
            task->task_func = func;
            task->location = loc;



            return task;
        }

        static std::shared_ptr<Task> RunAsync(
            std::shared_ptr<Task> parent,
            const std::function<void(std::shared_ptr<Task>)>& func,
            TaskLocation loc
        )
        {
            // TODO
            return std::shared_ptr<Task>(new Task());
        }
        
        static std::shared_ptr<Task> DistributedFor(
            std::shared_ptr<Task> parent,
            const std::function<void(std::shared_ptr<Task>, int)>& func, 
            TaskLocation loc,
            int start,
            int end,
            int step
        )
        {
            // XXX: create a new Task that has the below function
            // and return the handle to it

            for (int i = start; i < end; i += step)
            {
                auto _f = std::bind(func, std::placeholders::_1, i);
                Run(nullptr, _f, loc);
            }

            return std::shared_ptr<Task>(new Task());
        }
        

    private:
        Task() : status(TaskStatus::Init)
        {

        }


        std::function<void(std::shared_ptr<Task>)> task_func;
        std::vector<std::shared_ptr<Task>> child_tasks;
        TaskLocation location;
        TaskStatus status;
    };

    using SharedTask = std::shared_ptr<Task>;
}
