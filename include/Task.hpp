#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dtl
{
    enum class TaskStatus
    {
        Init, Running, Waiting, Done
    };

    class Task : public std::enable_shared_from_this<Task>
    {
    public:

        static std::shared_ptr<Task> Create(
            const std::string& name,
            std::shared_ptr<Task> parent,
            const std::function<void(std::shared_ptr<Task>)>& f
        );

        ~Task();
        void Run();
        void Wait();
        void WaitForChildren();
        TaskStatus GetStatus() const noexcept;
        std::string GetName() const noexcept;

    private:
        Task(
            const std::string& name,
            std::shared_ptr<Task> parent,
            const std::function<void(std::shared_ptr<Task>)>& function
        );


        std::shared_ptr<Task> AddChildTask(
            const std::string& name,
            const std::function<void(std::shared_ptr<Task>)>& f
        );

        void CheckIfChildrenComplete();
        void _task_runner_shim();

        // concurrency related //
        bool childrenComplete;
        std::mutex childrenCheckMutex;
        std::condition_variable childrenSyncCV;
        std::thread taskThread;

        // members //
        std::string task_name;
        TaskStatus status;
        std::shared_ptr<Task> parentTask;
        std::vector<std::shared_ptr<Task>> childrenTasks;
        std::function<void(std::shared_ptr<Task>)> function;
    };
}