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
    enum class TaskLocation
    {
        Host, GPU
    };

    enum class TaskStatus
    {
        Init, Running, Waiting, Done
    };

    class Task : public std::enable_shared_from_this<Task>
    {
    public:

        static std::shared_ptr<Task> Create(const std::string& name, std::shared_ptr<Task> parent, TaskLocation location, const std::function<void(std::shared_ptr<Task>)>& f)
        {
            if (parent)
                return parent->AddChildTask(name, location, f);
            else
                return std::shared_ptr<Task>(new Task(name, nullptr, location, f));
        }

        void Run()
        {
            if (this->status != TaskStatus::Init)
                return;

            // run the function //
            std::cout << task_name << ": Running..." << std::endl;
            this->status = TaskStatus::Running;

            this->taskThread = std::thread(&Task::_task_runner_shim, this);
        }

        void Synchronize()
        {
            if (this->status != TaskStatus::Running)
                return;
            
            // block until children complete //
            std::mutex m;
            auto& children_done = this->childrenComplete;
            std::unique_lock<std::mutex> lk(m);
            this->childrenSyncCV.wait(lk, [&children_done] { return children_done; });
        }

        TaskLocation GetLocation() const noexcept
        {
            return this->location;
        }

        TaskStatus GetStatus() const noexcept
        {
            return this->status;
        }

        std::string GetName() const noexcept
        {
            return this->task_name;
        }

    private:
        Task(const std::string& name, std::shared_ptr<Task> parent, TaskLocation location, const std::function<void(std::shared_ptr<Task>)>& function)
        {
            this->task_name = name;
            this->parentTask = parent;
            this->status = TaskStatus::Init;
            this->location = location;
            this->function = function;
            this->childrenComplete = true; // no children - empty //
        }

        std::shared_ptr<Task> AddChildTask(const std::string& name, TaskLocation loc, const std::function<void(std::shared_ptr<Task>)>& f)
        {
            std::shared_ptr<Task> new_task(new Task(name, shared_from_this(), loc, f));
            this->childrenTasks.push_back(new_task);
            this->childrenComplete = false;
            return new_task;
        }

        void CheckIfChildrenComplete()
        {
            childrenCheckMutex.lock();
            auto done = std::all_of(childrenTasks.begin(), childrenTasks.end(), [](const std::shared_ptr<Task> t) { return t->GetStatus() == TaskStatus::Done; });

            if (done)
            {
                childrenComplete = done;
                childrenSyncCV.notify_all();
            }
                
            childrenCheckMutex.unlock();
        }

        void _task_runner_shim()
        {
            // execute function //
            this->function(this->shared_from_this());
            
            // done running - move to the wait state //
            this->status = TaskStatus::Waiting;

            // any children? if so, wait for them //
            if (!childrenComplete)
            {
                std::mutex m;
                auto& children_done = this->childrenComplete;
                std::unique_lock<std::mutex> lk(m);
                this->childrenSyncCV.wait(lk, [&children_done] { return children_done; });
            }

            // our children are done, so we are done too //
            this->status = TaskStatus::Done;

            // notify our parent, if it exists //
            if (this->parentTask)
                this->parentTask->CheckIfChildrenComplete();
        }

        // concurrency related //
        bool childrenComplete;
        std::mutex childrenCheckMutex;
        std::condition_variable childrenSyncCV;
        std::thread taskThread;

        // members //
        std::string task_name;
        TaskStatus status;
        TaskLocation location;
        std::shared_ptr<Task> parentTask;
        std::vector<std::shared_ptr<Task>> childrenTasks;
        std::function<void(std::shared_ptr<Task>)> function;
    };
}