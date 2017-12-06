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
    std::string gen_random(size_t len) 
    {
        static const char alphanum[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";

        std::string s;
        for (size_t i = 0; i < len; ++i)
            s += alphanum[rand() % (sizeof(alphanum) - 1)];
        
        return s;
    }

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
        static std::shared_ptr<Task> Create(std::shared_ptr<Task> parent, TaskLocation location, const std::function<void(std::shared_ptr<Task>)>& f)
        {
            if (parent)
                return parent->AddChildTask(location, f);
            else
                return std::shared_ptr<Task>(new Task(nullptr, location, f));
        }

        void Run()
        {
            if (this->status != TaskStatus::Init)
                return;

            // run the function //
            std::cout << task_name << ": Running..." << std::endl;
            this->status = TaskStatus::Running;

            auto _f = std::bind(this->function, this->shared_from_this());
            this->taskThread = std::thread(_f, this);

            // join on spun thread //
            if (this->taskThread.joinable())
                this->taskThread.join();

            // check if children complete //
            if (!childrenComplete)
            {
                std::cout << this->task_name << ": Waiting for children...." << std::endl;
                std::mutex m;
                auto& children_done = this->childrenComplete;
                std::unique_lock<std::mutex> lk(m);
                this->childrenSyncCV.wait(lk, [&children_done] { return children_done; });
            }

            // on completion //
            this->status = TaskStatus::Done;

            if (this->parentTask)
                this->parentTask->CheckIfChildrenComplete();

            std::cout << this->task_name << ": DONE!" << std::endl;
        }

        std::shared_ptr<Task> AddChildTask(TaskLocation loc, const std::function<void(std::shared_ptr<Task>)>& f)
        {
            std::shared_ptr<Task> new_task(new Task(shared_from_this(), loc, f));
            this->childrenTasks.push_back(new_task);
            return new_task;
        }

        void WaitForChildren()
        {

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

        Task(std::shared_ptr<Task> parent, TaskLocation location, const std::function<void(std::shared_ptr<Task>)>& function)
        {
            this->task_name = gen_random(10);

            this->parentTask = parent;
            this->status = TaskStatus::Init;
            this->location = location;
            this->function = function;
            this->childrenComplete = true; // no children - empty //
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