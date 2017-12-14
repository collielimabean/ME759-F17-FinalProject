#pragma once

#include <mpi.h>
#include <condition_variable>
#include <thread>
#include <string>
#include <memory>
#include <google/protobuf/message.h>
#include "MpiContext.hpp"

namespace dtl
{
    using TaskFunction = std::function<void(void *, size_t)>;
    using TaskFunctionMap = std::map<std::string, TaskFunction>;
    using CustomPacketHandler = std::function<void(void *, size_t)>;

    enum ChildStatus
    {
        Idle, Running, Terminated
    };

    class TaskManager
    {
    public:
        static TaskManager& GetInstance(const std::string& name, int argc, char **argv);

        void SetName(const std::string& new_name);
        explicit operator bool() const;
        bool HasGPU();
        bool IsMaster() const;
        void SetFunctionMap(const TaskFunctionMap& map);
        void RegisterFunction(const std::string& name, const TaskFunction& fn);
        void SetPacketCallback(CustomPacketHandler handler);

        bool SpawnChildNode(
            const std::string& name,
            const std::string& fn_name,
            void *data,
            size_t len,
            bool has_parameters = false,
            bool needs_gpu = false);

        void RunChildRoutine();
        void SynchronizeOnChildren();
        void TerminateChildren();

    private:
        TaskManager(const std::string& name, int argc, char **argv);
        ~TaskManager();

        void ListenOnChildren();

        struct ChildNode
        {
            std::string name;
            ChildStatus status;
            MPI_Comm comm;
        };

        // common data //
        std::string name;
        bool is_master;
        MPI_Comm parentComm;
        std::unique_ptr<bool> hasGpu;
        TaskFunctionMap fnMap;
        mpi::MpiContext context;
        bool is_valid_instance;
        std::string program_name;

        // child listener 
        std::thread childListenerThread;
        bool childListenerRunning;

        // master specific data //
        std::vector<ChildNode> children;
        bool childrenIdle;
        std::condition_variable childrenSyncCV;

        // child node specific data //
        std::thread childThread;
        bool childThreadRunning;
        CustomPacketHandler userPacketHandler;
    };
}
