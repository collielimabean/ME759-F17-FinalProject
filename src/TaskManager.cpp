#include "TaskManager.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "commands.pb.h"
#include "Packets.hpp"
#include "util.hpp"
#include <mutex>

using namespace dtl;
using namespace dtl::packets;

/**
 * TODO: Wrapper functions over MPI_Probe + MPI_Recv 
 * TODO: Ensure no arbitrary hardcoded buffers 
 */


TaskManager& TaskManager::GetInstance(const std::string& name, int argc, char **argv)
{
    static TaskManager instance(name, argc, argv);
    return instance;
} 

TaskManager::TaskManager(const std::string& name, int argc, char **argv)
{
    context.Initialize(&argc, &argv);
    is_valid_instance = static_cast<bool>(context);

    this->childrenIdle = true;
    this->childListenerRunning = false;
    this->name = name;
    MPI_Comm_get_parent(&parentComm);
    is_master = parentComm == MPI_COMM_NULL;
    program_name = std::string(argv[0]);
}

TaskManager::~TaskManager()
{
    childListenerRunning = false;
    if (childListenerThread.joinable())
        childListenerThread.join();

    if (parentComm != MPI_COMM_NULL 
        && parentComm != MPI_COMM_WORLD 
        && parentComm != MPI_COMM_SELF)
    {
        MPI_Comm_disconnect(&parentComm);
    }
}

void TaskManager::SetName(const std::string& new_name)
{
    this->name = new_name;
}

TaskManager::operator bool() const
{
    return is_valid_instance;
}

bool TaskManager::HasGPU()
{
    if (!hasGpu)
    {
        hasGpu = std::unique_ptr<bool>(new bool);
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        *(hasGpu) = num_devices > 0;
    }

    return *hasGpu;
}

bool TaskManager::IsMaster() const
{
    return is_master;
}

void TaskManager::SetFunctionMap(const TaskFunctionMap& map)
{
    fnMap = map;
}

void TaskManager::RegisterFunction(const std::string& name, const TaskFunction& fn)
{
    fnMap[name] = fn;
}

void TaskManager::SetPacketCallback(CustomPacketHandler handler)
{
    this->userPacketHandler = handler;
}

bool TaskManager::IssueJob(
    const std::string& node_name,
    const std::string& fn_name,
    void *data,
    size_t len,
    bool has_parameters,
    bool needs_gpu
)
{
    if (!is_master)
        return false;

    if (!has_parameters && (data || len > 0))
        return false;

    // search current list for anyone who can fulfill //
    for (auto& child : children)
    {
        // send function //
        SerializedPacket itrPkt;
        auto itr = GetIssueTaskRequest(node_name, fn_name, needs_gpu, has_parameters);
        if (!SerializePacket(itr, itrPkt))
        {
            std::cout << "[master] Failed to make ITR packet." << std::endl;
            return false;
        }

        MPI_Ssend(itrPkt.data, itrPkt.size, MPI_CHAR, 0, 0, child.comm);

        // can this node do it? //
        // since we have a dedicated thread listening on our children,
        // we need to use a condition variable and go to sleep

        // set up condition variable
        issueJobDone = false;

        std::mutex m;
        auto& _local_issueJobDone = this->issueJobDone;
        std::unique_lock<std::mutex> lk(m);
        issueJobCV.wait(lk, [&_local_issueJobDone] { return _local_issueJobDone; });

        std::cout << "wait complete!" << std::endl;

        if (!childCanDoTask)
            continue;

        // send data //
        if (data)
            MPI_Ssend(data, len, MPI_CHAR, 0, 0, child.comm);

        break;
    }

    return true;
}


bool TaskManager::SpawnChildNode(const std::string& name)
{
    if (!is_master)
        return false;

    if (!childListenerRunning)
    {
        childListenerRunning = true;
        childrenIdle = false;
        childListenerThread = std::thread(&TaskManager::ListenOnChildren, this);
    }

    // spawn new node //
    MPI_Comm child;

    MPI_Comm_spawn(program_name.c_str(), MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &child, MPI_ERRCODES_IGNORE);

    // add child to internal tracker //
    ChildNode new_node;
    new_node.name = name;
    new_node.status = ChildStatus::Idle;
    new_node.comm = child;
    children.push_back(new_node);

    // set name //
    SerializedPacket csiPkt;
    auto csi = GetChildSetInfoPacket(name);
    if (!SerializePacket(csi, csiPkt))
    {
        std::cout << "[master] Failed to make CSI packet." << std::endl;
        return false;
    }

    MPI_Ssend(csiPkt.data, csiPkt.size, MPI_CHAR, 0, 0, child);
    return true;
}

void TaskManager::RunChildRoutine()
{
    char buffer[1024];

    if (is_master)
        return;

    // set name to default //
    this->name = GetComputerName();

    childThreadRunning = true;

    while (childThreadRunning)
    {
        MPI_Status mpi_status;
        int sz;

        memset(buffer, 0, sizeof(buffer));        

        MPI_Recv(
            buffer, 
            1024, 
            MPI_CHAR, 
            MPI_ANY_SOURCE, 
            MPI_ANY_TAG, 
            parentComm, 
            &mpi_status
        );

        MPI_Get_count(&mpi_status, MPI_CHAR, &sz);

        // initialize all packet types //
        ChildSetInfo csi;
        TerminateAllChildrenRequest tacr;
        IssueTaskRequest itr;

        if (ParseChildSetInfoPacket(buffer, sz, csi))
        {
            std::cout << "[" << name << "] CSI msg received w/name: " << csi.name() << std::endl;
            this->name = csi.name();
        }
        else if (ParseIssueTaskRequestPacket(buffer, sz, itr))
        {
            std::cout << "[" << name << "] ITR name: "
                << itr.name() << " "
                << itr.function() << " "
                << itr.needsgpu() << " "
                << itr.hasparameters() << std::endl;

            // can exec if: name match or name is empty
            // function name must exist
            // if task needs gpu, make sure we have a gpu
            bool can_execute = 
                ((name.compare(itr.name()) == 0 ) || itr.name().length() == 0)
                && (fnMap.find(itr.function()) != fnMap.end())
                && !(itr.needsgpu() && !HasGPU());

            // notify parent //
            SerializedPacket pkt;
            auto response = GetIssueTaskResponse(name, can_execute);
            if (!SerializePacket(response, pkt))
            {
                std::cout << "[" << name << "] failed to serialize ITR response." << std::endl;
                continue;
            }
            MPI_Send(pkt.data, pkt.size, MPI_CHAR, 0, 0, parentComm);

            if (!can_execute)
            {
                std::cout << "[" << name << "] ignoring incoming request." << std::endl;
                continue;
            }

            char *buffer = nullptr;
            int len = 0;

            if (itr.hasparameters())
            {
                MPI_Status _mpi_status;

                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, parentComm, &_mpi_status);
                MPI_Get_count(&_mpi_status, MPI_CHAR, &len);

                buffer = new char[len];
                MPI_Recv(
                    buffer,
                    len,
                    MPI_CHAR,
                    MPI_ANY_SOURCE,
                    MPI_ANY_TAG,
                    parentComm,
                    MPI_STATUS_IGNORE
                );
            }

            // run the thing //
            fnMap[itr.function()](parentComm, buffer, len);

            if (buffer)
                delete[] buffer;
        }
        else if (ParseTerminateAllChildrenRequestPacket(buffer, sz, tacr))
        {
            // kill currently running task //
            std::cout << "[" << name << "] stopping..." << std::endl;

            SerializedPacket pkt;
            auto response = GetTerminateAllChildrenResponsePacket(name);
            if (!SerializePacket(response, pkt))
                std::cout << "[" << name << "] Failed to create terminate response!" << std::endl;

            MPI_Send(pkt.data, pkt.size, MPI_CHAR, 0, 0, parentComm);
            childThreadRunning = false;
            continue;
        }
        else
        {
            std::cout << "[" << name << "] mystery received" << std::endl;

            // call user packet handling routine //
            if (userPacketHandler)
                userPacketHandler(parentComm, buffer, sz);
        }
    }
}

void TaskManager::SynchronizeOnChildren()
{
    if (children.size() == 0 || !is_master)
        return;

    if (childrenIdle)
        return;

    // block until children idle //
    std::mutex m;
    auto& child_nodes_idle = this->childrenIdle;
    std::unique_lock<std::mutex> lk(m);
    this->childrenSyncCV.wait(lk, [&child_nodes_idle] { return child_nodes_idle; });
    std::cout << "sync complete." << std::endl;
}

void TaskManager::ListenOnChildren()
{
    if (!is_master)
        return;
        
    while (childListenerRunning)
    {
        for (auto& child : children)
        {
            MPI_Status _mpi_status;
            char *buffer;
            int iprobe_test;
            int sz;

            // any message from this child? 
            MPI_Iprobe(
                MPI_ANY_SOURCE, 
                MPI_ANY_TAG,
                child.comm,
                &iprobe_test,
                &_mpi_status
            );

            if (!iprobe_test)
                continue;

            MPI_Get_count(&_mpi_status, MPI_CHAR, &sz);
            buffer = new char[sz];

            // message waiting, grab it
            MPI_Recv(
                buffer,
                sz,
                MPI_CHAR,
                MPI_ANY_SOURCE,
                MPI_ANY_TAG,
                child.comm,
                MPI_STATUS_IGNORE
            );

            ChildComplete ccPkt;
            IssueTaskResponse itr_response;
            TerminateAllChildrenResponse tacr;

            if (ParseChildCompletePacket(buffer, sz, ccPkt))
            {
                child.status = ChildStatus::Idle;
            }
            else if (ParseIssueTaskResponsePacket(buffer, sz, itr_response))
            {
                if (itr_response.accepted())
                {
                    std::cout << "[master] " << itr_response.name() << " has accepted a task." << std::endl;
                    child.status = ChildStatus::Running;
                }
                else
                {
                    std::cout << "[master] " << itr_response.name() << " has rejected a task." << std::endl;
                }

                childCanDoTask = itr_response.accepted();
                issueJobDone = true;
                issueJobCV.notify_all();
                std::cout << "[master] listener notifying..." << std::endl;
            }
            else if (ParseTerminateAllChildrenResponsePacket(buffer, sz, tacr))
            {
                std::cout << "[Master] " << child.name << " marked as terminated." << std::endl;
                child.status = ChildStatus::Terminated;
                continueWithTermination = true;
                terminationCV.notify_all();
            }
            else
            {
                if (this->userPacketHandler)
                    this->userPacketHandler(parentComm, buffer, sz);
            }

            delete[] buffer;
        }

        bool all_children_idle = std::all_of(
            children.begin(),
            children.end(),
            [](const ChildNode& s) { return s.status == ChildStatus::Idle; }
        );

        if (all_children_idle)
        {
            childrenIdle = all_children_idle;
            childrenSyncCV.notify_all();
        }

        // spin loops are inefficient
        // the correct implementation is to create a global
        // intercommunicator between the master and all of its
        // child nodes, but that's actually somewhat annoying
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void TaskManager::TerminateChildren()
{
    SynchronizeOnChildren();
    std::cout << "terminating..." << std::endl;

    for (auto& child : children)
    {
        // reset termination flag for condition variable
        continueWithTermination = false;

        if (child.status == ChildStatus::Terminated)
            continue;

        SerializedPacket pkt;
        auto tacr = GetTerminateAllChildrenRequestPacket();

        if (!SerializePacket(tacr, pkt))
        {
            std::cout << "[master] failed to init terminate all children req" << std::endl;
            return;
        }

        MPI_Send(pkt.data, pkt.size, MPI_CHAR, 0, 0, child.comm);

        // go to sleep again - wait for response from the listener //
        std::mutex m;
        auto& _local_cont_with_term = this->continueWithTermination;
        std::unique_lock<std::mutex> lk(m);
        this->terminationCV.wait(lk, [&_local_cont_with_term] { return _local_cont_with_term; });
    }

    childListenerRunning = false;
    std::cout << "termination done." << std::endl;
}
