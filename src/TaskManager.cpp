#include "TaskManager.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include "commands.pb.h"
#include "Packets.hpp"
#include <mutex>

using namespace dtl;
using namespace dtl::packets;


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

bool TaskManager::SpawnChildNode(
    const std::string& name,
    const std::string& fn_name,
    void *data,
    size_t len,
    bool has_parameters,
    bool needs_gpu)
{
    if (!is_master)
        return false;

    if (!has_parameters && (data || len > 0))
        return false;

    if (!childListenerRunning)
    {
        childListenerRunning = true;
        childrenIdle = false;
        childListenerThread = std::thread(&TaskManager::ListenOnChildren, this);
    }

    // XXX: search current list for anyone who can fulfill //

    // spawn new node //
    MPI_Comm child;
    int err[1];

    MPI_Comm_spawn(program_name.c_str(), MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_SELF, &child, err);

    // add child to internal tracker //
    ChildNode new_node;
    new_node.name = name;
    new_node.status = ChildStatus::Running; // XXX: HACK
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

    // send function //
    SerializedPacket itrPkt;
    auto itr = GetIssueTaskRequest(name, fn_name, needs_gpu, has_parameters);
    if (!SerializePacket(itr, itrPkt))
    {
        std::cout << "[master] Failed to make ITR packet." << std::endl;
        return false;
    }

    MPI_Ssend(itrPkt.data, itrPkt.size, MPI_CHAR, 0, 0, child);

    // send data //
    if (data)
        MPI_Ssend(data, len, MPI_CHAR, 0, 0, child);
    return true;
}

void TaskManager::RunChildRoutine()
{
    char buffer[1024];

    if (is_master)
        return;

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
        packets::ChildSetInfo csi;
        packets::TerminateAllChildrenRequest tacr;
        packets::IssueTaskRequest itr;

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

            if ((name.compare(itr.name()) != 0 && name.length() != 0) 
                || fnMap.find(itr.function()) == fnMap.end()
                || itr.needsgpu() && HasGPU())
            {
                std::cout << "[" << name << "] ignoring incoming request." << std::endl;
            }
            else
            {
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
                fnMap[itr.function()](buffer, len);

                if (buffer)
                    delete[] buffer;
            }
        }
        else if (ParseTerminateAllChildrenRequestPacket(buffer, sz, tacr))
        {
            // kill currently running task //
            std::cout << "[" << name << "] stopping..." << std::endl;

            packets::TerminateAllChildrenResponse tac_response;
            tac_response.set_opcode(TerminateAllChildrenResponse_Opcode);
            tac_response.set_name(name);
            char *tac_response_buffer = new char[tac_response.ByteSize()];
            tac_response.SerializeToArray(tac_response_buffer, tac_response.ByteSize());
            MPI_Send(tac_response_buffer, tac_response.ByteSize(), MPI_CHAR, 0, 0, parentComm);
            delete[] tac_response_buffer;
            
            childThreadRunning = false;
            continue;
        }
        else
        {
            std::cout << "[" << name << "] mystery received" << std::endl;
            // call user packet handling routine //
            if (userPacketHandler)
                userPacketHandler(buffer, sz);
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

            packets::ChildComplete ccPkt;
            packets::TerminateAllChildrenResponse tacr;

            if (ParseChildCompletePacket(buffer, sz, ccPkt))
            {
                child.status = ChildStatus::Idle;
            }
            else if (ParseTerminateAllChildrenResponsePacket(buffer, sz, tacr))
            {
                std::cout << "[Master] " << child.name << " marked as terminated." << std::endl;
                child.status = ChildStatus::Terminated;
            }
            else
            {
                if (this->userPacketHandler)
                    this->userPacketHandler(buffer, sz);
            }
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

    while (true)
    {
        for (auto& child : children)
        {
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
        }

        bool all_children_terminated = std::all_of(
            children.begin(),
            children.end(),
            [](const ChildNode& s) { return s.status == ChildStatus::Terminated; }
        );

        if (all_children_terminated)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));    
    }

    childListenerRunning = false;
    std::cout << "termination done." << std::endl;
}
