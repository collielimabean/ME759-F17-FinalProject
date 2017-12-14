#include <iostream>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include "MpiContext.hpp"
#include "TaskManager.hpp"


void child_fn(MPI_Comm parent, void *data, size_t len)
{
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "child_fn called!" << std::endl;
}

int main(int argc, char **argv)
{
    auto& manager = dtl::TaskManager::GetInstance("master", argc, argv);

    if (!manager)
    {
        std::cerr << "Failed to initialize!" << std::endl;
        return 1;
    }

    // init map //
    manager.RegisterFunction("child_fn", child_fn);

    if (!manager.IsMaster())
    {
        // will block if child //
        std::cout << "Child!" << std::endl;
        manager.SetName("_child_default_");
        manager.RunChildRoutine();
        return 0;
    }

    std::cout << "Master!" << std::endl;

    // spin off child process //
    bool ok = manager.SpawnChildNode("New Node");
    std::cout << "Spawn: " << ok << std::endl;

    // look for a child process that can execute //
    ok = manager.IssueJob(
        "", // look for any available node
        "child_fn",
        nullptr,
        0
    );
    
    std::cout << "Issue: " << ok << std::endl;

    if (!ok)
    {
        std::cout << "Failed!" << std::endl;
        return 1;
    }

    // synchronize //
    manager.SynchronizeOnChildren();

    // terminate child //
    manager.TerminateChildren();

    return 0;
}
