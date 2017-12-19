#include <iostream>
#include <string>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include "MpiContext.hpp"
#include "TaskManager.hpp"


__global__ void device_increment_array(int *data, size_t length)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length)
        return;
    data[i] += 1;
}

void child_fn(MPI_Comm parent, void *data, size_t len)
{
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "child_fn called!" << std::endl;

    int *devArray;
    int *hostAfter = new int[len];

    cudaMalloc((void **) &devArray, len);
    cudaMemcpy(devArray, data, len, cudaMemcpyHostToDevice);
    device_increment_array<<<1, 1024>>>(devArray, len);
    cudaMemcpy(hostAfter, devArray, len, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < 1024; i++)
        if (((int *) data)[i] + 1 != hostAfter[i])
            errors += 1;

    std::cout << "[child_fn] errors: " << errors << std::endl;

    cudaFree(devArray);
    delete[] hostAfter;
}

void test_fn(MPI_Comm parent, void *data, size_t len)
{
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "[test_fn] called!" << std::endl;
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
    manager.RegisterFunction("test_fn", test_fn);

    if (!manager.IsMaster())
    {
        // will block if child //
        manager.RunChildRoutine();
        return 0;
    }

    // spin off child process //
    bool ok = manager.SpawnChildNode("Node 1");
    std::cout << "Node 1 spawn ok? " << ok << std::endl;

    ok = manager.SpawnChildNode("Node 2");
    std::cout << "Node 2 spawn ok? " << ok << std::endl;

    int *data = new int[1024];
    for (int i = 0; i < 1024; i++)
        data[i] = i;

    // look for a child process that can execute //
    ok = manager.IssueJob(
        "", // look for any available node
        "child_fn",
        data,
        1024 * sizeof(int),
        true,
        true
    );

    std::cout << "child_fn issue: " << ok << std::endl;

    ok = manager.IssueJob(
        "",
        "test_fn",
        nullptr,
        0
    );

    // synchronize //
    manager.SynchronizeOnChildren();

    // terminate child //
    manager.TerminateChildren();

    return 0;
}
