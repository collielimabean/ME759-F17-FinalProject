#include <iostream>
#include <string>
#include <cstring>
#include <mpi.h>
#include "commands.pb.h"
#include <cuda_runtime.h>
#include "MpiContext.hpp"
#include "TaskManager.hpp"


void child_fn(void *data, size_t len)
{
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
    bool ok = manager.SpawnChildNode(
        "New Node",
        "child_fn",
        nullptr,
        0
    );
    
    if (!ok)
    {
        std::cout << "Failed!" << std::endl;
        return 1;
    }


    std::cout << "[main] wait for user..." << std::endl;
    char x;
    std::cin >> x;

    // synchronize //
    manager.SynchronizeOnChildren();

    // terminate child //
    manager.TerminateChildren();

    return 0;
}



/*
constexpr const int err_size = 4;

int getCudaDevices()
{
	int n_devices;
	cudaGetDeviceCount(&n_devices);
	return n_devices;
}

int main(int argc, char **argv)
{
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
	{
		std::cout << "MPI_Init failed!" << std::endl;
		return 1;
	}

    MPI_Comm parent;
    MPI_Comm_get_parent(&parent);

    if (parent == MPI_COMM_NULL)
    {
        std::string program(argv[0]);

		MPI_Comm child;


		MPI_Comm_spawn(program.c_str(), MPI_ARGV_NULL, err_size, MPI_INFO_NULL, 0, MPI_COMM_SELF, &child, MPI_ERRCODES_IGNORE);

        // do master things //

        // send
        char csr_buffer[64];
        dtl::packets::ChildStatusRequest csr;
        csr.SerializeToArray(csr_buffer, 64);
        int e = MPI_Send((void *) csr_buffer, csr.ByteSize(), MPI_CHAR, 0, 0, child);

        char err_str[1024];
        int len;
        MPI_Error_string(e, err_str, &len);

        // listen
        dtl::packets::ChildStatusResponse response;
        char buf[1024];
        memset(buf, 1024, 1024 * sizeof(char));

        MPI_Status status;
        int num_amt;
        
        MPI_Recv(buf, 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, child, &status);
        MPI_Get_count(&status, MPI_CHAR, &num_amt);

        if (!response.ParseFromArray(buf, num_amt))
        {
            std::cout << "[Parent] failed to parse!" << std::endl;
        }
        else
        {
            std::cout << "[Parent] Received message from child with ID " << response.id() << std::endl;          
        }
    }
    else
    {
		int myid = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        if (myid == 0)
        {
            // child listens //
            char buf[1024];
            memset(buf, 1024, 1024 * sizeof(char));

            MPI_Status status;
            int num_amt;
            MPI_Recv(buf, 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, parent, &status);
            MPI_Get_count(&status, MPI_CHAR, &num_amt);

            dtl::packets::ChildStatusRequest csr;
            bool ok = csr.ParseFromArray(buf, num_amt);

            if (ok)
            {
                std::cout << "[Child] Sending response..." << std::endl;
            }
            else
            {
                std::cout << "[Child] Failed to parse!" << std::endl;
            }

            char output[64];
            dtl::packets::ChildStatusResponse response;
            response.set_id(myid);
            response.set_currentjob("A");
            response.set_hasgpu(getCudaDevices > 0);
            response.SerializeToArray(output, 64);
            MPI_Send(output, response.ByteSize(), MPI_CHAR, 0, 0, parent);
        }
    }

    MPI_Finalize();
    return 0;
}

*/