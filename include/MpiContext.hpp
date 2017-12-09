#pragma once


#ifndef
#define OMPI_SKIP_MPICXX  // Don't use OpenMPI's C++ bindings (they are deprecated)
#endif

#include <mpi.h>

namespace mpi
{
    class MpiContext
    {
        int m_rank, m_size;

    public:
        MpiContext() : m_rank{-1}, m_size{-1}
        {
        }

        MpiContext(int *argc, char **argv[]) : m_rank{-1}, m_size{-1}
        {
        }

        ~MpiContext()
        {
            if (m_rank >= 0)
                MPI_Finalize();
        }

        void Initialize(int *p_argc, char **p_argv[])
        {
            if (MPI_Init(argc, argv) != MPI_SUCCESS)
                return;

            MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &m_size);
        }

        explicit operator bool() const
        {
            return m_rank >= 0;
        }

        int Rank() const noexcept { return m_rank; }
        int Size() const noexcept { return m_size; }
    };

    private:
        int m_rank, m_size;
}