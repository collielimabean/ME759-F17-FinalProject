#pragma once

#include <string>
#include <map>
#include <memory>

#include "cxxopts.hpp"
#include "MpiContext.hpp"
#include "Task.hpp"


namespace dtl
{
    using TaskFunction = std::function<void(std::shared_ptr<Task>)>;
    using TaskFunctionMap = std::map<std::string, TaskFunction>;
    

    class TaskMaster
    {
    public:
        TaskMaster()
        {
        }

        bool Initialize(const int argc, const char **argv, const TaskFunctionMap& map)
        {
            context.Initialize(&argc, &argv);
            if (!context)
                return false;

            try
            {
                cxxopts::Options options("", "");
                options.add_options()
                    ("__child_process__", "[Internal] Mark this as a child process.")
                ;

                auto result = options.parse(argc, argv);
                this->is_master = result["__child_process__"].count() == 0;
                this->fnMap = map;
            }
            catch (const cxxopts::OptionException& e)
            {
                return false;
            }

            return true;
        }

        void ListenForInstructions()
        {
        }

        bool IsMaster() const 
        {
            return this->is_master;
        }

        const mpi::MpiContext& GetContext() const
        {
            return this->context;
        }

        explicit operator bool() const 
        {
			return static_cast<bool>(context);
		}

    private:
        mpi::MpiContext context;
        TaskFunctionMap fnMap;
        bool is_master;
    };
}