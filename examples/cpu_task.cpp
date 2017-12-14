#include <iostream>
#include "Task.hpp"
#include "Topic.hpp"
#include "TaskMaster.hpp"

using namespace dtl;

void jobB(std::shared_ptr<Task> parent)
{
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Entered jobB, parent is: " << (parent ? parent->GetName() : "null") << std::endl;
}


void jobA(std::shared_ptr<Task> parent)
{
    std::cout << "Entered jobA, parent is: " << (parent ? parent->GetName() : "null") << std::endl;
    for (int i = 0; i < 3; i++)
    {
        std::cout << "Iteration " << i << std::endl;
        auto child = Task::Create("jobB", parent, TaskLocation::Host, jobB);
        child->Run();
        parent->Synchronize();
        std::cout << "-------" << std::endl;
    }

    std::cout << "Loop complete" << std::endl;
}


int main(int argc, char **argv)
{
    auto startingTask = Task::Create("jobA", nullptr, TaskLocation::Host, jobA);
    startingTask->Run();

    char x;
    std::cin >> x;
    return 0;
}