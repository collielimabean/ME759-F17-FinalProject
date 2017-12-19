#include <iostream>
#include "Task.hpp"

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
        auto child = Task::Create("jobB", parent, jobB);
        child->Run();
        parent->WaitForChildren();
        std::cout << "-------" << std::endl;
    }

    std::cout << "Loop complete" << std::endl;
}


int main()
{
    auto startingTask = Task::Create("jobA", nullptr, jobA);
    startingTask->Run();
    startingTask->Wait();
    return 0;
}