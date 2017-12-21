#include <iostream>
#include "Task.hpp"

using namespace dtl;

void jobB(std::shared_ptr<Task> parent) 
{ 
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "[JobB] Task Name: " << parent->GetName() << std::endl; 
}

void jobA(std::shared_ptr<Task> parent)
{
    Task::Create("jobB-1", parent, jobB)->Run();
    Task::Create("jobB-2", parent, jobB)->Run();
    Task::Create("jobB-3", parent, jobB)->Run();
}

int main()
{
    auto t = Task::Create("jobA", nullptr, jobA);
    t->Run();
    t->Wait();
    return 0;
}