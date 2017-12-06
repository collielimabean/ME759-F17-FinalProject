#include <iostream>
#include "Task.hpp"
#include "Topic.hpp"

using namespace dtl;

void jobB(std::shared_ptr<Task> parent)
{
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Entered jobB, parent is: " << (parent ? parent->GetName() : "null") << std::endl;
}


void jobA(std::shared_ptr<Task> parent)
{
    std::cout << "Entered jobA, parent is: " << (parent ? parent->GetName() : "null") << std::endl;

    auto child = Task::Create(parent, TaskLocation::Host, jobB);
    child->Run();


}


int main(void)
{
    srand(time(nullptr));
    auto startingTask = Task::Create(nullptr, TaskLocation::Host, jobA);
    startingTask->Run();

    char x;
    std::cin >> x;
    return 0;
}