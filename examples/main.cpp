#include <iostream>
#include "Task.hpp"
#include "Topic.hpp"



void input(dtl::SharedTask parent, int i)
{

}

int main(void)
{
    auto a = dtl::Task::Create<int>(nullptr, &input, dtl::TaskLocation::Host);

    return 0;
}