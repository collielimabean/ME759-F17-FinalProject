ME759 Fall 2017 Final Project - Distributed Task Library
==============================

## Default Project 1 Problem Statement Summary:
Suppose we have a mainJobA that is computing a time integration of 
a deforming elastic body. This integration requires a Jacobian J, which is
computationally expensive to generate. A helperJobB is tasked to update that
Jacobian J. HelperJobB may not be fast enough to update the entire Jacobian
before each timestep. 

We should have flexibility on where mainJobA and helperJobB are run.

After talking with Professor Negrut, we have refined this to be something
similar to TBB or OpenMP. In this project, however, the programmer can create a parent task that can spawn child subtasks on either the host, GPU, or a
different node entirely. These subtasks must complete before the parent task can move on.

## High Level Approach
* We need to know how to do task scheduling. How do we establish a
parent task and a child task? The parent task is in a "for loop", and the
child task needs to complete before the parent task can continue.

* Can we do this via a publish/subscribe model? Suppose jobA subs to 
a topic. jobB is spawned, and pubs to the topic. Where's the sync barrier?

* 

## Building
