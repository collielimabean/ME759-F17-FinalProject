WFLAGS	:= -Wall -Wextra
OPT		:= -O3
CXXSTD	:= -std=c++14
INCLUDES := -I./include/ -I./gen/
.DEFAULT_GOAL := all
EXECS := mpi_task cpu_task

all : Makefile $(EXECS)

cpu_task:
	g++ $(INCLUDES) -g `mpicxx -showme:compile` -Wall -Wextra -pedantic -std=c++11 -o cpu_task examples/cpu_task.cpp src/Task.cpp

mpi_task:
	mkdir -p gen
	protoc -I=include/ --proto_path=src/ --cpp_out=gen/  src/*.proto
	nvcc -c $(INCLUDES) -g -G `mpicxx -showme:compile | sed 's/-pthread//g'` -std=c++11 /usr/local/protobuf/3.5.0/lib/libprotobuf.a src/*.cpp gen/*.pb.cc examples/mpi_task.cu -ccbin=$(CU_CCBIN)
	nvcc -std=c++11 -g -G -lm -lcudart -L/usr/lib/openmpi/lib -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm *.o  /usr/local/protobuf/3.5.0/lib/libprotobuf.a -o $@ 

.PHONY: clean
clean:
	@ rm -f $(EXECS) *.o -rf gen
