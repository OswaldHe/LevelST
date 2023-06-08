GCC=g++
OPT=-ltapa -lfrt -lglog -lgflags -lOpenCL -I/opt/tools/xilinx/Vitis_HLS/2021.2/include

trig-solver: solver-general.cpp solver-host.cpp
	g++ -o $@ -O2 $^ $(OPT)

clean:
	rm trig-solver