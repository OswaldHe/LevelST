GCC=g++
OPT=-I${TAPA_ROOT}/include -I/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.1.0/fpga-runtime-0.0.20221212.1-bhpyoymq4dpp7xjlkwhfctc4b4a2zckg/include -I/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.1.0/glog-0.6.0-jwqrskeqi6xgr7mucxz5dzhb7inup6fa/include -I/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.1.0/gflags-2.2.2-rp5kjnrnuwmeqen72tn6qcth4dm3xl25/include -L${TAPA_ROOT}/lib -L/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.1.0/fpga-runtime-0.0.20221212.1-bhpyoymq4dpp7xjlkwhfctc4b4a2zckg/lib -L/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.1.0/glog-0.6.0-jwqrskeqi6xgr7mucxz5dzhb7inup6fa/lib -L/opt/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.1.0/gflags-2.2.2-rp5kjnrnuwmeqen72tn6qcth4dm3xl25/lib -ltapa -lfrt -lglog -lgflags -lOpenCL -I/opt/xilinx/Vitis_HLS/2021.2/include

trig-solver: solver-general.cpp solver-host.cpp
	$(GCC) -o $@ -O2 $^ $(OPT)

clean:
	rm trig-solver
