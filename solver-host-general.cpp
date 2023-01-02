#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <tapa.h>
#include <gflags/gflags.h>

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

void TrigSolver(tapa::mmap<float> A, tapa::mmap<int> IA, tapa::mmap<int> JA, tapa::mmap<int> csc_col_ptr, tapa::mmap<int> csc_row_ind, tapa::mmap<float> f, tapa::mmap<float> x, int N, int K, tapa::mmap<int> cycle_count);

DEFINE_string(bitstream, "", "path to bitstream file");

void convertCSRToCSC(int N, int K /* num of non-zeros*/, 
		const aligned_vector<int>& csr_row_ptr,
		const aligned_vector<int>& csr_col_ind,
		const aligned_vector<float>& csr_val,
		aligned_vector<int>& csc_col_ptr,
		aligned_vector<int>& csc_row_ind,
		aligned_vector<float>& csc_val){

	csc_col_ptr.resize(N, 0);
	csc_row_ind.resize(K, 0);
	csc_val.resize(K, 0.0);

	for(int i = 0; i < K; i++){
		csc_col_ptr[csr_col_ind[i]]++;
	}

	for(int i = 1; i < N; i++){
		csc_col_ptr[i] += csc_col_ptr[i-1];
	}

	std::vector<int> col_nz(N, 0);
	for(int i = 0; i < N; i++){
		for(int j = (i==0)?0:csr_row_ptr[i-1]; j < csr_row_ptr[i]; j++){
			int c = csr_col_ind[j];
			int r = i;
			float val = csr_val[j];

			int pos = ((c == 0) ? 0 : csc_col_ptr[c-1]) + col_nz[c];
			csc_val[pos] = val;
			csc_row_ind[pos] = r;
			col_nz[c]++;
		}
	}
}

void readCSRMatrix(std::string filename, aligned_vector<int>& csr_row_ptr, aligned_vector<int>& csr_col_ind, aligned_vector<float>& csr_val){
	std::ifstream file(filename);
	std::string line;
	int line_num = 0;
	while(std::getline(file, line)){
		std::stringstream s_stream(line);
		while(s_stream.good()){
			std::string substr;
			std::getline(s_stream, substr, ',');
			if(line_num == 0){
				if(substr != "0") csr_row_ptr.push_back(std::stoi(substr));
			} else if(line_num == 1) {
				csr_col_ind.push_back(std::stoi(substr));
			} else if(line_num == 2){
				csr_val.push_back((float)(std::stod(substr)));
			}
		}
		line_num++;
	}
}

int main(int argc, char* argv[]){
        gflags::ParseCommandLineFlags(&argc, &argv, true);

        const int N = argc > 1 ? atoll(argv[1]) : 256;
        aligned_vector<float> A;
        aligned_vector<int> IA;
        aligned_vector<int> JA;
	aligned_vector<float> f(N);
	aligned_vector<float> x(N);
	aligned_vector<int> cycle(1); 

	int acc = 1;
	int ind = 0;
	//populate lower triangular sparse matrix
        for (int i = 0; i < N; ++i) {
               // IA[i] = acc;
		//acc+=2;
		f[i] = 0.5*(i+1);
		/*
		if(i == 0) {
			A[ind] = i+1;
			JA[ind] = i;
			ind++;
		} else {
			A[ind] = 1.f;
			JA[ind] = i/2;
			A[ind+1] = i+1;
			JA[ind+1] = i;
			ind+=2;
		}*/
        }
	readCSRMatrix("L_can256.txt", IA, JA, A);
	
	std::clog << IA.size() << std::endl;
	std::clog << A.size() << std::endl;

	int K = A.size();

	aligned_vector<float> csc_val(K, 0.0);
	aligned_vector<int> csc_col_ptr(N, 0);
	aligned_vector<int> csc_row_ind(K, 0);
	convertCSRToCSC(N, K, IA, JA, A, csc_col_ptr, csc_row_ind, csc_val);

	//triangular solver in cpu
	float expected_x[N];
	int next = 0;
	for(int i = 0; i < N; i++){
		float image = f[i];
		float num = (i == 0) ? IA[0] : IA[i] - IA[i-1];
		for(int j = 0; j < num-1; j++){
			image -= expected_x[JA[next]]*A[next];
			next++;
		}
		expected_x[JA[next]] = image / A[next];
		//sanity check
		/* 
		if(i == 0){
			if(std::fabs(expected_x[i]-0.5) > 1.0e-5){
				std::clog << "Incorrect base solver! Index: " << i << ", expect: 0.5, actual: " << expected_x[i] << std::endl;
				return 1;
			}
		}else{
			if(std::fabs(expected_x[i]-((0.5*(i+1))-expected_x[i/2])/(i+1)) > 1.0e-5){
                                std::clog << "Incorrect base solver! Index: " << i << ", expect: " << ((0.5*(i+1))-expected_x[i/2])/(i+1) << ", actual: " << expected_x[i] << std::endl;
                                return 1;
                        }
		}
		*/
		next++;
	}

	cycle[0] = 0;

        int64_t kernel_time_ns = tapa::invoke(TrigSolver, FLAGS_bitstream,
                        tapa::read_only_mmap<float>(A),
                        tapa::read_only_mmap<int>(IA),
			tapa::read_only_mmap<int>(JA),
			tapa::read_only_mmap<int>(csc_col_ptr),
			tapa::read_only_mmap<int>(csc_row_ind),
			tapa::read_only_mmap<float>(f),
                        tapa::write_only_mmap<float>(x), N, K, tapa::write_only_mmap<int>(cycle));
        std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
	std::clog << "cycle count: " << cycle[0] << std::endl;
	
	int unmatched = 0;

        for (int i = 0; i < N; ++i){
		if(std::fabs(x[i]-expected_x[i]) > 1.0e-5 * K){
			std::clog << "index: " << i << ", expected: " << expected_x[i] << ", actual: " << x[i] << std::endl;
			unmatched++;
		}
        }

        if(unmatched == 0) { // tolerance dependends on number of elements
                std::clog << "PASS!" << std::endl;
        }else{
                std::clog << "FAIL!" << std::endl;
        }
        return unmatched != 0 ? 1 : 0;
}