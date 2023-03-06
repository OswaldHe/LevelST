#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <tapa.h>
#include <gflags/gflags.h>
#include "mmio.h"
#include "sparse_helper.h"

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using std::vector;

constexpr int NUM_CH = 6;
constexpr int WINDOW_SIZE = 512;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> csr_edge_list_ch,
			tapa::mmaps<int, NUM_CH> csr_edge_list_ptr,
			tapa::mmaps<int, NUM_CH> csc_col_ptr, 
			tapa::mmaps<int, NUM_CH> csc_row_ind, 
			tapa::mmaps<float, NUM_CH> f, 
			tapa::mmap<float> x, 
			int N, tapa::mmap<int> K_csc);

DEFINE_string(bitstream, "", "path to bitstream file");

void convertCSRToCSC(int N, int K /* num of non-zeros*/, 
		const aligned_vector<int>& csr_row_ptr,
		const aligned_vector<int>& csr_col_ind,
		const aligned_vector<float>& csr_val,
		aligned_vector<int>& csc_col_ptr,
		aligned_vector<int>& csc_row_ind,
		aligned_vector<float>& csc_val,
		vector<aligned_vector<int>>& csc_col_ptr_fpga,
		vector<aligned_vector<int>>& csc_row_ind_fpga,
		aligned_vector<int>& K_csc){

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
	int bound = (N%WINDOW_SIZE == 0) ? N/WINDOW_SIZE : N/WINDOW_SIZE+1;
	std::vector<int> k_count(bound, 0);
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

	int prev = 0;
	int acc = 0;
	for(int i = 0; i < N; i++){
		if(i % WINDOW_SIZE == 0) acc = 0;
		int next = csc_col_ptr[i];
		for(int j = prev; j < next; j++){
			if(csc_row_ind[j] < (i/WINDOW_SIZE + 1)*WINDOW_SIZE){
				csc_row_ind_fpga[(i / WINDOW_SIZE) % NUM_CH].push_back(csc_row_ind[j] - (i/WINDOW_SIZE)*WINDOW_SIZE);
				k_count[i/WINDOW_SIZE]++;
				acc++;
			} else {
				break;
			}
		}
		csc_col_ptr_fpga[(i / WINDOW_SIZE) % NUM_CH].push_back(acc);
		prev = next;
	}

	for(int i = 0; i < bound; i++){
		K_csc.push_back(k_count[i]);
	}
}

void generate_edgelist_for_pes(int N,  
		const aligned_vector<int>& csr_row_ptr,
		const aligned_vector<int>& csr_col_ind,
		const aligned_vector<float>& csr_val,
		vector<aligned_vector<ap_uint<64>>>& edge_list_ch,
		vector<aligned_vector<int>>& edge_list_ptr){
			int bound = (N % WINDOW_SIZE == 0) ? N/WINDOW_SIZE:N/WINDOW_SIZE+1;
			for(int i = 0; i < bound; i++){
				vector<aligned_vector<ap_uint<64>>> tmp_edge_list(i+1);
				for(int j = i*WINDOW_SIZE; j < (i+1)*WINDOW_SIZE && j < N; j++){
					int start = (j == 0)? 0 : csr_row_ptr[j-1];
					int end = csr_row_ptr[j];
					for(int k = start; k < end; k++){
						ap_uint<64> a = 0;
						a(63, 52) = (ap_uint<12>)(j - i*WINDOW_SIZE & 0xFFF);
						a(51, 32) = (ap_uint<20>)(csr_col_ind[k] & 0xFFFFF);
						a(31, 0) = tapa::bit_cast<ap_uint<32>>(csr_val[k]);
						tmp_edge_list[csr_col_ind[k]/WINDOW_SIZE].push_back(a);
					}
				}
				
				//std::clog << "pe: " << i << std::endl;
				for(int j = 0; j < i+1; j++){
					//std::clog << tmp_edge_list[j].size() << std::endl;
					edge_list_ptr[i%NUM_CH].push_back(tmp_edge_list[j].size());
					for(int k = 0; k < tmp_edge_list[j].size(); k++){
						edge_list_ch[i%NUM_CH].push_back(tmp_edge_list[j][k]);
					}
					int rest = tmp_edge_list[j].size() % 8 == 0 ? 0 : 8 - (tmp_edge_list[j].size() % 8);
					for(int k = 0; k < rest; k ++){
						ap_uint<64> a = 0;
						a(63, 52) = (ap_uint<12>) 0xFFF;
						edge_list_ch[i%NUM_CH].push_back(a);
					}
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

	// const int N = argc > 1 ? atoll(argv[1]) : 8192;
	// const int remainder = N % WINDOW_SIZE;
	// int K = N*2-N/WINDOW_SIZE;
	// if(remainder != 0) K --;
	// int K = N*2 - 1;
	int M, K, nnz;
    vector<int> CSRRowPtr;
    vector<int> CSRColIndex;
    vector<float> CSRVal;

	read_suitsparse_matrix_FP64("lp1.mtx",
                           CSRRowPtr,
                           CSRColIndex,
                           CSRVal,
                           M,
                           K,
                           nnz);

	aligned_vector<float> A;
	aligned_vector<int> IA;
	aligned_vector<int> JA;
	aligned_vector<float> f;
	// aligned_vector<float> x(N);
	aligned_vector<int> cycle(1, 0); 

	extract_lower_triangular_matrix(M, K, nnz, CSRRowPtr, CSRColIndex, CSRVal, IA, JA, A);

	nnz = A.size();

	IA.erase(IA.begin());
	const int N = M;

	// for kernel
	vector<aligned_vector<ap_uint<64>>> edge_list_ch(NUM_CH);
	vector<aligned_vector<int>> edge_list_ptr(NUM_CH);
	vector<aligned_vector<float>> f_fpga(NUM_CH);
	aligned_vector<float> x_fpga(N, 0.0);
	aligned_vector<int> K_fpga;

	// for(int i = 0; i < NUM_CH; i++){
	// 	for(int j = 0; j < N/NUM_CH; j++){
	// 		x_fpga[i].push_back(0.0);
	// 	}
	// }

	for(int i = 0; i < N; i++){
		f.push_back(i/10.0);
	    f_fpga[(i/WINDOW_SIZE)%NUM_CH].push_back(f[i]);
	}

	std::clog << M << std::endl;
	std::clog << nnz << std::endl;


	// int ind = 0;
	// int acc = 0;
	// //populate lower triangular sparse matrix
    // for (int i = 0; i < N; ++i) {
	// 	f[i] = 0.5*(i+1);
	// 	f_fpga[(i/WINDOW_SIZE)%NUM_CH].push_back(f[i]);
	// 	if(i == 0) {
	// 		A[ind] = i+1;
	// 		JA[ind] = i;
	// 		acc += 1;
	// 		ind++;
	// 	} else {
	// 		A[ind] = 1.f;
	// 		JA[ind] = i-(i/2)-1;
	// 		A[ind+1] = i+1;
	// 		JA[ind+1] = i;
	// 		if(i % WINDOW_SIZE == 0) {
	// 			K_fpga.push_back(acc);
	// 			acc = 0;
	// 		}
	// 		acc +=2;
	// 		ind+=2;
	// 	}
	// 	IA[i] = ind;
    // }
	// K_fpga.push_back(acc);
	//readCSRMatrix("L_can256.txt", IA, JA, A);
	
	//std::clog << K_fpga[1] << std::endl;
	// std::clog << IA.size() << std::endl;
	// std::clog << A.size() << std::endl;

	aligned_vector<float> csc_val(nnz, 0.0);
	aligned_vector<int> csc_col_ptr(N, 0);
	aligned_vector<int> csc_row_ind(nnz, 0);
	aligned_vector<int> K_csc;

	//for kernel
	vector<aligned_vector<int>> csc_col_ptr_fpga(NUM_CH);
	vector<aligned_vector<int>> csc_row_ind_fpga(NUM_CH);

	convertCSRToCSC(N, nnz, IA, JA, A, csc_col_ptr, csc_row_ind, csc_val, csc_col_ptr_fpga, csc_row_ind_fpga, K_csc);
	generate_edgelist_for_pes(N, IA, JA, A, edge_list_ch, edge_list_ptr);
	
	// std::clog << K_csc[156] << std::endl;
	// std::clog << edge_list_ptr[1].size() << std::endl;
	// std::clog << csc_row_ind[2] << std::endl;

	for(int i = 0; i < NUM_CH; i++){
		if(edge_list_ch[i].size() == 0){
			for(int j = 0; j < 8; j++){
				ap_uint<64> a = 0;
				edge_list_ch[i].push_back(a);
			}
		}
		if(edge_list_ptr[i].size() == 0){
			edge_list_ptr[i].push_back(0);
		}
		if(csc_col_ptr_fpga[i].size() == 0){
			csc_col_ptr_fpga[i].push_back(0);
		}
		if(csc_row_ind_fpga[i].size() == 0){
			csc_row_ind_fpga[i].push_back(0);
		}
		if(f_fpga[i].size() == 0){
			f_fpga[i].push_back(0.0);
		}
	}

	//triangular solver in cpu
	vector<float> expected_x(N);
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
                        tapa::read_only_mmaps<ap_uint<64>, NUM_CH>(edge_list_ch).reinterpret<ap_uint<512>>(),
						tapa::read_only_mmaps<int, NUM_CH>(edge_list_ptr),
						tapa::read_only_mmaps<int, NUM_CH>(csc_col_ptr_fpga),
						tapa::read_only_mmaps<int, NUM_CH>(csc_row_ind_fpga),
						tapa::read_only_mmaps<float, NUM_CH>(f_fpga),
                        tapa::read_write_mmap<float>(x_fpga), N, tapa::read_only_mmap<int>(K_csc));
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
	std::clog << "cycle count: " << cycle[0] << std::endl;
	
	int unmatched = 0;

        for (int i = 0; i < N; ++i){
		if(std::fabs(x_fpga[i]-expected_x[i]) > 1.0e-5 * K){
			std::clog << "index: " << i << ", expected: " << expected_x[i] << ", actual: " << x_fpga[i] << std::endl;
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
