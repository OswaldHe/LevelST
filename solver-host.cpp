#include <ap_int.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>
#include <cassert>
#include <tapa.h>
#include <unordered_map>
#include <queue>
#include <gflags/gflags.h>
#include "mmio.h"
#include "sparse_helper.h"
#include <chrono>

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using std::vector;

constexpr int NUM_CH = 16;
constexpr int WINDOW_SIZE = 8192;
constexpr int WINDOW_SIZE_div_2 = 8192;
constexpr int WINDOW_LARGE_SIZE = WINDOW_SIZE*NUM_CH;
int WINDOW_SIZE_SPMV = 32;
int MULT_SIZE = 1;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> comp_packet_ch,
			tapa::mmap<int> merge_inst_ptr,
			tapa::mmaps<float_v16, 2> f, 
			tapa::mmap<float_v16> x, 
			tapa::mmap<int> if_need,
			tapa::mmap<int> block_fwd,
			const int N,
			const int NUM_ITE,
			const int A_LEN
			);

DEFINE_string(bitstream, "", "path to bitstream file");
DEFINE_string(file, "lp1.mtx", "path to matrix file");

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

void generate_edgelist_spmv_cyclic(
	int N,
	const aligned_vector<int>& csr_row_ptr,
	const aligned_vector<int>& csr_col_ind,
	const aligned_vector<float>& csr_val,
	vector<aligned_vector<ap_uint<64>>>& edge_list_ch,
	vector<aligned_vector<int>>& edge_list_ptr,
	aligned_vector<int>& if_need
){
	int bound = (N % WINDOW_LARGE_SIZE == 0) ? N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
	for(int i = 0; i < bound; i++){
		if(i == 0) continue;
		// distribute to each peg
		vector<vector<aligned_vector<ap_uint<64>>>> tmp_edge_list(NUM_CH, vector<aligned_vector<ap_uint<64>>>((i+1)*NUM_CH));
		for(int j = i*WINDOW_LARGE_SIZE; j < (i+1)*WINDOW_LARGE_SIZE && j < N; j++){
			int start = (j == 0)? 0 : csr_row_ptr[j-1];
			int end = csr_row_ptr[j];
			for(int k = start; k < end; k++){
				int row = (j - i*WINDOW_LARGE_SIZE)/NUM_CH;
				int ch = (j - i*WINDOW_LARGE_SIZE)%NUM_CH;
				ap_uint<64> a = 0;
				a(63, 48) = (ap_uint<16>)(row & 0xFFFF);
				a(47, 32) = (ap_uint<16>)((csr_col_ind[k]%(WINDOW_SIZE_div_2)) & 0xFFFF);
				a(31, 0) = tapa::bit_cast<ap_uint<32>>(csr_val[k]);
				tmp_edge_list[ch][csr_col_ind[k]/WINDOW_SIZE_div_2].push_back(a);
			}
		}

		// std::clog << "pe: " << i << std::endl;
		// int count_non_zero = 0;
		// int total_cycle = 0;
		for(int ch = 0; ch < NUM_CH; ch++){
			for(int j = 0; j < i*NUM_CH; j++){
				// if(tmp_edge_list[j].size() != 0) std::clog << tmp_edge_list[j].size() << std::endl;
				int list_size = tmp_edge_list[ch][j].size();
				int total_size = 0;
				vector<bool> used_edge(list_size, false);
				vector<std::set<int>> row_raw(8); // last 11 elements
				int next_slot = 0;
				int pack_chunk_count = 0;
				for(int k = 0; k < list_size;){
					std::set<int> row;
					int pack_count = 0;
					vector<ap_uint<64>> packet(8);
					row_raw[next_slot].clear();
					for(int l = 0; l < 8; l++){
						ap_uint<64> a = 0;
						a(63, 48) = (ap_uint<16>) 0xFFFF;
						packet[l] = a;
					}
					for(int l = 0; l < list_size; l++){
						int row_i = (tmp_edge_list[ch][j][l](63, 48) | (int) 0);
						bool found = false;
						for(int n = 0; n < 8; n++){
							if(row_raw[n].find(row_i) != row_raw[n].end()){
								found = true;
								break;
							}
						}
						if(!used_edge[l] && row.find(row_i%8) == row.end() && !found){
							packet[row_i%8] = tmp_edge_list[ch][j][l];
							row.insert(row_i%8);
							row_raw[next_slot].insert(row_i);
							used_edge[l] = true;
							pack_count++;
							if(pack_count == 8) break;
						}
					}
					k+= pack_count;
					total_size += 8;
					for(int l = 0; l < 8; l++){
						edge_list_ch[ch].push_back(packet[l]);
					}
					pack_chunk_count++;
					next_slot = (next_slot + 1) % 8;
				}
				// if(total_size != 0) {
				// 	count_non_zero++;
				// 	total_cycle+=(11+total_size/8);
				// }
				edge_list_ptr[ch].push_back(total_size);
			}
		}
		// std::clog << std::max(count_non_zero*WINDOW_SIZE_SPMV, total_cycle) << std::endl;
	}
	// for(int i = 0; i < NUM_CH; i++){
	// 	int size = edge_list_ptr[i].size();
	// 	edge_list_ptr[i].insert(edge_list_ptr[i].begin(), size);
	// }

	for(int i = 0; i < edge_list_ptr[0].size(); i++){
		bool is_non_zero = false;
		for(int j = 0; j < NUM_CH; j++){
			if(edge_list_ptr[j][i] != 0) is_non_zero = true;
		}
		if(is_non_zero) if_need.push_back(1);
		else if_need.push_back(0);
	}
	int size = if_need.size();
	if_need.insert(if_need.begin(), size);
}

void process_spmv_ptr(
	vector<aligned_vector<ap_uint<64>>>& edge_list_ch,
	vector<aligned_vector<int>>& edge_list_ptr,
	vector<aligned_vector<ap_uint<64>>>& edge_list_ch_out,
	aligned_vector<int>& edge_list_ptr_out,
	int& max){
		vector<int> offset(NUM_CH, 0);
		max = 0;
		for(int i = 0; i < edge_list_ptr[0].size(); i++){
			int maxLen = 0;
			for(int j = 0; j < NUM_CH; j++){
				if(edge_list_ptr[j][i] > maxLen){
					maxLen = edge_list_ptr[j][i];
				}
			}
			edge_list_ptr_out.push_back(maxLen);
			max += maxLen;
			for(int j = 0; j < NUM_CH; j++){
				for(int k = offset[j]; k < edge_list_ptr[j][i] + offset[j]; k++){
					edge_list_ch_out[j].push_back(edge_list_ch[j][k]);
				}
				for(int k = 0; k < maxLen - edge_list_ptr[j][i]; k++){
					ap_uint<64> a = 0;
					a(63, 48) = (ap_uint<16>) 0xFFFF;
					edge_list_ch_out[j].push_back(a);
				}
				offset[j]+=edge_list_ptr[j][i];
			}
		}
	}

void generate_dependency_graph_for_pes_cyclic(
	int N,
	const aligned_vector<int>& csr_row_ptr,
	const aligned_vector<int>& csr_col_ind,
	const aligned_vector<float>& csr_val,
	vector<aligned_vector<ap_uint<64>>>& dep_graph_ch,
	aligned_vector<int>& dep_graph_ptr
){
	int bound = (N % WINDOW_LARGE_SIZE == 0) ? N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
	int total_iter_count = 0;
	int total_effect_iter_count = 0;
	for(int i = 0; i < bound; i++){
		// std::clog << "level: " << i << std::endl;
		vector<int> csrRowPtr;
		std::unordered_map<int, vector<edge<float>>> dep_map;

		//extract csr
		int row_ptr = 0;
		for(int j = 0; j < WINDOW_LARGE_SIZE && j < N - i * WINDOW_LARGE_SIZE; j++){
			int start = (i*WINDOW_LARGE_SIZE+j == 0) ? 0:csr_row_ptr[i*WINDOW_LARGE_SIZE+j-1];
			for(int k = start; k < csr_row_ptr[i*WINDOW_LARGE_SIZE+j]; k++){
				if(csr_col_ind[k] >= i * WINDOW_LARGE_SIZE){
					int c = csr_col_ind[k] - i * WINDOW_LARGE_SIZE;
					float v = csr_val[k];
					if(c == j) v = 1.0/v;
					edge<float> e(c, j, v);
					if(dep_map.find(c) == dep_map.end()){
						vector<edge<float>> vec;
						dep_map[c] = vec;
					}
					dep_map[c].push_back(e);
					row_ptr++;
				}
			}
			csrRowPtr.push_back(row_ptr);
		}

		//generate level-sets
		vector<int> parents;
		std::queue<int> roots;
		int prev = 0;
		for(int j = 0; j < WINDOW_LARGE_SIZE && j < N - i * WINDOW_LARGE_SIZE; j++){
			parents.push_back(csrRowPtr[j]-prev-1);
			if(csrRowPtr[j]-prev-1 == 0) {
				roots.push(j);
			}
			prev = csrRowPtr[j];
		}

		vector<int> inst;
		int layer_count = 0;

		while(!roots.empty()){
			int size = roots.size();

			//TODO: split node/edge list into 8 PEs
			aligned_vector<vector<ap_uint<64>>> nodes_pe(NUM_CH);
			aligned_vector<vector<ap_uint<64>>> edges_pe(NUM_CH*NUM_CH);
			vector<int> node_count_pe(NUM_CH);
			aligned_vector<vector<int>> edge_count_pe(NUM_CH);

			for(int j = 0; j < size; j++){
				int root = roots.front();
				for(auto e : dep_map[root]){
					ap_uint<64> a;
					int ch = e.row%NUM_CH;
					a(63,48) = (ap_uint<16>)((e.row/NUM_CH) & 0xFFFF);
					a(47,32) = (ap_uint<16>)((e.col/NUM_CH) & 0xFFFF);
					a(31,0) = tapa::bit_cast<ap_uint<32>>(e.attr);
					if(e.row == e.col){
						nodes_pe[ch].push_back(a);
					}else{
						edges_pe[ch*NUM_CH+(e.col%NUM_CH)].push_back(a);
						parents[e.row]--;
						if(parents[e.row] == 0) {
							roots.push(e.row);
						}
					}
				}
				roots.pop();
			}


			int maxNode = 0;
			int maxEdge = 0;
			vector<vector<ap_uint<64>>> dep_graph_tmp(NUM_CH);

			int effect_edge = 0;

			//schedule each PEs
			for(int pe_i = 0; pe_i < NUM_CH; pe_i++){
				int max_effect_edge = 0;
				int node_count = 0;
				vector<ap_uint<64>> nodes = nodes_pe[pe_i];

			int rem_node_num = nodes.size();
			int pushed_node_count = 0;
			vector<bool> used_node(rem_node_num, false);
			while(rem_node_num > pushed_node_count){
				std::set<int> row;
				vector<ap_uint<64>> packet(8);
				for(int n = 0; n < 8; n++){
					ap_uint<64> a = 0;
					a(63,48) = 0xFFFF;
					a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
					packet[n] = a;
				}
				for(int n = 0; n < rem_node_num; n++){
					if(!used_node[n]){
						auto nd = nodes[n];
						int row_i = (nd(63,48) | (int) 0);
						if(row.find(row_i % 8) == row.end()){
							row.insert(row_i % 8);
							packet[row_i % 8] = nd;
							used_node[n] = true;
							pushed_node_count++;
						}
					}
					if(row.size() == 8) break;
				}
				for(int n = 0; n < 8; n++){
					dep_graph_tmp[pe_i].push_back(packet[n]);
				}
				node_count++;
			}
			node_count_pe[pe_i] = node_count;
			if(node_count > maxNode) maxNode = node_count;

			for(int block_id = 0; block_id < NUM_CH; block_id++){
				int edge_count = 0;
				vector<ap_uint<64>> edge_list = edges_pe[pe_i*NUM_CH+block_id];

				int rem_edge_num = edge_list.size();
				int pushed_edge_count = 0;
				vector<bool> used_edge(rem_edge_num, false);
				int pack_chunk_count = 0;
				vector<std::set<int>> row_raw(8); // last 8 elements
				int next_slot = 0;
				// std::set<int> row_raw;
				while(pushed_edge_count < rem_edge_num){
					std::set<int> row;
					// std::set<int> col;
					vector<ap_uint<64>> packet(8);
					row_raw[next_slot].clear();
					for(int n = 0; n < 8; n++){
						ap_uint<64> a = 0;
						a(63,48) = 0xFFFF;
						packet[n] = a;
					}
					for(int n = 0; n < rem_edge_num; n++){
						if(!used_edge[n]){
							auto e = edge_list[n];
							int row_i = (e(63,48) | (int) 0);
							bool found = false;
							for(int m = 0; m < 8; m++){
								if(row_raw[m].find(row_i) != row_raw[m].end()){
									found = true;
									break;
								}
							}
							if(row.find(row_i%8) == row.end() && !found){
								row.insert(row_i%8);
								row_raw[next_slot].insert(row_i);
								packet[row_i % 8] = e;
								used_edge[n] = true;
							}
						}
						if(row.size() == 8) break;
					}
					for(int n = 0; n < 8; n++){
						dep_graph_tmp[pe_i].push_back(packet[n]);
					}
					pack_chunk_count++;
					edge_count++;
					next_slot = (next_slot+1)%8;
					pushed_edge_count += row.size();
				}
				edge_count_pe[pe_i].push_back(edge_count);
				if(edge_count > maxEdge) maxEdge = edge_count;
				if(edge_count > max_effect_edge) max_effect_edge = edge_count;
			}
			effect_edge += max_effect_edge;

			}

			inst.push_back(maxNode);
			inst.push_back(maxEdge);

			total_iter_count += (maxNode + maxEdge)*NUM_CH;
			total_effect_iter_count += maxNode * NUM_CH + effect_edge;

			//process dep graph ptr
			for(int pe_i = 0; pe_i < NUM_CH; pe_i++){
				// std::clog << "pe: " << pe_i << std::endl;
				int offset = 0;
				int prev_size = dep_graph_ch[pe_i].size();
				vector<ap_uint<64>> node_tmp_cache;
				for(int b = offset; b < offset + node_count_pe[pe_i]*8; b++){
					node_tmp_cache.push_back(dep_graph_tmp[pe_i][b]);
				}
				for(int b = 0; b < maxNode - node_count_pe[pe_i]; b++){
					for(int n = 0; n < 8; n++){
						ap_uint<64> a = 0;
						a(63,48) = 0xFFFF;
						a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
						node_tmp_cache.push_back(a);
					}
				}
				offset += node_count_pe[pe_i]*8;
				for(int l = 0; l < maxNode*8; l++){
					dep_graph_ch[pe_i].push_back(node_tmp_cache[l]);
				}
				for(int b = 0; b < NUM_CH; b++){
					for(int l = offset; l < offset + edge_count_pe[pe_i][b]*8; l++){
						dep_graph_ch[pe_i].push_back(dep_graph_tmp[pe_i][l]);
					}
					for(int l = 0; l < maxEdge - edge_count_pe[pe_i][b]; l++){
						for(int n = 0; n < 8; n++){
							ap_uint<64> a = 0;
							a(63,48) = 0xFFFF;
							dep_graph_ch[pe_i].push_back(a);
						}
					}
					offset += edge_count_pe[pe_i][b]*8;
				}
			}
			layer_count++;

		}
		dep_graph_ptr.push_back(layer_count);
		for(auto num : inst){
			dep_graph_ptr.push_back(num);
		}

	}

	// LOG(INFO) << "total count: " << total_iter_count;
	// LOG(INFO) << "total effective count: " << total_effect_iter_count;
}

void merge_ptr(int N,
	aligned_vector<int>& dep_graph_ptr,
	aligned_vector<int>& edge_list_ptr,
	aligned_vector<int>& merge_inst_ptr){
	
	int bound = (N%WINDOW_LARGE_SIZE == 0)?N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
	int edge_list_offset = 0;
	int dep_graph_offset = 0;
	for(int round = 0; round < bound; round++){
		merge_inst_ptr.push_back((NUM_CH*round)*MULT_SIZE);
		int N_level = dep_graph_ptr[dep_graph_offset++];
		merge_inst_ptr.push_back(N_level);
		int sum = 0;
		for(int i = 0; i < (NUM_CH*round)*MULT_SIZE; i++){
			sum +=edge_list_ptr[i+edge_list_offset];
		}
		merge_inst_ptr.push_back((sum + 7)/8);
		for(int i = 0; i < (NUM_CH*round)*MULT_SIZE; i++){
			merge_inst_ptr.push_back(edge_list_ptr[i+edge_list_offset]);
		}
		edge_list_offset+=(NUM_CH*round)*MULT_SIZE;
		for(int i = 0; i < N_level*2; i++){
			merge_inst_ptr.push_back(dep_graph_ptr[i+dep_graph_offset]);
		}
		dep_graph_offset+= N_level*2;
	}
	int size = merge_inst_ptr.size();
	merge_inst_ptr.insert(merge_inst_ptr.begin(), size);
}

void merge_data(
	int N,
	aligned_vector<int>& dep_graph_ptr,
	aligned_vector<int>& edge_list_ptr,
	vector<aligned_vector<ap_uint<64>>>& dep_graph_ch,
	vector<aligned_vector<ap_uint<64>>>& edge_list_ch,
	vector<aligned_vector<ap_uint<64>>>& comp_packet_ch
){
	int bound = (N%WINDOW_LARGE_SIZE == 0)?N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
	for(int ch = 0; ch < NUM_CH; ch++){
		int edge_list_offset = 0;
		int dep_graph_offset = 0;
		int edge_list_ch_offset = 0;
		int dep_graph_ch_offset = 0;

		for(int round = 0; round < bound; round++){
			for(int i = 0; i < (NUM_CH*round)*MULT_SIZE; i++){
				int len = edge_list_ptr[i+edge_list_offset];
				for(int j = 0; j < len; j++){
					comp_packet_ch[ch].push_back(edge_list_ch[ch][j+edge_list_ch_offset]);
				}
				edge_list_ch_offset+=len;
			}
			edge_list_offset += (NUM_CH*round)*MULT_SIZE;
			int N_level = dep_graph_ptr[dep_graph_offset++];
			for(int i = 0; i < N_level; i++){
				int N_node = dep_graph_ptr[i*2 + dep_graph_offset];
				int N_edge = dep_graph_ptr[i*2 + 1 + dep_graph_offset];
				for(int j = 0; j < N_node; j++){
					for(int k = 0; k < 8; k++){
						comp_packet_ch[ch].push_back(dep_graph_ch[ch][(j*8 + k) + dep_graph_ch_offset]);
					}
				}
				dep_graph_ch_offset+=N_node*8;
				for(int j = 0; j < N_edge * NUM_CH; j++){
					for(int k = 0; k < 8; k++){
						comp_packet_ch[ch].push_back(dep_graph_ch[ch][(j*8 + k) + dep_graph_ch_offset]);
					}
				}
				dep_graph_ch_offset+=(N_edge)*NUM_CH * 8;
			}
			dep_graph_offset+=N_level*2;
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

	const int D = argc > 1 ? atoll(argv[1]) : 60000;
	int M, K, nnz;
    vector<int> CSRRowPtr;
    vector<int> CSRColIndex;
    vector<float> CSRVal;

	read_suitsparse_matrix_FP64(FLAGS_file.c_str(), 
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

	if(argc > 1) M = D;

	std::clog << "get matrix, start scheduling..." << std::endl;

	extract_lower_triangular_matrix(M, K, nnz, CSRRowPtr, CSRColIndex, CSRVal, IA, JA, A);

	std::clog << "extract lower triangular matrix..." << std::endl;

	nnz = A.size();

	IA.erase(IA.begin());
	const int N = M;

	// for kernel
	vector<aligned_vector<ap_uint<64>>> edge_list_ch(NUM_CH);
	vector<aligned_vector<int>> edge_list_ptr(NUM_CH);
	vector<aligned_vector<ap_uint<64>>> edge_list_ch_mod(NUM_CH);
	aligned_vector<int> edge_list_ptr_mod;
	vector<aligned_vector<float>> f_fpga(2);
	aligned_vector<float> x_fpga(((N+15)/16)*16, 0.0);
	aligned_vector<int> K_fpga;

	for(int i = 0; i < N; i++){
		f.push_back(i/10.0);
	}

	if((N % 16) != 0){
		int index = ((N-1)/WINDOW_SIZE)% NUM_CH;
		int rem = 16 - (N % 16);
		for(int i = 0; i < rem; i++) {
			f.push_back(0.0);
		}
	}

	// distribute f
	int num_ite = (f.size() / 8);
	for(int i = 0, c_idx = 0; i < num_ite; i++){
		for(int j = 0; j < 8; j++){
			f_fpga[c_idx].push_back(f[i*8+j]);
		}
		c_idx++;
		if(c_idx == 2) c_idx = 0;
	}

	for(int i = 0; i < 2; i++){
		int size = f_fpga[i].size();
		if((size % 16) != 0){
			for(int j = 0; j < 16 - (size % 16); j++){
				f_fpga[i].push_back(0.0);
			}
		}
	}

	std::clog << M << std::endl;
	std::clog << nnz << std::endl;

	aligned_vector<float> csc_val(nnz, 0.0);
	aligned_vector<int> csc_col_ptr(N, 0);
	aligned_vector<int> csc_row_ind(nnz, 0);
	aligned_vector<int> K_csc;

	//for kernel
	vector<aligned_vector<int>> csc_col_ptr_fpga(NUM_CH);
	vector<aligned_vector<int>> csc_row_ind_fpga(NUM_CH);
	vector<aligned_vector<ap_uint<64>>> dep_graph_ch(NUM_CH);
	vector<aligned_vector<ap_uint<64>>> comp_packet_ch(NUM_CH);
	aligned_vector<int> dep_graph_ptr;
	aligned_vector<int> merge_inst_ptr;
	aligned_vector<int> if_need;

	int maxLenCounter = 0;

	convertCSRToCSC(N, nnz, IA, JA, A, csc_col_ptr, csc_row_ind, csc_val, csc_col_ptr_fpga, csc_row_ind_fpga, K_csc);
	generate_edgelist_spmv_cyclic(N, IA, JA, A, edge_list_ch, edge_list_ptr, if_need);
	process_spmv_ptr(edge_list_ch, edge_list_ptr, edge_list_ch_mod, edge_list_ptr_mod, maxLenCounter);
	generate_dependency_graph_for_pes_cyclic(N, IA, JA, A, dep_graph_ch, dep_graph_ptr);
	merge_ptr(N, dep_graph_ptr, edge_list_ptr_mod, merge_inst_ptr);
	merge_data(N, dep_graph_ptr, edge_list_ptr_mod, dep_graph_ch, edge_list_ch_mod, comp_packet_ch);

	int need_count = 0;
	for(int i = 1; i < if_need.size(); i++){
		if(if_need[i] == 1) need_count++;
	}

	std::clog << "need to read hbm: " << need_count << " times" << std::endl;
	std::clog << "hbm + spmv latency: " << need_count*(WINDOW_SIZE_div_2/16+10) + (if_need.size()-need_count) + maxLenCounter/8 << " cycles" << std::endl;

	for(int i = 0; i < NUM_CH; i++){
		if(edge_list_ch_mod[i].size() == 0){
			for(int j = 0; j < 8; j++){
				ap_uint<64> a = 0;
				edge_list_ch_mod[i].push_back(a);
			}
		}
		if(dep_graph_ch[i].size() == 0){
			for(int j = 0; j < 8; j++){
				ap_uint<64> a = 0;
				dep_graph_ch[i].push_back(a);
			}
		}
	}

	cycle[0] = 0;

	int NUM_ITE = (N%WINDOW_LARGE_SIZE == 0)?N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
	int A_LEN = (comp_packet_ch[0].size() + 7) / 8;

	aligned_vector<int> if_need_fpga;
	aligned_vector<int> block_fwd;

	int start_fwd = 1;
	for(int round = 1; round < NUM_ITE; round++){
		for(int ite = 0; ite < (round-1)*NUM_CH; ite++){
			if_need_fpga.push_back(if_need[start_fwd+ite]);
		}
		start_fwd += (round-1)*NUM_CH;
		int count = 0;
		for(int ite = 0; ite < NUM_CH; ite++){
			if(if_need[start_fwd+ite]==1) count++;
			block_fwd.push_back(if_need[start_fwd+ite]);
		}
		if_need_fpga.push_back(count);
		for(int ite = 0; ite < NUM_CH; ite++){
			if_need_fpga.push_back(if_need[start_fwd+ite]);
		}
		start_fwd += NUM_CH;

	}

	int size = if_need_fpga.size();
	if_need_fpga.insert(if_need_fpga.begin(), size);
	for(int i = 0; i < NUM_CH; i++){
		block_fwd.push_back(0);
	}
	size = block_fwd.size();
	block_fwd.insert(block_fwd.begin(), size);

    int64_t kernel_time_ns = tapa::invoke(TrigSolver, FLAGS_bitstream,
                        tapa::read_only_mmaps<ap_uint<64>, NUM_CH>(comp_packet_ch).reinterpret<ap_uint<512>>(),
						tapa::read_only_mmap<int>(merge_inst_ptr),
						tapa::read_only_mmaps<float, 2>(f_fpga).reinterpret<float_v16>(),
                        tapa::read_write_mmap<float>(x_fpga).reinterpret<float_v16>(),
						tapa::read_only_mmap<int>(if_need),
						tapa::read_only_mmap<int>(block_fwd), N, NUM_ITE, A_LEN
						);
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
	std::clog << "cycle count: " << cycle[0] << std::endl;
	
	int unmatched = 0;

	//triangular solver in cpu
	vector<float> expected_x(N);
	vector<float> round_off_error(N);
	int next = 0;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	for(int i = 0; i < N; i++){
		float test_sum = 0.f;
		float image = f[i];
		if(i == 131072) std::clog << "f: " << f[i] << std::endl;
		float num = (i == 0) ? IA[0] : IA[i] - IA[i-1];
		for(int j = 0; j < num-1; j++){
			image -= x_fpga[JA[next]]*A[next];
			test_sum += x_fpga[JA[next]]*A[next];
			if(i == 131072) {
				std::clog << "col:" << JA[next] << ", t1:" << x_fpga[JA[next]] << ", t2:" << A[next] << std::endl; 
			}
			next++;
		}
		if(i == 131072) std::clog << "row:" << JA[next] << ", val:" << (f[i] - test_sum) * (1/A[next]) << ", cpu: " << image / A[next] << ", diff:" << std::fabs((f[i] - test_sum) * (1/A[next]) - (image / A[next]))<< std::endl;
		expected_x[JA[next]] = image / A[next];
		round_off_error[JA[next]] = std::fabs((f[i] - test_sum) * (1/A[next]) - (image / A[next]));
		next++;
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::clog << "CPU time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

        for (int i = 0; i < N; ++i){
		if(std::fabs((std::fabs(x_fpga[i]-expected_x[i]) - round_off_error[i])/expected_x[i]) > 0.1 
			&& std::fabs((x_fpga[i]-expected_x[i])/expected_x[i]) > 0.1 
			){
			// std::clog << "index: " << i << ", expected: " << expected_x[i] << ", actual: " << x_fpga[i] << ", diff: " << std::fabs(x_fpga[i]-expected_x[i]) << std::endl;
			unmatched++;
		}
        }

        if(unmatched == 0) { // tolerance dependends on number of elements
                std::clog << "PASS!" << std::endl;
        }else{
                std::clog << "Please check each element manually. Usually it's caused by ordering of computation!" << std::endl;
        }
        return 0;
}
