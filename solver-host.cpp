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

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using std::vector;

constexpr int NUM_CH = 10;
constexpr int WINDOW_SIZE = 8192;
constexpr int WINDOW_SIZE_div_2 = 8192;
constexpr int WINDOW_LARGE_SIZE = WINDOW_SIZE*NUM_CH;
int WINDOW_SIZE_SPMV = 32;
int MULT_SIZE = 1;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> csr_edge_list_ch,
			// tapa::mmaps<int, NUM_CH> csr_edge_list_ptr,
			tapa::mmaps<ap_uint<512>, NUM_CH> dep_graph_ch,
			// tapa::mmaps<int, NUM_CH> dep_graph_ptr,
			tapa::mmap<int> merge_inst_ptr,
			tapa::mmap<float_v16> f, 
			tapa::mmap<float_v16> x, 
			tapa::mmap<int> if_need,
			const int N,
			const int NUM_ITE
			// tapa::mmap<int> K_csc
			);

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

void generate_edgelist_spmv(
	int N,
	const aligned_vector<int>& csr_row_ptr,
	const aligned_vector<int>& csr_col_ind,
	const aligned_vector<float>& csr_val,
	vector<aligned_vector<ap_uint<64>>>& edge_list_ch,
	vector<aligned_vector<int>>& edge_list_ptr,
	aligned_vector<int>& if_need
){
	int bound = (N % WINDOW_SIZE == 0) ? N/WINDOW_SIZE:N/WINDOW_SIZE+1;
	for(int i = 0; i < bound; i++){
		vector<aligned_vector<ap_uint<64>>> tmp_edge_list(i*MULT_SIZE+MULT_SIZE);
		for(int j = i*WINDOW_SIZE; j < (i+1)*WINDOW_SIZE && j < N; j++){
			int start = (j == 0)? 0 : csr_row_ptr[j-1];
			int end = csr_row_ptr[j];
			for(int k = start; k < end; k++){
				ap_uint<64> a = 0;
				a(63, 48) = (ap_uint<16>)((j - i*WINDOW_SIZE) & 0xFFFF);
				a(47, 32) = (ap_uint<16>)((csr_col_ind[k]%(WINDOW_SIZE_div_2)) & 0xFFFF);
				a(31, 0) = tapa::bit_cast<ap_uint<32>>(csr_val[k]);
				tmp_edge_list[csr_col_ind[k]/WINDOW_SIZE_div_2].push_back(a);
			}
		}
		
		// std::clog << "pe: " << i << std::endl;
		// int count_non_zero = 0;
		// int total_cycle = 0;
		for(int j = 0; j < (i/NUM_CH)*NUM_CH*MULT_SIZE; j++){
			// if(tmp_edge_list[j].size() != 0) std::clog << tmp_edge_list[j].size() << std::endl;
			int list_size = tmp_edge_list[j].size();
			int total_size = 0;
			vector<bool> used_edge(list_size, false);
			vector<std::set<int>> row_raw(11); // last 11 elements
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
					int row_i = (tmp_edge_list[j][l](63, 48) | (int) 0);
					bool found = false;
					for(int n = 0; n < 11; n++){
						if(row_raw[n].find(row_i) != row_raw[n].end()){
							found = true;
							break;
						}
					}
					if(!used_edge[l] && row.find(row_i%8) == row.end() && !found){
						packet[row_i%8] = tmp_edge_list[j][l];
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
					edge_list_ch[i%NUM_CH].push_back(packet[l]);
				}
				pack_chunk_count++;
				next_slot = (next_slot + 1) % 11;
			}
			// if(total_size != 0) {
			// 	count_non_zero++;
			// 	total_cycle+=(11+total_size/8);
			// }
			edge_list_ptr[i%NUM_CH].push_back(total_size);
		}
		// std::clog << std::max(count_non_zero*WINDOW_SIZE_SPMV, total_cycle) << std::endl;
	}
	// for(int i = 0; i < NUM_CH; i++){
	// 	int size = edge_list_ptr[i].size();
	// 	edge_list_ptr[i].insert(edge_list_ptr[i].begin(), size);
	// }

	if(bound % NUM_CH != 0){
		for(int i = bound % NUM_CH; i < NUM_CH; i++){
			for(int j = 0; j < ((bound - 1)/NUM_CH)*NUM_CH*MULT_SIZE; j++){
				edge_list_ptr[i].push_back(0);
			}
		}
	}

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
				vector<std::set<int>> row_raw(11); // last 11 elements
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
						for(int n = 0; n < 11; n++){
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
					next_slot = (next_slot + 1) % 11;
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

void generate_dependency_graph_for_pes(
	int N,
	const aligned_vector<int>& csr_row_ptr,
	const aligned_vector<int>& csr_col_ind,
	const aligned_vector<float>& csr_val,
	vector<aligned_vector<ap_uint<64>>>& dep_graph_ch,
	aligned_vector<int>& dep_graph_ptr
){
	int bound = (N % WINDOW_LARGE_SIZE == 0) ? N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;
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
					a(61,47) = (ap_uint<15>)((e.row%WINDOW_SIZE) & 0x7FFF);
					a(46,32) = (ap_uint<15>)((e.col%WINDOW_SIZE) & 0x7FFF);
					a(31,0) = tapa::bit_cast<ap_uint<32>>(e.attr);
					if(e.row == e.col){
						a(63,62) = (ap_uint<2>)(1);
						nodes_pe[e.row/WINDOW_SIZE].push_back(a);
					}else{
						a(63,62) = (ap_uint<2>)(0);
						edges_pe[(e.row/WINDOW_SIZE)*NUM_CH+(e.col/WINDOW_SIZE)].push_back(a);
						parents[e.row]--;
						if(parents[e.row] == 0) {
							roots.push(e.row);
						}
					}
				}
				roots.pop();
			}

			//TODO: split edge into blocks
			

			// for(int j = 0; j < nodes.size(); j ++){
			// 	dep_graph_ch[i%NUM_CH].push_back(nodes[j]);
			// }
			// if(nodes.size() % 8 != 0) {
			// 	for(int j = 0; j < 8 - (nodes.size() % 8); j++){
			// 		ap_uint<64> a = 0;
			// 		a(63,62) = (ap_uint<2>)(2);
			// 		a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
			// 		dep_graph_ch[i%NUM_CH].push_back(a);
			// 	}
			// 	node_count++;
			// }

			int maxNode = 0;
			int maxEdge = 0;
			vector<vector<ap_uint<64>>> dep_graph_tmp(NUM_CH);

			//schedule each PEs
			for(int pe_i = 0; pe_i < NUM_CH; pe_i++){
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
					a(63,62) = (ap_uint<2>)(2);
					a(61,47) = 0x7FFF;
					a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
					packet[n] = a;
				}
				for(int n = 0; n < rem_node_num; n++){
					if(!used_node[n]){
						auto nd = nodes[n];
						int row_i = (nd(61,47) | (int) 0);
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
			
			for(int block_id = pe_i; block_id >= 0; block_id--){
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
						a(63,62) = (ap_uint<2>)(2);
						a(61,47) = 0x7FFF;
						packet[n] = a;
					}
					for(int n = 0; n < rem_edge_num; n++){
						if(!used_edge[n]){
							auto e = edge_list[n];
							int row_i = (e(61,47) | (int) 0);
							// int col_i = (e(46,32) | (int) 0);
							bool found = false;
							for(int m = 0; m < 8; m++){
								if(row_raw[m].find(row_i) != row_raw[m].end()){
									found = true;
									break;
								}
							}
							if(row.find(row_i%8) == row.end() 
							&& !found
							// && col.find(col_i) == col.end()
							){
								row.insert(row_i%8);
								row_raw[next_slot].insert(row_i);
								packet[row_i % 8] = e;
								// col.insert(col_i);
								// shift_edge_list.push_back(edge_list[n]);
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
					// if(pack_chunk_count == 18){
					// 	pack_chunk_count = 0;
					// 	row_raw.clear();
					// }
					next_slot = (next_slot+1)%8;
					pushed_edge_count += row.size();
				}
				edge_count_pe[pe_i].push_back(edge_count);
				if(edge_count > maxEdge) maxEdge = edge_count;

			}
			// for(int j = 0; j < shift_edge_list.size(); j ++){
			// 	dep_graph_ch[i%NUM_CH].push_back(shift_edge_list[j]);
			// }
			// node_count += nodes.size() / 8;
			// edge_count += shift_edge_list.size() / 8;
			// std::clog << "node: " << node_count << std::endl;
			// std::clog << "edge: " << edge_count << std::endl;
			// inst.push_back(node_count);
			// inst.push_back(edge_count);

			}

			inst.push_back(maxNode);
			inst.push_back(maxEdge);
			
			//process dep graph ptr
			for(int pe_i = 0; pe_i < NUM_CH; pe_i++){
				// std::clog << "pe: " << pe_i << std::endl;
				int offset = 0;
				int prev_size = dep_graph_ch[pe_i].size();
				for(int b = offset; b < offset + node_count_pe[pe_i]*8; b++){
					dep_graph_ch[pe_i].push_back(dep_graph_tmp[pe_i][b]);
				}
				for(int b = 0; b < maxNode - node_count_pe[pe_i]; b++){
					for(int n = 0; n < 8; n++){
						ap_uint<64> a = 0;
						a(63,62) = (ap_uint<2>)(2);
						a(61,47) = 0x7FFF;
						a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
						dep_graph_ch[pe_i].push_back(a);
					}
				}
				offset += node_count_pe[pe_i]*8;
				for(int b = 0; b <= pe_i; b++){
					for(int l = offset; l < offset + edge_count_pe[pe_i][b]*8; l++){
						dep_graph_ch[pe_i].push_back(dep_graph_tmp[pe_i][l]);
					}
					for(int l = 0; l < maxEdge - edge_count_pe[pe_i][b]; l++){
						for(int n = 0; n < 8; n++){
							ap_uint<64> a = 0;
							a(63,62) = (ap_uint<2>)(2);
							a(61,47) = 0x7FFF;
							dep_graph_ch[pe_i].push_back(a);
						}
					}
					offset += edge_count_pe[pe_i][b]*8;
				}
			}
			layer_count++;

		}
		// assert(inst.size() % 2 == 0);
		// dep_graph_ptr[i%NUM_CH].push_back(inst.size()/2);
		// for(auto num : inst){
		// 	dep_graph_ptr[i%NUM_CH].push_back(num);
		// }

		dep_graph_ptr.push_back(layer_count);
		for(auto num : inst){
			dep_graph_ptr.push_back(num);
		}

	}

	// for(int i = 0; i < NUM_CH; i++){
	// 	int size = dep_graph_ptr[i].size();
	// 	dep_graph_ptr[i].insert(dep_graph_ptr[i].begin(), size);
	// }
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
					a(61,47) = (ap_uint<15>)((e.row/NUM_CH) & 0x7FFF);
					a(46,32) = (ap_uint<15>)((e.col/NUM_CH) & 0x7FFF);
					a(31,0) = tapa::bit_cast<ap_uint<32>>(e.attr);
					if(e.row == e.col){
						a(63,62) = (ap_uint<2>)(1);
						nodes_pe[ch].push_back(a);
					}else{
						a(63,62) = (ap_uint<2>)(0);
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
					a(63,62) = (ap_uint<2>)(2);
					a(61,47) = 0x7FFF;
					a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
					packet[n] = a;
				}
				for(int n = 0; n < rem_node_num; n++){
					if(!used_node[n]){
						auto nd = nodes[n];
						int row_i = (nd(61,47) | (int) 0);
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
						a(63,62) = (ap_uint<2>)(2);
						a(61,47) = 0x7FFF;
						packet[n] = a;
					}
					for(int n = 0; n < rem_edge_num; n++){
						if(!used_edge[n]){
							auto e = edge_list[n];
							int row_i = (e(61,47) | (int) 0);
							// int col_i = (e(46,32) | (int) 0);
							bool found = false;
							for(int m = 0; m < 8; m++){
								if(row_raw[m].find(row_i) != row_raw[m].end()){
									found = true;
									break;
								}
							}
							if(row.find(row_i%8) == row.end() 
							&& !found
							// && col.find(col_i) == col.end()
							){
								row.insert(row_i%8);
								row_raw[next_slot].insert(row_i);
								packet[row_i % 8] = e;
								// col.insert(col_i);
								// shift_edge_list.push_back(edge_list[n]);
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
					// if(pack_chunk_count == 18){
					// 	pack_chunk_count = 0;
					// 	row_raw.clear();
					// }
					next_slot = (next_slot+1)%8;
					pushed_edge_count += row.size();
				}
				edge_count_pe[pe_i].push_back(edge_count);
				if(edge_count > maxEdge) maxEdge = edge_count;
				if(edge_count > max_effect_edge) max_effect_edge = edge_count;
			}
			effect_edge += max_effect_edge;
			// for(int j = 0; j < shift_edge_list.size(); j ++){
			// 	dep_graph_ch[i%NUM_CH].push_back(shift_edge_list[j]);
			// }
			// node_count += nodes.size() / 8;
			// edge_count += shift_edge_list.size() / 8;
			// std::clog << "node: " << node_count << std::endl;
			// std::clog << "edge: " << edge_count << std::endl;
			// inst.push_back(node_count);
			// inst.push_back(edge_count);

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
						a(63,62) = (ap_uint<2>)(2);
						a(61,47) = 0x7FFF;
						a(31,0) = tapa::bit_cast<ap_uint<32>>((float)(1.0));
						node_tmp_cache.push_back(a);
					}
				}
				offset += node_count_pe[pe_i]*8;
				for(int b = 0; b < NUM_CH; b++){
					if(b == pe_i){
						for(int l = 0; l < maxNode*8; l++){
							dep_graph_ch[pe_i].push_back(node_tmp_cache[l]);
						}
					}
					for(int l = offset; l < offset + edge_count_pe[pe_i][b]*8; l++){
						dep_graph_ch[pe_i].push_back(dep_graph_tmp[pe_i][l]);
					}
					for(int l = 0; l < maxEdge - edge_count_pe[pe_i][b]; l++){
						for(int n = 0; n < 8; n++){
							ap_uint<64> a = 0;
							a(63,62) = (ap_uint<2>)(2);
							a(61,47) = 0x7FFF;
							dep_graph_ch[pe_i].push_back(a);
						}
					}
					offset += edge_count_pe[pe_i][b]*8;
				}
			}
			layer_count++;

		}
		// assert(inst.size() % 2 == 0);
		// dep_graph_ptr[i%NUM_CH].push_back(inst.size()/2);
		// for(auto num : inst){
		// 	dep_graph_ptr[i%NUM_CH].push_back(num);
		// }

		dep_graph_ptr.push_back(layer_count);
		for(auto num : inst){
			dep_graph_ptr.push_back(num);
		}

	}

	LOG(INFO) << "total count: " << total_iter_count;
	LOG(INFO) << "total effective count: " << total_effect_iter_count;

	// for(int i = 0; i < NUM_CH; i++){
	// 	int size = dep_graph_ptr[i].size();
	// 	dep_graph_ptr[i].insert(dep_graph_ptr[i].begin(), size);
	// }
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
	aligned_vector<float> f_fpga;
	aligned_vector<float> x_fpga(((N+159)/160)*160, 0.0);
	aligned_vector<int> K_fpga;

	// for(int i = 0; i < NUM_CH; i++){
	// 	for(int j = 0; j < N/NUM_CH; j++){
	// 		x_fpga[i].push_back(0.0);
	// 	}
	// }

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

	// reordering
	int bound = (f.size() % WINDOW_LARGE_SIZE == 0) ? f.size() /WINDOW_LARGE_SIZE : (f.size()/WINDOW_LARGE_SIZE + 1);
	vector<aligned_vector<float>> f_ch(NUM_CH);
	for(int i = 0; i < bound; i++){
		for(int j = i*WINDOW_LARGE_SIZE; j < (i+1)*WINDOW_LARGE_SIZE && j < f.size(); j++){
			f_ch[(j-i*WINDOW_LARGE_SIZE)%NUM_CH].push_back(f[j]);
		}
		// make sure every ch has multi of 16
		for(int j = 0; j < NUM_CH; j++){
			int size = f_ch[j].size();
			for(int k = 0; k < size % 16; k++){
				f_ch[j].push_back(0);
			}
		}
	}
	
	int max = 0;
	for(int j = 0; j < NUM_CH; j++){
		if(max < f_ch[j].size()) max = f_ch[j].size();
	}
	for(int j = 0; j < NUM_CH; j++){
		int size = f_ch[j].size();
		for(int k = 0; k < max - size; k++){
			f_ch[j].push_back(0);
		}
	}

	for(int j = 0; j < f_ch[0].size()/16; j++){
		for(int k = 0; k < NUM_CH; k++){
			for(int l = 0; l < 16; l++){
				f_fpga.push_back(f_ch[k][j*16+l]);
			}
		}
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
	vector<aligned_vector<ap_uint<64>>> dep_graph_ch(NUM_CH);
	aligned_vector<int> dep_graph_ptr;
	aligned_vector<int> merge_inst_ptr;
	aligned_vector<int> if_need;

	int maxLenCounter = 0;

	convertCSRToCSC(N, nnz, IA, JA, A, csc_col_ptr, csc_row_ind, csc_val, csc_col_ptr_fpga, csc_row_ind_fpga, K_csc);
	generate_edgelist_spmv_cyclic(N, IA, JA, A, edge_list_ch, edge_list_ptr, if_need);
	process_spmv_ptr(edge_list_ch, edge_list_ptr, edge_list_ch_mod, edge_list_ptr_mod, maxLenCounter);
	generate_dependency_graph_for_pes_cyclic(N, IA, JA, A, dep_graph_ch, dep_graph_ptr);
	merge_ptr(N, dep_graph_ptr, edge_list_ptr_mod, merge_inst_ptr);

	int need_count = 0;
	for(int i = 1; i < if_need.size(); i++){
		if(if_need[i] == 1) need_count++;
	}

	std::clog << "need to read hbm: " << need_count << " times" << std::endl;
	std::clog << "hbm + spmv latency: " << need_count*(WINDOW_SIZE_div_2/16+10) + (if_need.size()-need_count) + maxLenCounter/8 << " cycles" << std::endl;

	// std::clog << K_csc[156] << std::endl;
	// std::clog << edge_list_ptr[1].size() << std::endl;
	// std::clog << csc_row_ind[2] << std::endl;

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

	std::clog << dep_graph_ch[9].size() << std::endl;

	cycle[0] = 0;

	int NUM_ITE = (N%WINDOW_LARGE_SIZE == 0)?N/WINDOW_LARGE_SIZE:N/WINDOW_LARGE_SIZE+1;

    int64_t kernel_time_ns = tapa::invoke(TrigSolver, FLAGS_bitstream,
                        tapa::read_only_mmaps<ap_uint<64>, NUM_CH>(edge_list_ch_mod).reinterpret<ap_uint<512>>(),
						// tapa::read_only_mmaps<int, NUM_CH>(edge_list_ptr),
						tapa::read_only_mmaps<ap_uint<64>, NUM_CH>(dep_graph_ch).reinterpret<ap_uint<512>>(),
						// tapa::read_only_mmaps<int, NUM_CH>(dep_graph_ptr),
						// tapa::read_only_mmaps<int, NUM_CH>(csc_col_ptr_fpga),
						// tapa::read_only_mmaps<int, NUM_CH>(csc_row_ind_fpga),
						tapa::read_only_mmap<int>(merge_inst_ptr),
						tapa::read_only_mmap<float>(f_fpga).reinterpret<float_v16>(),
                        tapa::read_write_mmap<float>(x_fpga).reinterpret<float_v16>(),
						tapa::read_only_mmap<int>(if_need), N, NUM_ITE
						// tapa::read_only_mmap<int>(K_csc)
						);
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
	std::clog << "cycle count: " << cycle[0] << std::endl;


	// // reorder
	// for(int i = 0; i < x_fpga.size()/(16*NUM_CH); i++){
	// 	vector<vector<float>> cache(NUM_CH);
	// 	for(int j = i*(16*NUM_CH); j < (i+1)*(16*NUM_CH); j++){
	// 		cache[(j-i*(16*NUM_CH))/16].push_back(x_fpga[j]);
	// 	}
	// 	for(int j = 0; j < NUM_CH; j++){
	// 		for(int k = 0; k < 16; k++){
	// 			x_fpga[i*(16*NUM_CH)+k*NUM_CH+j] = cache[j][k];
	// 		}
	// 	}
	// }
	
	int unmatched = 0;

	//triangular solver in cpu
	vector<float> expected_x(N);
	int next = 0;
	float test_sum = 0.f;
	for(int i = 0; i < N; i++){
		float image = f[i];
		if(i == 59816) std::clog << "f: " << f[i] << std::endl;
		float num = (i == 0) ? IA[0] : IA[i] - IA[i-1];
		for(int j = 0; j < num-1; j++){
			image -= x_fpga[JA[next]]*A[next];
			if(i == 59816) {
				std::clog << "col:" << JA[next] << ", t1:" << x_fpga[JA[next]] << ", t2:" << A[next] << std::endl; 
				test_sum += x_fpga[JA[next]]*A[next];
			}
			next++;
		}
		if(i == 59816) std::clog << "row:" << JA[next] << ", val:" << (f[i] - test_sum) * (1/A[next]) << ", cpu: " << image / A[next] << ", diff:" << std::fabs((f[i] - test_sum) * (1/A[next]) - (image / A[next]))<< std::endl;
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

        for (int i = 0; i < N; ++i){
		if(std::fabs((x_fpga[i]-expected_x[i])/expected_x[i]) > 0.01){
			std::clog << "index: " << i << ", expected: " << expected_x[i] << ", actual: " << x_fpga[i] << ", diff: " << std::fabs(x_fpga[i]-expected_x[i]) << std::endl;
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
