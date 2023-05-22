#include <ap_int.h>
#include <cstdint>
#include <tapa.h>
#include <iomanip>
//#include <glog/logging.h>

constexpr int WINDOW_SIZE = 8192;
constexpr int WINDOW_SIZE_div_8 = 1024;
constexpr int WINDOW_SIZE_div_2 = 512;
constexpr int WINDOW_SIZE_div_16 = 512;
constexpr int WINDOW_SIZE_div_32 = 32;
constexpr int NUM_CH = 10;
constexpr int MULT_SIZE = 16;
constexpr int FIFO_DEPTH = 2;
constexpr int WINDOW_LARGE_SIZE = WINDOW_SIZE * NUM_CH;

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;

struct MultXVec {
	tapa::vec_t<ap_uint<16>, 8> row;
	float_v8 axv;
};

struct MultDVec {
	tapa::vec_t<ap_uint<15>, 8> row;
	float_v8 axv;
};

template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1
        data_t tmp; q.try_read(tmp);
    }
}

void black_hole_int(tapa::istream<int> & fifo_in) {
    bh(fifo_in);
}

void black_hole_float(tapa::istream<float>& fifo_in) {
	bh(fifo_in);
}

void black_hole_float_vec(tapa::istream<float_v16>& fifo_in) {
	bh(fifo_in);
}

void black_hole_ap_uint(tapa::istream<ap_uint<512>>& fifo_in){
	bh(fifo_in);
}

void black_hole_dvec(tapa::istream<MultDVec>& fifo_in){
	bh(fifo_in);
}

void read_int_vec(int pe_i, int total_N, tapa::async_mmap<int>& mmap, tapa::istream<int>& len_in, tapa::ostream<int>& len_out,
                 tapa::ostream<int>& stream) {
	int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
	int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
	int offset = 0;
	for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off
		const int N = len_in.read();
		len_out.write(N);
		for (int i_req = 0, i_resp = 0; i_resp < N;) {
				if(i_req < N && !mmap.read_addr.full()){
						mmap.read_addr.try_write(i_req+offset);
						++i_req;
				}
				if(!mmap.read_data.empty() && !stream.full()){
						int tmp;
						mmap.read_data.try_read(tmp);
						stream.try_write(tmp);
						++i_resp;
				}
		}
		offset += N;
	}
}

void write_x(tapa::istream<float_v16>& x, tapa::ostream<bool>& fin_write, tapa::async_mmap<float_v16>& mmap, int total_N){
	int num_block = total_N / WINDOW_SIZE_div_2;
	for(int i = 0; i < num_block; i++){
	    // LOG(INFO) << "process level " << i << " ...";
		int start = WINDOW_SIZE_div_32*i;
		int N = WINDOW_SIZE_div_32;
		for(int i_req = 0, i_resp = 0; i_resp < N;){
			if((i_req < N) & !x.empty() & !mmap.write_addr.full() & !mmap.write_data.full()){
				mmap.write_addr.try_write(i_req + start);
				mmap.write_data.try_write(x.read(nullptr));
				++i_req;
			}
			if(!mmap.write_resp.empty()){
				i_resp += unsigned(mmap.write_resp.read(nullptr))+1;
				//LOG(INFO) << i_resp;
			}
		}
		fin_write.write(true);
	}

	//residue
	int residue = ((total_N % WINDOW_SIZE_div_2) + 15)/16;
	for(int i_req = 0, i_resp = 0; i_resp < residue;){
		if(i_req < residue && !x.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
			mmap.write_addr.try_write(i_req + num_block * WINDOW_SIZE_div_32);
			mmap.write_data.try_write(x.read(nullptr));
			++i_req;
		}
		if(!mmap.write_resp.empty()){
			i_resp += unsigned(mmap.write_resp.read(nullptr))+1;
			//LOG(INFO) << i_resp;
		}
	}

	fin_write.write(false);
}

void read_float_vec(int pe_i, int total_N, tapa::async_mmap<float_v16>& mmap, tapa::istream<int>& len_in, tapa::ostream<int>& len_out,
                 tapa::ostream<float_v16>& stream) {
	int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
	int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
	int offset = 0;
	for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off
		const int N = len_in.read();
		len_out.write(N);
		const int num_ite = (N + 15) / 16;
		for (int i_req = 0, i_resp = 0; i_resp < num_ite;) {
				if(i_req < num_ite && !mmap.read_addr.full()){
						mmap.read_addr.try_write(i_req+offset);
						++i_req;
				}
				if(!mmap.read_data.empty() && !stream.full()){
						float_v16 tmp;
						mmap.read_data.try_read(tmp);
						stream.try_write(tmp);
						++i_resp;
				}
		}
		offset+=num_ite;
	}
}

void read_f(int N, tapa::async_mmap<float_v16>& mmap, tapa::ostreams<float_v16, NUM_CH>& f_fifo_out){
	const int num_ite = (N + 15) / 16;
	for(int i_req = 0, i_resp = 0, c_idx = 0; i_resp < num_ite;){
		if((i_req < num_ite) & !mmap.read_addr.full()){
				mmap.read_addr.try_write(i_req);
				++i_req;
		}
		if(!mmap.read_data.empty()){
				float_v16 tmp;
				mmap.read_data.try_read(tmp);
				f_fifo_out[c_idx].write(tmp);
				++i_resp;
				if(i_resp % WINDOW_SIZE_div_16 == 0) c_idx++;
				if(c_idx == NUM_CH) c_idx = 0;
		}
	}
}

//TODO: replace with serpen after modifying the read width
void PEG_Xvec( int pe_i, int total_N,
	tapa::istream<int>& fifo_inst_in, tapa::ostream<int>& fifo_inst_out,
	tapa::istream<ap_uint<512>>& spmv_A, // 16-bit row + 16-bit col + 32-bit float
	tapa::ostream<MultXVec>& fifo_aXVec,
	tapa::istream<int>& fifo_need_in,
	tapa::ostream<int>& fifo_need_out,
	tapa::istream<float_v16>& req_x_in,
	tapa::ostream<float_v16>& req_x_out){

		int bound = (total_N%WINDOW_LARGE_SIZE == 0)?total_N/WINDOW_LARGE_SIZE:total_N/WINDOW_LARGE_SIZE+1;
		// int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

round:
		for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off

load_x:
			for(int ite = 0; ite < (round*NUM_CH)*MULT_SIZE; ite++){

				float local_x[4][WINDOW_SIZE_div_2];
#pragma HLS bind_storage variable=local_x latency=2
#pragma HLS array_partition variable=local_x complete dim=1
#pragma HLS array_partition variable=local_x cyclic factor=8 dim=2

				const int N = fifo_inst_in.read();
				const int need = fifo_need_in.read();
				fifo_inst_out.write(N);
				fifo_need_out.write(need);
				
				const int num_ite = (N + 7) / 8;
				//read x
				if(need == 1){
					for(int i = 0; i < WINDOW_SIZE_div_32;){
						#pragma HLS pipeline II=1
						if(!req_x_in.empty()){
							float_v16 x_val; req_x_in.try_read(x_val);
							for(int l = 0; l < 16; l++){
								for(int k = 0; k < 4; k++){
									#pragma HLS unroll
									local_x[k][i*16+l] = x_val[l];
								}
							}
							req_x_out.write(x_val);
							i++;
						}
					}
				}

				//read A and compute
compute:
				for(int i = 0; i < num_ite;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=128
					if(!spmv_A.empty()){
						ap_uint<512> spmv_block; spmv_A.try_read(spmv_block);
						MultXVec raxv;

						for(int k = 0; k < 8; k++){
							ap_uint<64> a = spmv_block(64*k+63, 64*k);
							ap_uint<16> a_row = a(63, 48);
							ap_uint<16> a_col = a(47, 32);
							ap_uint<32> a_val = a(31, 0);

							raxv.row[k] = a_row;
							if(a_row[15] == 0){
								float a_val_f = tapa::bit_cast<float>(a_val);
								raxv.axv[k] = local_x[k/2][a_col] * a_val_f;
								// if(a_row == 2102) LOG(INFO) << std::fixed << std::setprecision(6) << "col: " << a_col << ", t1: " << local_x[k%4][a_col] << ", t2: " << a_val_f; 
							}
						}
						fifo_aXVec.write(raxv);
						i++;
					}
				}
			}
		}
		
	}

void PEG_YVec(int pe_i, int total_N,
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<MultXVec>& fifo_aXVec,
	tapa::ostream<float_v16>& fifo_y_out,
	tapa::istream<int>& N_in, tapa::ostream<int>& N_out){
	int bound = (total_N%WINDOW_LARGE_SIZE == 0)?total_N/WINDOW_LARGE_SIZE:total_N/WINDOW_LARGE_SIZE+1;
	// int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

round:
		for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off
			const int num_x = N_in.read();
			const int num_ite_x = (num_x + 15) / 16;
			N_out.write(num_x);

			float local_c[8][WINDOW_SIZE_div_8];
#pragma HLS bind_storage variable=local_c type=RAM_2P impl=URAM
#pragma HLS array_partition variable=local_c complete dim=1

reset:
			for(int i = 0; i < WINDOW_SIZE_div_8; i++){
				for(int j = 0; j < 8; j++){
					local_c[j][i] = 0;
				}
			}

load_axv:
			for(int ite = 0; ite < (round*NUM_CH)*MULT_SIZE; ite++){
				const int N = fifo_inst_in.read();
				const int num_ite = (N + 7) / 8;

acc:
				for(int i = 0; i < num_ite;){
					#pragma HLS pipeline II=1
					#pragma HLS dependence variable=local_c distance=11 true

					if(!fifo_aXVec.empty()){
						MultXVec ravx; fifo_aXVec.try_read(ravx);
						for(int k = 0; k < 8; k++){
							#pragma HLS unroll
							auto a_row = ravx.row[k];
							if(a_row[15] == 0){
								local_c[k][a_row >> 3] += ravx.axv[k];
							}
						}
						i++;
					}
				}
			}

write_y:
			for(int i = 0; i < num_ite_x; i++){
#pragma HLS pipeline II=1

				float_v16 tmp;
				for(int j = 0; j < 16; j++){
					#pragma HLS unroll

					tmp[j] = local_c[j%8][(i << 1)+(j>>3)];
				}
				fifo_y_out.write(tmp);
			}
			
		}
	}


void Yvec_minus_f(tapa::istream<float_v16>& f,
		tapa::istream<float_v16>& spmv_in, 
		tapa::ostream<float_v16>& update_f,
		tapa::istream<int>& N_in, tapa::ostream<int>& N_out){

for(;;){
#pragma HLS loop_flatten off
			const int N = N_in.read();
			N_out.write(N);
			const int num_ite = (N+15)/16;

			float local_y[WINDOW_SIZE];

#pragma HLS array_partition cyclic variable=local_y factor=16

read_f:
			for(int i = 0; i < num_ite;){
			#pragma HLS pipeline II=1
				if(!f.empty()){
					float_v16 tmp_f; f.try_read(tmp_f);
					for(int j = 0; j < 16; j++){
						#pragma HLS unroll
						local_y[(i << 4)+j] = tmp_f[j];
					}
					i++;
				}
			}

read_spmv_and_subtract:
			for(int i = 0; i < num_ite;){
			#pragma HLS loop_tripcount min=1 max=64
			#pragma HLS dependence variable=local_y false
			#pragma HLS pipeline II=1
				if(!spmv_in.empty()){
					float_v16 tmp_spmv; spmv_in.try_read(tmp_spmv);
					float_v16 new_f;
					for(int j = 0; j < 16; j++){
						#pragma HLS unroll
						new_f[j] = local_y[(i << 4) + j] - tmp_spmv[j];
					}
					update_f.write(new_f);
					i++;
				}
			}
		}
	}

void solve_node(
	int pe_i,
	tapa::istream<ap_uint<512>>& dep_graph_ch,
	tapa::istream<int>& dep_graph_ptr,
	tapa::istream<float_v16>& y_in,
	tapa::ostream<MultDVec>& fifo_node_to_edge,
	tapa::istream<MultDVec>& fifo_edge_to_node,
	tapa::istream<int>& N_in,
	tapa::ostream<int>& N_out){
		
for(;;){
#pragma HLS loop_flatten off
	const int N = N_in.read();
	N_out.write(N);
	const int N_layer = dep_graph_ptr.read();
	const int num_ite = (N + 15) / 16;

	float local_y[8][WINDOW_SIZE_div_8];

#pragma HLS array_partition complete variable=local_y dim=1

read_y:
	for(int i = 0; i < num_ite;){
#pragma HLS pipeline II=1
		if(!y_in.empty()){
			float_v16 tmp_f; y_in.try_read(tmp_f);
			for(int j = 0; j < 16; j++){
				#pragma HLS unroll
				local_y[j%8][(i << 1) + (j >> 3)] = tmp_f[j];
			}
			i++;
		}
	}

compute:
    for(int i = 0; i < N_layer; i++){
		const int N_node = dep_graph_ptr.read();
		const int N_edge = dep_graph_ptr.read();

compute_node:
		for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1
			#pragma HLS dependence variable=local_y false
			if(!dep_graph_ch.empty()){
				ap_uint<512> dep_block; dep_graph_ch.try_read(dep_block);
				MultDVec dvec;

				for(int k = 0; k < 8; k++){
					#pragma HLS unroll
					ap_uint<64> a = dep_block(64*k + 63, 64*k);
					ap_uint<2> opcode = a(63,62);
					ap_uint<15> row = a(61,47);
					ap_uint<32> val = a(31,0);
					float val_f = tapa::bit_cast<float>(val);

					dvec.row[k] = row;
					if((opcode | (int) 0) == 1){
						float ans = local_y[k][(row >> 3)] * val_f;
						dvec.axv[k] = ans;
						// if(pe_i == 0 && row == 3) LOG(INFO) << "row: " << row << ", t1: " << local_y[k][(row >> 3)] << ", t2: " << val_f; 
					}
				}
				fifo_node_to_edge.write(dvec);
				j++;
			}
		}

load_edge:
		for(int j = 0; j < N_edge;){
			#pragma HLS pipeline II=1
			#pragma HLS dependence variable=local_y distance=8 true
			if(!fifo_edge_to_node.empty()){
				MultDVec edge_block; fifo_edge_to_node.try_read(edge_block);

				for(int k = 0; k < 8; k++){
					#pragma HLS unroll
					auto row = edge_block.row[k];
					auto val = edge_block.axv[k];
					if(row[14] == 0){
						local_y[k][(row >> 3)] -= val;
						// if(row == 3) LOG(INFO) << "after reduction: " << local_y[k][(row >> 3)];
					}
				}
				j++;
			}
		}

	}

}


	}

void solve_edge(
		int pe_i,
		tapa::istream<ap_uint<512>>& dep_graph_ch,
		tapa::istream<int>& dep_graph_ptr,
		tapa::istream<MultDVec>& fifo_node_to_edge,
		tapa::ostream<MultDVec>& fifo_edge_to_node,
		tapa::istream<MultDVec>& fifo_prev_pe,
		tapa::ostream<MultDVec>& fifo_next_pe,
		tapa::ostream<float_v16>& x_out, 
		tapa::istream<int>& N_in,
		tapa::ostream<int>& N_out){

for(;;){
#pragma HLS loop_flatten off

	if(pe_i == 0 && !fifo_prev_pe.empty()){
		fifo_prev_pe.read(nullptr);
	}

	const int N = N_in.read();
	N_out.write(N);
	const int N_layer = dep_graph_ptr.read();
	const int num_ite = (N + 15) / 16;

	float local_x[8][WINDOW_SIZE_div_8];

#pragma HLS bind_storage variable=local_x type=RAM_2P impl=URAM
#pragma HLS array_partition complete variable=local_x dim=1

	float local_x_tmp[4][8][WINDOW_SIZE_div_8];
#pragma HLS bind_storage variable=local_x_tmp latency=2
#pragma HLS array_partition complete variable=local_x_tmp dim=1
#pragma HLS array_partition complete variable=local_x_tmp dim=2
// #pragma HLS array_partition cyclic variable=local_x_tmp factor=2 dim=3	

compute:
    for(int i = 0; i < N_layer; i++){

		for(int level = pe_i; level >= 0; level--){
			const int N_node = dep_graph_ptr.read();
			const int N_edge = dep_graph_ptr.read();

	load_node:
			if(level == pe_i){
				for(int j = 0; j < N_node;){
					#pragma HLS pipeline II=1
					#pragma HLS dependence variable=local_x false
					#pragma HLS dependence variable=local_x_tmp false
					if(!fifo_node_to_edge.empty()){
						MultDVec node_block; fifo_node_to_edge.try_read(node_block);
						fifo_next_pe.write(node_block);

						for(int k = 0; k < 8; k++){
							#pragma HLS unroll
							auto row = node_block.row[k];
							auto val = node_block.axv[k];
							if(row[14] == 0){
								for(int l = 0; l < 4; l++){
									local_x_tmp[l][k][(row >> 3)] = val;
								}
								local_x[k][row>>3] = val;
							}
						}
						j++;
					}
				}
			} else {
				for(int j = 0; j < N_node;){
					#pragma HLS pipeline II=1
					#pragma HLS dependence variable=local_x_tmp false
					if(!fifo_prev_pe.empty()){
						MultDVec node_block; fifo_prev_pe.try_read(node_block);
						fifo_next_pe.write(node_block);

						for(int k = 0; k < 8; k++){
							#pragma HLS unroll
							auto row = node_block.row[k];
							auto val = node_block.axv[k];
							if(row[14] == 0){
								for(int l = 0; l < 4; l++){
									local_x_tmp[l][k][(row >> 3)] = val;
								}
							}
						}
						j++;
					}
				}
			}

	compute_edge:
			for(int j = 0; j < N_edge;){
				#pragma HLS pipeline II=1

				if(!dep_graph_ch.empty()){
					ap_uint<512> dep_block; dep_graph_ch.try_read(dep_block);
					MultDVec dvec;
					for(int k = 0; k < 8; k++){
						#pragma HLS unroll
						ap_uint<64> a = dep_block(64*k + 63, 64*k);
						ap_uint<2> opcode = a(63,62);
						ap_uint<15> row = a(61,47);
						ap_uint<15> col = a(46,32);
						ap_uint<32> val = a(31,0);

						dvec.row[k] = row;
						if((opcode | (int) 0) == 0){
							float val_f = tapa::bit_cast<float>(val);
							dvec.axv[k] = (local_x_tmp[k/2][col%8][(col >> 3)] * val_f);
							// if(pe_i == 0 && row == 3) LOG(INFO) << "row: " << row << ", col:" << col << ", t1: " << local_x_tmp[k/2][col%8][(col >> 3)] << ", t2: " << val_f; 
						}
					}
					fifo_edge_to_node.write(dvec);
					j++;
				}
			}
		}
	}

write_x:
	for(int i = 0; i < num_ite; i++){
#pragma HLS loop_tripcount min=1 max=64
#pragma HLS pipeline II=1
		float_v16 tmp_x;
		for(int j = 0; j < 16; j++){
		#pragma HLS unroll
			tmp_x[j] = local_x[j%8][(i << 1)+(j >> 3)];
		}
		x_out.write(tmp_x);
	}


}

		}

void solve(tapa::istream<ap_uint<512>>& dep_graph_ch,
		tapa::istream<int>& dep_graph_ptr,
		tapa::istream<float_v16>& y_in,
		tapa::ostream<float_v16>& x_out, 
		tapa::istream<int>& N_in,
		tapa::ostream<int>& N_out){

for(;;){
#pragma HLS loop_flatten off
	const int N = N_in.read();
	N_out.write(N);
	const int N_layer = dep_graph_ptr.read();
	const int num_ite = (N + 15) / 16;

	float local_x[8][8][WINDOW_SIZE_div_8];
	float local_y[8][WINDOW_SIZE_div_8];

#pragma HLS array_partition complete variable=local_x dim=1
#pragma HLS array_partition complete variable=local_x dim=2
#pragma HLS array_partition complete variable=local_y dim=1

read_y:
	for(int i = 0; i < num_ite;){
#pragma HLS pipeline II=1
		if(!y_in.empty()){
			float_v16 tmp_f; y_in.try_read(tmp_f);
			for(int j = 0; j < 16; j++){
				#pragma HLS unroll
				local_y[j%8][(i << 1) + (j >> 3)] = tmp_f[j];
			}
			i++;
		}
	}

compute:
    for(int i = 0; i < N_layer; i++){
		const int N_node = dep_graph_ptr.read();
		const int N_edge = dep_graph_ptr.read();

compute_node:
		for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1
			#pragma HLS dependence variable=local_x false
			#pragma HLS dependence variable=local_y false
			if(!dep_graph_ch.empty()){
				ap_uint<512> dep_block; dep_graph_ch.try_read(dep_block);

				for(int k = 0; k < 8; k++){
					#pragma HLS unroll
					ap_uint<64> a = dep_block(64*k + 63, 64*k);
					ap_uint<2> opcode = a(63,62);
					ap_uint<15> row = a(61,47);
					ap_uint<32> val = a(31,0);
					float val_f = tapa::bit_cast<float>(val);
					if((opcode | (int) 0) == 1){
						float ans = local_y[k][(row >> 3)] * val_f;
						for(int l = 0; l < 8; l++){
							local_x[l][k][(row >> 3)] = ans;
						}
					}
				}
				j++;
			}
		}

compute_edge:
		for(int j = 0; j < N_edge;){
			#pragma HLS pipeline II=1
			#pragma HLS dependence true variable=local_y distance=6
			if(!dep_graph_ch.empty()){
				ap_uint<512> dep_block; dep_graph_ch.try_read(dep_block);

				for(int k = 0; k < 8; k++){
					#pragma HLS unroll
					ap_uint<64> a = dep_block(64*k + 63, 64*k);
					ap_uint<2> opcode = a(63,62);
					ap_uint<15> row = a(61,47);
					ap_uint<15> col = a(46,32);
					ap_uint<32> val = a(31,0);
					float val_f = tapa::bit_cast<float>(val);
					if((opcode | (int) 0) == 0){
						local_y[k][(row >> 3)] -= (local_x[k][col%8][(col >> 3)] * val_f);
					}
				}
				j++;
			}
		}
	}


write_x:
	for(int i = 0; i < num_ite; i++){
#pragma HLS loop_tripcount min=1 max=64
#pragma HLS pipeline II=1
		float_v16 tmp_x;
		for(int j = 0; j < 16; j++){
		#pragma HLS unroll
			tmp_x[j] = local_x[j/2][j%8][(i << 1)+(j >> 3)];
		}
		x_out.write(tmp_x);
	}
}
}

void read_A_and_split(int pe_i, int total_N,
	tapa::async_mmap<ap_uint<512>>& csr_edge_list_ch,
	tapa::istream<int>& csr_edge_list_ptr,
	tapa::ostream<int>& csr_edge_list_out,
	tapa::ostream<ap_uint<512>>& spmv_val,
	tapa::ostream<int>& spmv_inst){
		int offset = 0;
		int bound = (total_N%WINDOW_LARGE_SIZE == 0)?total_N/WINDOW_LARGE_SIZE:total_N/WINDOW_LARGE_SIZE+1;
		// int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
round:
		for(int round = 0;round < bound;round++){
#pragma HLS loop_flatten off
			const int block_N = csr_edge_list_ptr.read();
			csr_edge_list_out.write(block_N);

traverse_block:
			for(int i = 0; i < block_N; i++){
				#pragma HLS pipeline II=1
				const int N = csr_edge_list_ptr.read();
				const int num_ite = (N + 7) / 8;
				csr_edge_list_out.write(N);
				spmv_inst.write(N);
split:
				for (int i_req = 0, i_resp = 0; i_resp < num_ite;) {
					if(i_req < num_ite && !csr_edge_list_ch.read_addr.full()){
							csr_edge_list_ch.read_addr.try_write(i_req+offset);
							++i_req;
					}
					if(!csr_edge_list_ch.read_data.empty()){
						ap_uint<512> tmp;
						csr_edge_list_ch.read_data.try_read(tmp);
						spmv_val.write(tmp);
						++i_resp;
					}
				}
				offset+=num_ite;
			}
		}
	}

void read_dep_graph(int pe_i, int total_N,
	tapa::async_mmap<ap_uint<512>>& dep_graph_ch,
	tapa::istream<int>& dep_graph_ptr,
	tapa::ostream<int>& dep_graph_ptr_out,
	tapa::ostream<ap_uint<512>>& dep_graph_node,
	tapa::ostream<ap_uint<512>>& dep_graph_edge,
	tapa::ostream<int>& dep_graph_inst_node,
	tapa::ostream<int>& dep_graph_inst_edge){
		int offset = 0;
		int bound = (total_N%WINDOW_LARGE_SIZE == 0)?total_N/WINDOW_LARGE_SIZE:total_N/WINDOW_LARGE_SIZE+1;
		// int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
round:
		for(int round = 0;round < bound;round++){
#pragma HLS loop_flatten off
			const int N_layer = dep_graph_ptr.read();
			dep_graph_ptr_out.write(N_layer);
			dep_graph_inst_node.write(N_layer);
			dep_graph_inst_edge.write(N_layer);

layer:
			for(int i = 0; i < N_layer; i++){
				#pragma HLS pipeline II=1 style=frp
				const int N_node = dep_graph_ptr.read();
				const int N_edge = dep_graph_ptr.read();
				const int N_edge_total = N_edge * (pe_i + 1);

				dep_graph_ptr_out.write(N_node);
				dep_graph_ptr_out.write(N_edge);

				dep_graph_inst_node.write(N_node);
				dep_graph_inst_node.write(N_edge_total);
				dep_graph_inst_edge.write(N_node);
				dep_graph_inst_edge.write(N_edge);
				
split:
				for (int i_req = 0, i_resp = 0; i_resp < N_node;) {
					if((i_req < N_node) & !dep_graph_ch.read_addr.full()){
							dep_graph_ch.read_addr.try_write(i_req+offset);
							++i_req;
					}
					if(!dep_graph_ch.read_data.empty()){
						ap_uint<512> tmp;
						dep_graph_ch.read_data.try_read(tmp);
						dep_graph_node.write(tmp);
						++i_resp;
					}
				}
				offset+=N_node;

				for (int i_req = 0, i_resp = 0; i_resp < N_edge;) {
					if((i_req < N_edge) & !dep_graph_ch.read_addr.full()){
						dep_graph_ch.read_addr.try_write(i_req+offset);
						++i_req;
					}
					if(!dep_graph_ch.read_data.empty()){
						ap_uint<512> tmp;
						dep_graph_ch.read_data.try_read(tmp);
						dep_graph_edge.write(tmp);
						++i_resp;
					}
				}
				offset+=N_edge;

				for(int edge_sec = 0; edge_sec < pe_i; edge_sec++){

					dep_graph_inst_edge.write(N_node);
					dep_graph_inst_edge.write(N_edge);
					for (int i_req = 0, i_resp = 0; i_resp < N_edge;) {
						if((i_req < N_edge) & !dep_graph_ch.read_addr.full()){
								dep_graph_ch.read_addr.try_write(i_req+offset);
								++i_req;
						}
						if(!dep_graph_ch.read_data.empty()){
							ap_uint<512> tmp;
							dep_graph_ch.read_data.try_read(tmp);
							dep_graph_edge.write(tmp);
							++i_resp;
						}
					}
					offset+=N_edge;
				}
			}
		}
	}

void split_ptr_inst(int total_N,
	tapa::istream<int>& merge_inst_q,
	tapa::ostream<int>& dep_graph_ptr_q,
	tapa::ostream<int>& csr_edge_list_ptr_q){
		int bound = (total_N%WINDOW_LARGE_SIZE == 0)?total_N/WINDOW_LARGE_SIZE:total_N/WINDOW_LARGE_SIZE+1;
		// int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

		for(int round = 0; round < bound; round++){
			const int spmv_N = merge_inst_q.read();
			const int dep_graph_N = merge_inst_q.read();
			csr_edge_list_ptr_q.write(spmv_N);
			dep_graph_ptr_q.write(dep_graph_N);

			const int dep_graph_ite = dep_graph_N * 2;

			for(int i = 0; i < spmv_N;){
				if(!merge_inst_q.empty()){
					int tmp; merge_inst_q.try_read(tmp);
					csr_edge_list_ptr_q.write(tmp);
					i++;
				}
			}
			for(int i = 0; i < dep_graph_ite;){
				if(!merge_inst_q.empty()){
					int tmp; merge_inst_q.try_read(tmp);
					dep_graph_ptr_q.write(tmp);
					i++;
				}
			}
		}
	}

void read_all_ptr(
	tapa::async_mmap<int>& merge_inst_ptr,
	tapa::ostream<int>& merge_inst_q){
		int N = 0;
		for (int i_req = 0, i_resp = 0; i_resp < 1;) {
			if((i_req < 1) & !merge_inst_ptr.read_addr.full()){
					merge_inst_ptr.read_addr.try_write(i_req);
					++i_req;
			}
			if(!merge_inst_ptr.read_data.empty()){
				int tmp;
				merge_inst_ptr.read_data.try_read(tmp);
				N = tmp;
				++i_resp;
			}
		}
		for (int i_req = 0, i_resp = 0; i_resp < N;) {
			if((i_req < N) & !merge_inst_ptr.read_addr.full()){
					merge_inst_ptr.read_addr.try_write(i_req+1);
					++i_req;
			}
			if(!merge_inst_ptr.read_data.empty()){
				int tmp;
				merge_inst_ptr.read_data.try_read(tmp);
				merge_inst_q.write(tmp);
				++i_resp;
			}
		}
	}

void X_Merger(int pe_i, int N, tapa::istream<float_v16>& x_first, tapa::istream<float_v16>& x_second, tapa::istream<float_v16>& x_bypass, tapa::ostream<float_v16>& x_out, tapa::istream<int>& N_in){
	int level = N % (WINDOW_LARGE_SIZE) == 0 ? N / WINDOW_LARGE_SIZE : N/WINDOW_LARGE_SIZE + 1;
	int last = (N % WINDOW_LARGE_SIZE)%WINDOW_SIZE == 0 ? (N % WINDOW_LARGE_SIZE)/WINDOW_SIZE - 1:(N % WINDOW_LARGE_SIZE)/WINDOW_SIZE;
	const int num_ite_x = ((N % (WINDOW_SIZE * NUM_CH)) + 15)/16;
	for(int round = 0; round < level;round++){
#pragma HLS loop_flatten off
		const int num_x = N_in.read();

		if(round == level-1 && pe_i > last && last != -1){
bypass:
			for(int i = 0; i < num_ite_x; i++){
				#pragma HLS loop_tripcount min=1 max=385
				x_out.write(x_bypass.read());
			}
		} else {

first:
			for(int i = 0; i < pe_i * WINDOW_SIZE_div_16; i++){
				#pragma HLS loop_tripcount min=1 max=320
				x_out.write(x_first.read());
			}

			const int num_ite = (num_x + 15) / 16;
second:
			for(int i = 0; i < num_ite; i++){
				#pragma HLS loop_tripcount min=1 max=64
				x_out.write(x_second.read());
			}
		}
	}
}

void X_bypass(int pe_i, int N, tapa::istream<float_v16>& fifo_x_in, tapa::ostream<float_v16>& fifo_x_out, tapa::ostream<float_v16>& fifo_x_bypass){
	int level = N % (WINDOW_LARGE_SIZE) == 0 ? N / WINDOW_LARGE_SIZE : N/WINDOW_LARGE_SIZE + 1;
	int last = (N % WINDOW_LARGE_SIZE)%WINDOW_SIZE == 0 ? (N % WINDOW_LARGE_SIZE)/WINDOW_SIZE - 1:(N % WINDOW_LARGE_SIZE)/WINDOW_SIZE;
	const int num_ite = ((N % (WINDOW_SIZE * NUM_CH)) + 15)/16;
	if(pe_i == 0){
		fifo_x_in.read();
	}else {
round: 
		for(int round = 0; round < level ;round++){
#pragma HLS loop_flatten off
			if(round < level-1 || pe_i <= last || last == -1){
		through:
				for(int j = 0; j < WINDOW_SIZE_div_16*pe_i; j++){
					#pragma HLS loop_tripcount min=1 max=320
					fifo_x_out.write(fifo_x_in.read());
				}
			}else{
		bypass:
				for(int j = 0; j < num_ite; j++){
					#pragma HLS loop_tripcount min=1 max=385
					fifo_x_bypass.write(fifo_x_in.read());
				}
			}
		}
	}
}

void read_len( int N, 
	tapa::ostreams<int, NUM_CH>& N_val){
	int bound = (N%WINDOW_LARGE_SIZE == 0) ? N/WINDOW_LARGE_SIZE : N/WINDOW_LARGE_SIZE+1;
	for(int i = 0; i < bound; i++){
#pragma HLS loop_flatten off
		for(int j = 0; j < NUM_CH; j++){
			int remain = N - i*WINDOW_LARGE_SIZE - j*WINDOW_SIZE;
			int len = WINDOW_SIZE;
			if(remain < 0) len = 0;
			else if(remain < WINDOW_SIZE) len = remain;
			N_val[j].write(len);
		}
	}
}

void fill_zero(tapa::ostream<float_v16>& fifo_out){
	float_v16 tmp_x;
	fifo_out.try_write(tmp_x);
}

void fill_zero_dvec(tapa::ostream<MultDVec>& fifo_out){
	MultDVec tmp;
	fifo_out.try_write(tmp);
}

void read_x(int total_N, 
	tapa::istream<int>& fifo_inst_in, 
	tapa::ostream<int>& fifo_inst_out, 
	tapa::ostream<int>& block_id, 
	tapa::istream<float_v16>& fifo_x_in, 
	tapa::ostream<float_v16>& fifo_x_out){

	int bound = (total_N%WINDOW_LARGE_SIZE == 0)?total_N/WINDOW_LARGE_SIZE:total_N/WINDOW_LARGE_SIZE+1;

	for(int round = 0; round < bound; round++){
		for(int ite = 0; ite < (round*NUM_CH)*MULT_SIZE; ite++){
			int need = fifo_inst_in.read();
			fifo_inst_out.write(need);

			if(need == 1){
				block_id.write(ite);
				for(int i = 0; i < WINDOW_SIZE_div_32;){
					#pragma HLS pipeline II=1
					if(!fifo_x_in.empty()){
						float_v16 x_val; fifo_x_in.try_read(x_val);
						fifo_x_out.write(x_val);
						i++;
					}
				}
			}
		}
	}
}

void request_X(tapa::async_mmap<float_v16>& mmap, tapa::istream<int>& block_id, tapa::ostream<float_v16>& fifo_x_out, tapa::istream<bool>& fin_write){

	int block = 0;
	int block_done = 0;

handle_request:
	for(;;){
#pragma HLS loop_flatten off
		if(!fin_write.empty()){
			bool last = fin_write.read(nullptr);
			if(last){
				block_done += 1;
			}else{
				block_done = 0;
			}
		}
scan:
		if(!block_id.empty()){
			block = block_id.peek(nullptr);
			if(block_done <= block) continue;
			else {
				block_id.read(nullptr);
load_x_to_cache:
				for(int i_req = 0, i_resp = 0; i_resp < WINDOW_SIZE_div_32;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=0 max=32
					if((i_req < WINDOW_SIZE_div_32) & !mmap.read_addr.full()){
						mmap.read_addr.try_write(i_req+block*WINDOW_SIZE_div_32);
						++i_req;
					}
					if(!mmap.read_data.empty() & !fifo_x_out.full()){
						float_v16 tmp;
						mmap.read_data.try_read(tmp);
						fifo_x_out.try_write(tmp);
						++i_resp;
					}
				}
			}
		}
	}
}

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> csr_edge_list_ch,
			// tapa::mmaps<int, NUM_CH> csr_edge_list_ptr,
			tapa::mmaps<ap_uint<512>, NUM_CH> dep_graph_ch,
			// tapa::mmaps<int, NUM_CH> dep_graph_ptr,
			tapa::mmap<int> merge_inst_ptr, // TODO: only use one hbm port to afford more channels
			tapa::mmap<float_v16> f, 
			tapa::mmap<float_v16> x, 
			tapa::mmap<int> if_need,
			int N // # of dimension
			){

	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_val("n_val");
	tapa::streams<float_v16, NUM_CH+1, FIFO_DEPTH> x_q("x_pipe");
	tapa::stream<bool, FIFO_DEPTH> fin_write("fin_write");
	tapa::stream<int, FIFO_DEPTH> block_id("block_id");
	tapa::streams<float_v16, NUM_CH+2, FIFO_DEPTH> req_x("req_x");

	tapa::stream<int, FIFO_DEPTH> merge_inst_q("merge_inst_q");
	// tapa::streams<ap_uint<512>, NUM_CH> dep_graph_ch_q("dep_graph_ch_q");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> dep_graph_ch_node("dep_graph_ch_node");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> dep_graph_ch_edge("dep_graph_ch_edge");
	tapa::streams<int, NUM_CH+1, FIFO_DEPTH> dep_graph_ptr_q("dep_graph_ptr_q");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> dep_graph_inst_edge("dep_graph_inst_edge");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> dep_graph_inst("dep_graph_inst");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> f_q("f");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> spmv_val("spmv_val");
	tapa::streams<int, NUM_CH+1, FIFO_DEPTH> csr_edge_list_ptr_q("csr_edge_list_ptr_q");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> spmv_inst("spmv_inst");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> spmv_inst_2("spmv_inst_2");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_sub_1("n_sub_1");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_sub_2("n_sub_2");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_sub_3("n_sub_3");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_sub_4("n_sub_4");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> y("y");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> y_update("y_update");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> x_next("x_next");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> x_apt("x_apt");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> x_bypass("x_bypass");
	tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_aXVec("fifo_aXVec");

	tapa::streams<MultDVec, NUM_CH, FIFO_DEPTH> fifo_node_to_edge("fifo_node_to_edge");
	tapa::streams<MultDVec, NUM_CH, FIFO_DEPTH> fifo_edge_to_node("fifo_edge_to_node");
	tapa::streams<MultDVec, NUM_CH+1, FIFO_DEPTH> fifo_broadcast("fifo_broadcast");

	tapa::streams<int, NUM_CH+2, FIFO_DEPTH> fifo_need("fifo_need");


	tapa::task()
		.invoke<tapa::join>(read_len, N, N_val)
		.invoke<tapa::join>(fill_zero, x_q)
		.invoke<tapa::detach>(request_X, x, block_id, req_x, fin_write)
		.invoke<tapa::join>(read_all_ptr, if_need, fifo_need)
		.invoke<tapa::join>(read_x, N, fifo_need, fifo_need, block_id, req_x, req_x)
		.invoke<tapa::join>(read_f, N, f, f_q)
		// .invoke<tapa::join, NUM_CH>(read_float_vec, tapa::seq(), N, f, N_val, N_sub_1, f_q)
		.invoke<tapa::join>(read_all_ptr, merge_inst_ptr, merge_inst_q)
		.invoke<tapa::join>(split_ptr_inst, N, merge_inst_q, dep_graph_ptr_q, csr_edge_list_ptr_q)
		// .invoke<tapa::join, NUM_CH>(read_A_ptr, tapa::seq(), N, csr_edge_list_ptr, csr_edge_list_ptr_q)
		.invoke<tapa::join, NUM_CH>(read_A_and_split, tapa::seq(), N, csr_edge_list_ch, csr_edge_list_ptr_q, csr_edge_list_ptr_q, spmv_val, spmv_inst)
		.invoke<tapa::detach>(black_hole_int, csr_edge_list_ptr_q)
		// .invoke<tapa::join, NUM_CH>(read_dep_graph_ptr, tapa::seq(), N, dep_graph_ptr, dep_graph_ptr_q)
		.invoke<tapa::join, NUM_CH>(read_dep_graph, tapa::seq(), N, dep_graph_ch, dep_graph_ptr_q, dep_graph_ptr_q, dep_graph_ch_node, dep_graph_ch_edge, dep_graph_inst, dep_graph_inst_edge)
		.invoke<tapa::detach>(black_hole_int, dep_graph_ptr_q)
		.invoke<tapa::join, NUM_CH>(X_bypass, tapa::seq(), N, x_q, x_apt, x_bypass)
		.invoke<tapa::join, NUM_CH>(PEG_Xvec, tapa::seq(), N, spmv_inst, spmv_inst_2, spmv_val, fifo_aXVec, fifo_need, fifo_need, req_x, req_x)
		.invoke<tapa::detach>(black_hole_float_vec, req_x)
		.invoke<tapa::detach>(black_hole_int, fifo_need)
		.invoke<tapa::join, NUM_CH>(PEG_YVec, tapa::seq(), N, spmv_inst_2, fifo_aXVec, y, N_val, N_sub_1)
		.invoke<tapa::detach, NUM_CH>(Yvec_minus_f, f_q, y, y_update, N_sub_1, N_sub_2)
		// .invoke<tapa::detach, NUM_CH>(solve, dep_graph_ch_q, dep_graph_inst, y_update, x_next, N_sub_2, N_sub_3)
		.invoke<tapa::detach, NUM_CH>(solve_node, tapa::seq(), dep_graph_ch_node, dep_graph_inst, y_update, fifo_node_to_edge, fifo_edge_to_node, N_sub_2, N_sub_3)
		.invoke<tapa::join>(fill_zero_dvec, fifo_broadcast)
		.invoke<tapa::detach, NUM_CH>(solve_edge, tapa::seq(), dep_graph_ch_edge, dep_graph_inst_edge, fifo_node_to_edge, fifo_edge_to_node, fifo_broadcast, fifo_broadcast, x_next, N_sub_3, N_sub_4)
		.invoke<tapa::detach>(black_hole_dvec, fifo_broadcast)
		.invoke<tapa::join, NUM_CH>(X_Merger, tapa::seq(), N, x_apt, x_next, x_bypass, x_q, N_sub_4)
		.invoke<tapa::join>(write_x, x_q, fin_write, x, N);
}
