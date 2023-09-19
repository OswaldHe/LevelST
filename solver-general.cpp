#include <ap_int.h>
#include <cstdint>
#include <tapa.h>
#include <iomanip>
//#include <glog/logging.h>

constexpr int WINDOW_SIZE = 8192;
constexpr int WINDOW_SIZE_div_8 = 1024;
constexpr int WINDOW_SIZE_MAX = 8192;
constexpr int WINDOW_SIZE_div_16 = 512;
// constexpr int WINDOW_SIZE_div_32 = 32;
constexpr int NUM_CH = 16;
// constexpr int MULT_SIZE = 16;
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

//TODO: both spmv and dep_graph take data in the same format (row 16 + col 16 + val 32)

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

void black_hole_xvec(tapa::istream<MultXVec>& fifo_in){
	bh(fifo_in);
}

void write_x(tapa::istream<float_v16>& x, tapa::async_mmap<float_v16>& mmap, const int total_N){
	int num_block = total_N / WINDOW_SIZE;
	for(int i = 0; i < num_block; i++){
	    // LOG(INFO) << "process level " << i << " ...";
		int start = WINDOW_SIZE_div_16*i;

write_main:
		for(int i_req = 0, i_resp = 0; i_resp < WINDOW_SIZE_div_16;){
			#pragma HLS pipeline II=1
			if((i_req < WINDOW_SIZE_div_16) & !x.empty() & !mmap.write_addr.full() & !mmap.write_data.full()){
				mmap.write_addr.try_write(i_req + start);
				float_v16 tmp; 
				x.try_read(tmp);
				mmap.write_data.try_write(tmp);
				++i_req;
			}
			if(!mmap.write_resp.empty()){
				i_resp += unsigned(mmap.write_resp.read(nullptr))+1;
				//LOG(INFO) << i_resp;
			}
		}
	}

	//residue
	int residue = ((total_N % WINDOW_SIZE) + 15)/16;
	int offset = num_block * WINDOW_SIZE_div_16;

write_residue:
	for(int i_req = 0, i_resp = 0; i_resp < residue;){
		if(i_req < residue && !x.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
			mmap.write_addr.try_write(i_req + offset);
			mmap.write_data.try_write(x.read(nullptr));
			++i_req;
		}
		if(!mmap.write_resp.empty()){
			i_resp += unsigned(mmap.write_resp.read(nullptr))+1;
			//LOG(INFO) << i_resp;
		}
	}

}

void read_f(const int N, tapa::async_mmap<float_v16>& mmap, tapa::ostreams<float_v2, 8>& f_fifo_out){
	const int num_ite = (N + 31) / 32;
	for(int i_req = 0, i_resp = 0; i_resp < num_ite;){
		if((i_req < num_ite) & !mmap.read_addr.full()){
				mmap.read_addr.try_write(i_req);
				++i_req;
		}
		if(!mmap.read_data.empty()){
				float_v16 tmp;
				mmap.read_data.try_read(tmp);
				for(int i = 0; i < 8; i++){
					float_v2 tmp_x;
					tmp_x[0] = tmp[i];
					tmp_x[1] = tmp[8+i];
					f_fifo_out[i].write(tmp_x);
				}
				++i_resp;
		}
	}
}

void dispatch_inst_x(
	const int pe_i,
	const int NUM_ITE,
	tapa::istream<int>& fifo_inst_in,
	tapa::ostream<int>& fifo_inst_down,
	tapa::istream<int>& fifo_need_in,
	tapa::ostream<int>& fifo_need_out,
	tapa::istream<MultXVec>& fifo_prev_pe,
	tapa::ostream<MultXVec>& fifo_next_pe,
	tapa::istream<float_v16>& req_x_in,
	tapa::ostream<float_v16>& req_x_out,
	tapa::ostream<MultXVec>& fifo_node_down,
	tapa::ostream<float_v16>& req_x_down
){
	
	for(int round = 0; round < NUM_ITE; round++){
		const int N_round = fifo_inst_in.read();
		const int N_layer = fifo_inst_in.read();
		fifo_inst_down.write(0); //phase 0: SpMV
		fifo_inst_down.write(N_round);

		for(int ite = 0; ite < N_round; ite++){
			const int N = fifo_inst_in.read();
			const int need = fifo_need_in.read();
			const int num_ite = (N + 7) / 8;
			fifo_need_out.write(need);

			fifo_inst_down.write(num_ite);

			// dispatch SpMV
			if(need == 1){
				fifo_inst_down.write(WINDOW_SIZE_div_16);
				for(int i = 0; i < WINDOW_SIZE_div_16;){
					#pragma HLS pipeline II=1
					if(!req_x_in.empty()){
						float_v16 x_val; req_x_in.try_read(x_val);
						req_x_down.write(x_val);
						req_x_out.write(x_val);
						i++;
					}
				}
			} else {
				fifo_inst_down.write(0);
			}

		}

		for(int i = 0; i < N_layer - 1; i++){
			const int N_node = fifo_inst_in.read();
			const int N_edge = fifo_inst_in.read();
			fifo_inst_down.write(1); //phase 1: TrigSolve
			fifo_inst_down.write(NUM_CH);

			//dispatch trigsolver
			for(int level = 0; level < NUM_CH; level++){
				fifo_inst_down.write(N_edge);
				fifo_inst_down.write(N_node);

				for(int j = 0; j < N_node;){
					#pragma HLS pipeline II=1
					if(!fifo_prev_pe.empty()){
						MultXVec node_block; fifo_prev_pe.try_read(node_block);
						fifo_next_pe.write(node_block);
						fifo_node_down.write(node_block);
						j++;
					}
				}

			}
		}

	}
}

// void forward(
// 	tapa::istream<MultXVec>& fifo_x_in,
// 	tapa::ostream<MultXVec>& fifo_x_out
// ) {
// 	for(;;){
// 		#pragma HLS pipeline II=1
// 		if(!fifo_x_in.empty()){
// 			MultXVec tmp; fifo_x_in.try_read(tmp);
// 			fifo_x_out.write(tmp);
// 		}
// 	}
// }

void PEG_Xvec(
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<MultXVec>& fifo_node_in,
	tapa::istream<ap_uint<512>>& spmv_A,
	tapa::istream<float_v16>& fifo_req_x,
	tapa::ostream<MultXVec>& fifo_aXVec
){
	for(;;){
		#pragma HLS loop_flatten off

		float local_x[4][8][WINDOW_SIZE_div_8];
#pragma HLS bind_storage variable=local_x latency=2
#pragma HLS array_partition variable=local_x complete dim=1
#pragma HLS array_partition variable=local_x complete dim=2

		const int phase = fifo_inst_in.read(); //phase 0: spmv, phase 1: trigsolver
		const int N_round = fifo_inst_in.read(); // number of blocks

		for(int i = 0; i < N_round; i++){
			const int num_write = fifo_inst_in.read();
			const int num_read = fifo_inst_in.read();

			if(phase == 0){
				for(int j = 0; j < num_read;){
					#pragma HLS pipeline II=1
					if(!fifo_req_x.empty()){
						float_v16 x_val; fifo_req_x.try_read(x_val);
						for(int l = 0; l < 16; l++){
							for(int k = 0; k < 4; k++){
								local_x[k][l%8][(j << 1) + (l >> 3)] = x_val[l];
							}
						}
						j++;
					}
				}
			} else {
				for(int j = 0; j < num_read;){
					#pragma HLS pipeline II=1
					#pragma HLS dependence variable=local_x false
					if(!fifo_node_in.empty()){
						MultXVec node_block; fifo_node_in.try_read(node_block);

						for(int k = 0; k < 8; k++){
							auto row = node_block.row[k];
							auto val = node_block.axv[k];
							if(row[15] == 0){
								for(int l = 0; l < 4; l++){
									local_x[l][k][(row >> 3)] = val;
								}
							}
						}
						j++;
					}
				}
			}

			for(int j = 0; j < num_write;){
				#pragma HLS pipeline II=1

				if(!spmv_A.empty()){
					ap_uint<512> dep_block; spmv_A.try_read(dep_block);
					MultXVec xvec;
					for(int k = 0; k < 8; k++){
						#pragma HLS unroll
						ap_uint<64> a = dep_block(64*k + 63, 64*k);
						ap_uint<16> row = a(63,48);
						ap_uint<16> col = a(47,32);
						ap_uint<32> val = a(31,0);

						xvec.row[k] = row;
						if(row[15] == 0){
							float val_f = tapa::bit_cast<float>(val);
							xvec.axv[k] = (local_x[k/2][col%8][(col >> 3)] * val_f);
							// if(pe_i == 0 && row == 0) LOG(INFO) << "row: " << row << ", col:" << col << ", t1: " << local_x[k/2][col%8][(col >> 3)] << ", t2: " << val_f; 
						}
					}
					fifo_aXVec.write(xvec);
					j++;
				}
			}

		}
	}
}

void solve_node(
	tapa::istream<ap_uint<512>>& dep_graph_ch,
	tapa::istream<int>& dep_graph_ptr,
	tapa::ostream<int>& dep_graph_ptr_out,
	tapa::istream<float_v2>& y_in,
	// tapa::istream<float_v16>& spmv_in,
	tapa::ostream<MultXVec>& fifo_node_to_edge,
	tapa::istream<MultXVec>& fifo_aXVec,
	tapa::ostream<MultXVec>& fifo_x_out,
	tapa::ostream<int>& fifo_inst_out,
	tapa::istream<int>& N_in){
		
for(;;){
#pragma HLS loop_flatten off
	const int N = N_in.read();
	const int N_layer = dep_graph_ptr.read();
	fifo_inst_out.write(N);
	fifo_inst_out.write(N_layer);

	const int num_ite = (N+1)/2;

	float local_y[8][WINDOW_SIZE_div_8];

#pragma HLS array_partition complete variable=local_y dim=1

read_y:
	for(int i = 0, c_idx = 0; i < num_ite;){
#pragma HLS pipeline II=1
		if(!y_in.empty()){
			float_v2 tmp_f; y_in.try_read(tmp_f);
			local_y[c_idx][i >> 2] = tmp_f[0];
			local_y[c_idx+1][i >> 2] = tmp_f[1];
			c_idx+=2;
			if(c_idx == 8) c_idx = 0;
			i++;
		}
	}

compute:
    for(int i = 0; i < N_layer; i++){

		const int num_round = dep_graph_ptr.read();

	computation:
		for(int j = 0; j < num_round;){
			#pragma HLS loop_tripcount min=1 max=200
			#pragma HLS pipeline II=1
			#pragma HLS dependence true variable=local_y distance=8

			if(!fifo_aXVec.empty()){
				MultXVec ravx; fifo_aXVec.try_read(ravx);
				for(int k = 0; k < 8; k++){
					auto a_row = ravx.row[k];
					auto a_val = ravx.axv[k];
					if(a_row[15] == 0){
						local_y[k][a_row >> 3] -= a_val;
					}
				}
				++j;
			}
		}

		const int N_node = dep_graph_ptr.read();
		fifo_inst_out.write(N_node);
		if (i < N_layer-1) dep_graph_ptr_out.write(N_node);

compute_node:
		for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1
			#pragma HLS dependence variable=local_y false
			if(!dep_graph_ch.empty()){
				ap_uint<512> dep_block; dep_graph_ch.try_read(dep_block);
				MultXVec xvec;

				for(int k = 0; k < 8; k++){
					#pragma HLS unroll
					ap_uint<64> a = dep_block(64*k + 63, 64*k);
					ap_uint<16> row = a(63,48);
					ap_uint<32> val = a(31,0);
					float val_f = tapa::bit_cast<float>(val);

					xvec.row[k] = row;
					if(row[15] == 0){
						float ans = local_y[k][(row >> 3)] * val_f;
						xvec.axv[k] = ans;
						// if(pe_i == 0 && row == 3) LOG(INFO) << "row: " << row << ", t1: " << local_y[k][(row >> 3)] << ", t2: " << val_f; 
					}
				}
				fifo_x_out.write(xvec);
				if (i < N_layer-1) fifo_node_to_edge.write(xvec);
				j++;
			}
		}

	}

}


	}

void cache_x_and_write(
	tapa::istream<MultXVec>& fifo_x_in,
	tapa::ostream<MultXVec>& fifo_x_to_fwd,
	tapa::ostream<float>& fifo_x_out,
	tapa::istream<int>& fifo_inst_in,
	tapa::ostream<int>& fifo_inst_out
){
	for(;;){
#pragma HLS loop_flatten off

		float local_x[8][WINDOW_SIZE_div_8];

#pragma HLS bind_storage variable=local_x type=RAM_2P impl=URAM
#pragma HLS array_partition complete variable=local_x dim=1

		const int num_ite = fifo_inst_in.read();
		const int num_layer = fifo_inst_in.read();
		fifo_inst_out.write(num_ite);
		fifo_inst_out.write(num_layer);

		for(int i = 0; i < num_layer; i++){
			const int N_node = fifo_inst_in.read();
			fifo_inst_out.write(N_node);

			for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1

				if(!fifo_x_in.empty()){
					MultXVec ravx; fifo_x_in.try_read(ravx);
					fifo_x_to_fwd.write(ravx);
					for(int k = 0; k < 8; k++){
						auto a_row = ravx.row[k];
						auto a_val = ravx.axv[k];
						if(a_row[15] == 0){
							local_x[k][a_row >> 3] = a_val;
						}
					}
					j++;
				}
			}
		}

		for(int i = 0; i < num_ite; i++){
		#pragma HLS loop_tripcount min=1 max=64
		#pragma HLS pipeline II=1
			fifo_x_out.write(local_x[i%8][i >> 3]);
		}
	}
}

void cache_x_and_fwd(
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<MultXVec>& fifo_x_in,
	tapa::istream<int>& block_fwd,
	tapa::ostream<int>& block_fwd_out,
	tapa::ostream<float>& fifo_x_fwd
){
	for(;;){
#pragma HLS loop_flatten off

		float local_x[8][WINDOW_SIZE_div_8];

#pragma HLS bind_storage variable=local_x type=RAM_2P impl=URAM
#pragma HLS array_partition complete variable=local_x dim=1

		const int num_ite = fifo_inst_in.read();
		const int num_layer = fifo_inst_in.read();

		for(int i = 0; i < num_layer; i++){
			const int N_node = fifo_inst_in.read();
			for(int j = 0; j < N_node;){
			#pragma HLS pipeline II=1
				if(!fifo_x_in.empty()){
					MultXVec ravx; fifo_x_in.try_read(ravx);
					for(int k = 0; k < 8; k++){
						auto a_row = ravx.row[k];
						auto a_val = ravx.axv[k];
						if(a_row[15] == 0){
							local_x[k][a_row >> 3] = a_val;
						}
					}
					j++;
				}
			}
		}
		
		for(int i = 0 ; i < NUM_CH; i++){
			//forwarding to next batch
			const int fwd_block = block_fwd.read();
			block_fwd_out.write(fwd_block);

			if(fwd_block == 1){
				for(int j = 0; j < WINDOW_SIZE_div_16; j++){
				#pragma HLS loop_tripcount min=1 max=64
				#pragma HLS pipeline II=1
					fifo_x_fwd.write(local_x[j%8][i*64 + (j >> 3)]);
				}
			}
		}
	}
}

void cache_x_and_feed(
	const int pe_i,
	tapa::istream<MultXVec>& fifo_x_in,
	tapa::istream<MultXVec>& fifo_prev,
	tapa::ostream<MultXVec>& fifo_x_out,
	tapa::istream<int>& fifo_inst
) {
	for(;;){

		if(pe_i == 0 && !fifo_prev.empty()){
			fifo_prev.read(nullptr);
		}

		MultXVec cache_x[WINDOW_SIZE_div_8];

		#pragma HLS aggregate variable=cache_x
		#pragma HLS bind_storage variable=cache_x type=RAM_2P impl=URAM

		const int N = fifo_inst.read();
		for(int i = 0; i < N;){
			#pragma HLS pipeline II=1
			if(!fifo_x_in.empty()){
				fifo_x_in.try_read(cache_x[i]);
				i++;
			}
		}

		for(int i = 0; i < pe_i+1; i++){
			for(int j = 0; j < N;){
				MultXVec tmp;
				if(pe_i == i){
					fifo_x_out.write(cache_x[j]);
					j++;
				} else if(!fifo_prev.empty()){
					fifo_prev.try_read(tmp);
					fifo_x_out.write(tmp);
					j++;
				}
			}
		}
	}
}

void read_comp_packet(
	const int A_LEN,
	tapa::async_mmap<ap_uint<512>>& comp_packet_ch,
	tapa::ostream<ap_uint<512>>& fifo_comp_packet_out
){
	for (int i_req = 0, i_resp = 0; i_resp < A_LEN;) {
	#pragma HLS pipeline II=1
		if(i_req < A_LEN && !comp_packet_ch.read_addr.full()){
				comp_packet_ch.read_addr.try_write(i_req);
				++i_req;
		}
		if(!comp_packet_ch.read_data.empty()){
			ap_uint<512> tmp;
			comp_packet_ch.read_data.try_read(tmp);
			fifo_comp_packet_out.write(tmp);
			++i_resp;
		}
	}
}

void split_comp_packet(
	const int pe_i,
	const int NUM_ITE,
	tapa::istream<ap_uint<512>>& comp_packet_ch,
	tapa::istream<int>& comp_packet_ptr,
	tapa::ostream<int>& comp_packet_ptr_out,
	tapa::ostream<ap_uint<512>>& fifo_solve_node,
	tapa::ostream<ap_uint<512>>& fifo_spmv,
	tapa::ostream<int>& fifo_inst_solve_node,
	tapa::ostream<int>& fifo_inst_spmv
){
	for(int round = 0; round < NUM_ITE; round++){
		const int spmv_N = comp_packet_ptr.read();
		const int dep_graph_N = comp_packet_ptr.read();
		const int spmv_total_len = comp_packet_ptr.read();
		comp_packet_ptr_out.write(spmv_N);
		comp_packet_ptr_out.write(dep_graph_N);
		comp_packet_ptr_out.write(spmv_total_len);

		fifo_inst_spmv.write(spmv_N);
		fifo_inst_spmv.write(dep_graph_N);
		fifo_inst_solve_node.write(dep_graph_N);

		fifo_inst_solve_node.write(spmv_total_len);

		for(int i = 0; i < spmv_N; i++){
			const int N = comp_packet_ptr.read();
			const int num_ite = (N + 7) / 8;
			comp_packet_ptr_out.write(N);
			fifo_inst_spmv.write(N);

			for (int j = 0; j < num_ite;) {
			#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet; comp_packet_ch.try_read(packet);
					fifo_spmv.write(packet);
					j++;
				}
			}
		}

		for(int i = 0; i < dep_graph_N-1; i++){
			const int N_node = comp_packet_ptr.read();
			const int N_edge = comp_packet_ptr.read();

			comp_packet_ptr_out.write(N_node);
			comp_packet_ptr_out.write(N_edge);
			fifo_inst_solve_node.write(N_node);
			fifo_inst_spmv.write(N_node);
			fifo_inst_spmv.write(N_edge);

			for (int j = 0; j < N_node;) {
			#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet; comp_packet_ch.try_read(packet);
					fifo_solve_node.write(packet);
					j++;
				}
			}

			fifo_inst_solve_node.write(N_edge * NUM_CH);

			for (int j = 0; j < N_edge * NUM_CH;) {
			#pragma HLS pipeline II=1
				if(!comp_packet_ch.empty()){
					ap_uint<512> packet; comp_packet_ch.try_read(packet);
					fifo_spmv.write(packet);
					j++;
				}
			}
		}

		const int N_node = comp_packet_ptr.read();
		const int N_edge = comp_packet_ptr.read();

		comp_packet_ptr_out.write(N_node);
		comp_packet_ptr_out.write(N_edge);
		fifo_inst_solve_node.write(N_node);

		for (int j = 0; j < N_node;) {
		#pragma HLS pipeline II=1
			if(!comp_packet_ch.empty()){
				ap_uint<512> packet; comp_packet_ch.try_read(packet);
				fifo_solve_node.write(packet);
				j++;
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
				merge_inst_ptr.read_data.try_read(N);
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

void X_merge_sub(tapa::istreams<float, 2>& x_in, tapa::ostream<float_v2>& x_out){
	for(;;){
		#pragma HLS pipeline II=1

		bool flag_nop = x_out.full();
		for(int i = 0; i < 2; i++){
			flag_nop |= x_in[i].empty();
		}

		if(!flag_nop){
			float_v2 tmp;
			for(int i = 0; i < 2; i++){
				float tmp_x; x_in[i].try_read(tmp_x);
				tmp[i] = tmp_x;
			}
			x_out.try_write(tmp);
		}
	}
}

void X_Merger(int N, tapa::istreams<float_v2, 8>& x_in, tapa::ostream<float_v16>& x_out){
	for(;;){
		#pragma HLS pipeline II=1

		bool flag_nop = x_out.full();
		for(int i = 0; i < 8; i++){
			flag_nop |= x_in[i].empty();
		}

		if(!flag_nop){
			float_v16 tmp_v16;
			for(int i = 0; i < 8; i++){
				float_v2 tmp_v2; x_in[i].try_read(tmp_v2);
				for(int d = 0; d < 2; d++){
					tmp_v16[i*2+d] = tmp_v2[d];
				}
			}
			x_out.try_write(tmp_v16);
		}
	}
}

void read_len( const int N, const int NUM_ITE,
	tapa::ostreams<int, NUM_CH>& N_val){
	for(int i = 0; i < NUM_ITE; i++){
		int remain = N - i * WINDOW_LARGE_SIZE;
		int len = WINDOW_SIZE;
		if(remain < WINDOW_LARGE_SIZE){
			len = (remain + 15) / 16;
		}
		for(int j = 0; j < NUM_CH; j++){
			N_val[j].write(len);
		}
	}
}

void fill_zero(tapa::ostream<float_v16>& fifo_out){
	float_v16 tmp_x;
	fifo_out.try_write(tmp_x);
}

void fill_zero_xvec(tapa::ostream<MultXVec>& fifo_out){
	MultXVec tmp;
	fifo_out.try_write(tmp);
}

// void read_x(const int NUM_ITE, 
// 	tapa::istream<int>& fifo_inst_in, 
// 	tapa::ostream<int>& fifo_inst_out, 
// 	tapa::ostream<int>& block_id){

// 	for(int round = 1; round < NUM_ITE; round++){
// 		for(int ite = 0; ite < ((round-1)*NUM_CH); ite++){
// 			int need = fifo_inst_in.read();
// 			fifo_inst_out.write(need);

// 			if(need == 1){
// 				block_id.write(ite);
// 			}
// 		}
// 		int num_block = fifo_inst_in.read();
// 		block_id.write(num_block);

// 		LOG(INFO) << num_block;

// 		for(int ite = 0; ite < NUM_CH; ite++){
// 			int need = fifo_inst_in.read();
// 			fifo_inst_out.write(need);
// 			LOG(INFO) << need;
// 		}
// 	}
// }

// void write_progress(tapa::istream<bool>& fin_write, tapa::istream<int>& block_id, tapa::ostream<int>& block_fin){
// 	int block_done = 0;

// 	for(;;){
// #pragma HLS pipeline II=1

// 		if(!fin_write.empty()){
// 			bool last = fin_write.read(nullptr);
// 			if(last){
// 				block_done += 1;
// 			}else{
// 				block_done = 0;
// 			}
// 		}

// 		if(!block_id.empty() & !block_fin.full()){
// 			int block = block_id.peek(nullptr);
// 			if(block_done > block){
// 				block_id.read(nullptr);
// 				block_fin.try_write(block);
// 			} 
// 		}
// 	}
// }

void request_X(
	const int NUM_ITE, 
	tapa::async_mmap<float_v16>& mmap, 
	tapa::istreams<float, NUM_CH>& fifo_x_fwd, 
	tapa::ostream<float_v16>& fifo_x_out, 
	tapa::istream<int>& fifo_inst_in,
	tapa::ostream<int>& fifo_inst_out){

	for(int round = 1; round < NUM_ITE; round++){
		for(int ite = 0; ite < ((round-1)*NUM_CH); ite++){
			int need = fifo_inst_in.read();
			fifo_inst_out.write(need);

			if(need == 1){	
load_x:
				for(int i_req = ite * WINDOW_SIZE_div_16, i_resp = 0; i_resp < WINDOW_SIZE_div_16;){
	#pragma HLS pipeline II=1
	#pragma HLS loop_tripcount min=0 max=32
					if((i_req < (ite+1) * WINDOW_SIZE_div_16) & !mmap.read_addr.full()){
						mmap.read_addr.try_write(i_req);
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

		// consume fwd x and pass to PEG_Xvec
		for(int ite = 0; ite < NUM_CH; ite++){
			int need = fifo_inst_in.read();
			fifo_inst_out.write(need);

			if(need == 1){

fwd_x:
				for(int i = 0; i < WINDOW_SIZE_div_16;){
				#pragma HLS pipeline II=1
					bool flag_nop = fifo_x_out.full();
					for(int j = 0; j < NUM_CH; j++){
						flag_nop |= fifo_x_fwd[j].empty();
					}
					if(!flag_nop){
						float_v16 tmp;
						for(int j = 0; j < NUM_CH; j++){
							fifo_x_fwd[j].try_read(tmp[j]);
						}
						fifo_x_out.try_write(tmp);
						i++;
					}
				}
			}
		}
	}
}

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> comp_packet_ch,
			tapa::mmap<int> merge_inst_ptr, // TODO: only use one hbm port to afford more channels
			tapa::mmaps<float_v16, 2> f, 
			tapa::mmap<float_v16> x, 
			tapa::mmap<int> if_need,
			tapa::mmap<int> block_fwd,
			const int N, // # of dimension
			const int NUM_ITE, // # number of rounds
			const int A_LEN
			){

	tapa::streams<int, NUM_CH, FIFO_DEPTH> N_val("n_val");
	tapa::streams<float, NUM_CH, FIFO_DEPTH> x_q("x_pipe");
	// tapa::stream<bool, FIFO_DEPTH> fin_write("fin_write");
	// tapa::stream<int, FIFO_DEPTH> block_id("block_id");
	// tapa::stream<int, FIFO_DEPTH> block_fin("block_fin");
	tapa::streams<int, NUM_CH+1, FIFO_DEPTH> fifo_block_fwd("fifo_block_fwd");
	tapa::streams<float_v16, NUM_CH+1, FIFO_DEPTH> req_x("req_x");
	tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> req_x_down("req_x_down");
	tapa::streams<float, NUM_CH, FIFO_DEPTH> fifo_x_fwd("fifo_x_fwd");

	tapa::streams<int, NUM_CH+1, FIFO_DEPTH> merge_inst_q("merge_inst_q");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> fifo_solve_node("fifo_solve_node");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> fifo_spmv("fifo_spmv");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> fifo_inst_solve_node("fifo_inst_solve_node");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> fifo_inst_spmv("fifo_inst_spmv");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> fifo_inst_spmv_down("fifo_inst_spmv_down");


	tapa::streams<float_v2, NUM_CH, FIFO_DEPTH> f_q("f");
	// tapa::streams<int, NUM_CH, FIFO_DEPTH> spmv_inst("spmv_inst");
	// tapa::streams<int, NUM_CH, FIFO_DEPTH> spmv_inst_out("spmv_inst_out");
	// tapa::streams<int, NUM_CH, FIFO_DEPTH> N_sub_1("n_sub_1");
	// tapa::streams<int, NUM_CH, FIFO_DEPTH> N_sub_2("n_sub_2");
	// tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> y("y");
	tapa::stream<float_v16, FIFO_DEPTH> x_next("x_next");
	// tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> x_apt("x_apt");
	// tapa::streams<float_v16, NUM_CH, FIFO_DEPTH> x_bypass("x_bypass");
	tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_aXVec("fifo_aXVec");

	tapa::streams<int, NUM_CH, FIFO_DEPTH> fifo_forward_node("fifo_forward_node");
	tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_node_to_edge("fifo_node_to_edge");
	// tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_node_out("fifo_node_out");
	tapa::streams<MultXVec, NUM_CH*2+1, FIFO_DEPTH> fifo_broadcast("fifo_broadcast");

	tapa::streams<int, NUM_CH+2, FIFO_DEPTH> fifo_need("fifo_need");

	tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_node_down("fifo_node_down");
	tapa::streams<float_v2, 8, FIFO_DEPTH> x_merge("x_merge");
	// tapa::stream<MultXVec, FIFO_DEPTH> fifo_apt("fifo_apt");

	tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_cache_x("fifo_cache_x");
	tapa::streams<MultXVec, NUM_CH, FIFO_DEPTH> fifo_cache_x_2("fifo_cache_x_2");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> fifo_inst_cache_x("fifo_inst_cache_x");
	tapa::streams<int, NUM_CH, FIFO_DEPTH> fifo_inst_cache_x_2("fifo_inst_cache_x_2");
	tapa::streams<ap_uint<512>, NUM_CH, FIFO_DEPTH> fifo_comp_packet_out("fifo_comp_packet_out");


	tapa::task()
		.invoke<tapa::join>(read_len, N, NUM_ITE, N_val)
		// .invoke<tapa::join>(fill_zero, x_q)
		// .invoke<tapa::detach>(write_progress, fin_write, block_id, block_fin)
		.invoke<tapa::join>(read_all_ptr, if_need, fifo_need)
		.invoke<tapa::join>(read_all_ptr, block_fwd, fifo_block_fwd)
		.invoke<tapa::join>(request_X, NUM_ITE, x, fifo_x_fwd, req_x, fifo_need, fifo_need)
		// .invoke<tapa::join>(read_x, NUM_ITE, fifo_need, fifo_need, block_id)
		.invoke<tapa::join, 2>(read_f, N, f, f_q)
		// .invoke<tapa::join, NUM_CH>(read_float_vec, tapa::seq(), N, f, N_val, N_sub_1, f_q)
		.invoke<tapa::join>(read_all_ptr, merge_inst_ptr, merge_inst_q)
		.invoke<tapa::join, NUM_CH>(
			read_comp_packet,
			A_LEN,
			comp_packet_ch,
			fifo_comp_packet_out
		)
		.invoke<tapa::join, NUM_CH>(
			split_comp_packet, 
			tapa::seq(),
			NUM_ITE,
			fifo_comp_packet_out,
			merge_inst_q,
			merge_inst_q,
			fifo_solve_node,
			fifo_spmv,
			fifo_inst_solve_node,
			fifo_inst_spmv)
		.invoke<tapa::detach>(black_hole_int, merge_inst_q)
		.invoke<tapa::join>(fill_zero_xvec, fifo_broadcast)
		.invoke<tapa::detach, NUM_CH>(cache_x_and_feed, tapa::seq(), fifo_node_to_edge, fifo_broadcast, fifo_broadcast, fifo_forward_node)
		.invoke<tapa::join, NUM_CH>(
			dispatch_inst_x,
			tapa::seq(),
			NUM_ITE,
			fifo_inst_spmv,
			fifo_inst_spmv_down,
			fifo_need,
			fifo_need,
			fifo_broadcast,
			fifo_broadcast,
			req_x,
			req_x,
			fifo_node_down,
			req_x_down
		)
		.invoke<tapa::detach, NUM_CH>(
			PEG_Xvec,
			fifo_inst_spmv_down, 
			fifo_node_down,
			fifo_spmv,
			req_x_down,
			fifo_aXVec)
		.invoke<tapa::detach>(black_hole_xvec, fifo_broadcast)
		.invoke<tapa::detach>(black_hole_float_vec, req_x)
		.invoke<tapa::detach>(black_hole_int, fifo_need)
		.invoke<tapa::detach, NUM_CH>(
			solve_node, 
			fifo_solve_node, 
			fifo_inst_solve_node, 
			fifo_forward_node, 
			f_q, 
			fifo_node_to_edge,
			fifo_aXVec, 
			fifo_cache_x, 
			fifo_inst_cache_x, 
			N_val)
		.invoke<tapa::detach, NUM_CH>(cache_x_and_write, fifo_cache_x, fifo_cache_x_2, x_q, fifo_inst_cache_x, fifo_inst_cache_x_2)
		.invoke<tapa::detach, NUM_CH>(cache_x_and_fwd, fifo_inst_cache_x_2, fifo_cache_x_2, fifo_block_fwd, fifo_block_fwd, fifo_x_fwd)
		.invoke<tapa::detach>(black_hole_int, fifo_block_fwd)
		.invoke<tapa::detach, 8>(X_merge_sub, x_q, x_merge)
		.invoke<tapa::detach>(X_Merger, N, x_merge, x_next)
		.invoke<tapa::join>(write_x, x_next, x, N);
}
