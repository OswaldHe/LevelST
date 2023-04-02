#include <ap_int.h>
#include <cstdint>
#include <tapa.h>
//#include <glog/logging.h>

constexpr int WINDOW_SIZE = 1024;
constexpr int WINDOW_SIZE_div_16 = 64;
constexpr int NUM_CH = 6;

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;

struct MultXVec {
	tapa::vec_t<ap_uint<12>, 8> row;
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

void black_hole_ap_uint(tapa::istream<ap_uint<96>>& fifo_in){
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

void float_to_float_vec(int N, tapa::istream<float>& x, tapa::ostream<float_v16>& x_vec){
	for(int i = 0; i < N / 16; i++){
		float_v16 tmp;
		for(int j = 0; j < 16;){
			if(!x.empty()){
				tmp[j] = x.read(nullptr);
				j++;
			}
		}
		x_vec.write(tmp);
	}

	float_v16 rem;
	for(int i = 0; i < N % 16;){
		if(!x.empty()){
			rem[i] = x.read(nullptr);
			i++;
		}
	}
	x_vec.write(rem);
}

void write_x(tapa::istream<float_v16>& x, tapa::ostream<bool>& fin_write, tapa::async_mmap<float_v16>& mmap, int total_N){
	int num_block = total_N / WINDOW_SIZE;
	for(int i = 0; i < num_block; i++){
	    // LOG(INFO) << "process level " << i << " ...";
		int start = WINDOW_SIZE_div_16*i;
		int N = WINDOW_SIZE_div_16;
		for(int i_req = 0, i_resp = 0; i_resp < N;){
			if(i_req < N && !x.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
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
	int residue = ((total_N % WINDOW_SIZE) + 15)/16;
	for(int i_req = 0, i_resp = 0; i_resp < residue;){
		if(i_req < residue && !x.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
			mmap.write_addr.try_write(i_req + num_block * WINDOW_SIZE_div_16);
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

//TODO: replace with serpen after modifying the read width
void PEG_Xvec( int pe_i, int total_N,
	tapa::istream<int>& fifo_inst_in, tapa::ostream<int>& fifo_inst_out,
	tapa::istream<ap_uint<512>>& spmv_A, // 12-bit row + 20-bit col + 32-bit float
	tapa::ostream<MultXVec>& fifo_aXVec,
	tapa::ostream<int>& block_id, tapa::istream<float_v16>& req_x){

		int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
		int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

round:
		for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off

			float local_x[4][WINDOW_SIZE];

#pragma HLS array_partition variable=local_x complete dim=1
#pragma HLS array_partition variable=local_x cyclic factor=16 dim=2

load_x:
			for(int ite = 0; ite < pe_i+round*NUM_CH; ite++){
				const int N = fifo_inst_in.read();
				fifo_inst_out.write(N);
				const int num_ite = (N + 7) / 8;
				//read x
				if(N != 0){
					block_id.write(ite);
					for(int i = 0; i < WINDOW_SIZE_div_16;){
						#pragma HLS pipeline II=1
						if(!req_x.empty()){
							float_v16 x_val; req_x.try_read(x_val);
							for(int l = 0; l < 16; l++){
								for(int k = 0; k < 4; k++){
									#pragma HLS unroll
									local_x[k][i*16+l] = x_val[l];
								}
							}
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
							ap_uint<12> a_row = a(63, 52);
							ap_uint<20> a_col = a(51, 32);
							ap_uint<32> a_val = a(31, 0);

							raxv.row[k] = a_row;
							if(a_row[11] == 0){
								float a_val_f = tapa::bit_cast<float>(a_val);
								raxv.axv[k] = local_x[i%4][a_col] * a_val_f;
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
	int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
	int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

round:
		for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off
			const int num_x = N_in.read();
			N_out.write(num_x);

			float local_c[8][WINDOW_SIZE];
#pragma HLS array_partition variable=local_c complete dim=1

reset:
			for(int i = 0; i < num_x; i++){
				for(int j = 0; j < 8; j++){
					local_c[j][i] = 0;
				}
			}

load_axv:
			for(int ite = 0; ite < pe_i+round*NUM_CH; ite++){
				const int N = fifo_inst_in.read();
				const int num_ite = (N + 7) / 8;

acc:
				for(int i = 0; i < num_ite;){
					#pragma HLS dependence variable=local_c type=intra false

					if(!fifo_aXVec.empty()){
						MultXVec ravx; fifo_aXVec.try_read(ravx);
						for(int k = 0; k < 8; k++){
							#pragma HLS unroll
							auto a_row = ravx.row[k];
							if(a_row[11] == 0){
								local_c[k][a_row] += ravx.axv[k];
							}
						}
						i++;
					}
				}
			}

write_y:
			for(int i = 0; i < num_x; i+=16){
#pragma HLS pipeline II=1

				float_v16 tmp;
				for(int j = 0; j < 16; j++){
					#pragma HLS unroll

					tmp[j] = local_c[j%8][i+j];
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
			#pragma HLS loop_tripcount min=1 max=256
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

void solve(tapa::istream<ap_uint<512>>& dep_graph_ch,
		tapa::istream<int>& dep_graph_ptr,
		tapa::istream<float_v16>& y_in,
		tapa::ostream<float>& x_out, 
		tapa::istream<int>& N_in,
		tapa::ostream<int>& N_out){

for(;;){
#pragma HLS loop_flatten off
	const int N = N_in.read();
	N_out.write(N);
	const int N_layer = dep_graph_ptr.read();
	const int num_ite = (N + 15) / 16;

	float local_x[WINDOW_SIZE];
	float local_y[WINDOW_SIZE];

#pragma HLS array_partition cyclic variable=local_x factor=8
#pragma HLS array_partition cyclic variable=local_y factor=16

read_y:
	for(int i = 0; i < num_ite;){
#pragma HLS pipeline II=1
		if(!y_in.empty()){
			float_v16 tmp_f; y_in.try_read(tmp_f);
			for(int j = 0; j < 16; j++){
				#pragma HLS unroll
				local_y[(i << 4) + j] = tmp_f[j];
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
						local_x[row] = local_y[row] * val_f;
					}
				}
				j++;
			}
		}

compute_edge:
		for(int j = 0; j < N_edge;){
			#pragma HLS pipeline II=1
			#pragma HLS dependence variable=local_x type=intra false
			#pragma HLS dependence variable=local_y type=intra false
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
						local_y[row] -= (local_x[col] * val_f);
					}
				}
				j++;
			}
		}
	}


write_x:
	for(int i = 0; i < N; i++){
#pragma HLS loop_tripcount min=1 max=256
#pragma HLS pipeline II=1
		x_out.write(local_x[i]);
	}
}
}

void read_A_and_split(int pe_i, int total_N,
	tapa::async_mmap<ap_uint<512>>& csr_edge_list_ch,
	tapa::mmap<int> csr_edge_list_ptr,
	tapa::ostream<ap_uint<512>>& spmv_val,
	tapa::ostream<int>& spmv_inst){
		int offset = 0;
		int offset_i = 0;
		int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
		int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
round:
		for(int round = 0;round < bound;round++){
#pragma HLS loop_flatten off
traverse_block:
			for(int i = 0; i < pe_i+NUM_CH*round; i++){
				#pragma HLS pipeline II=1
				const int N = csr_edge_list_ptr[i+offset_i];
				const int num_ite = (N + 7) / 8;
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
			offset_i += (pe_i+NUM_CH*round);
		}
	}

void read_dep_graph(int pe_i, int total_N,
	tapa::async_mmap<ap_uint<512>>& dep_graph_ch,
	tapa::istream<int>& dep_graph_ptr,
	tapa::ostream<ap_uint<512>>& dep_graph_q,
	tapa::ostream<int>& dep_graph_inst){
		int offset = 0;
		int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
		int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
round:
		for(int round = 0;round < bound;round++){
#pragma HLS loop_flatten off
			const int N_layer = dep_graph_ptr.read();
			dep_graph_inst.write(N_layer);

layer:
			for(int i = 0; i < N_layer; i++){
				#pragma HLS pipeline II=1 style=frp
				const int N_node = dep_graph_ptr.read();
				const int N_edge = dep_graph_ptr.read();
				const int N_chunk = N_node + N_edge;

				dep_graph_inst.write(N_node);
				dep_graph_inst.write(N_edge);

split:
				for (int i_req = 0, i_resp = 0; i_resp < N_chunk;) {
					if(i_req < N_chunk && !dep_graph_ch.read_addr.full()){
							dep_graph_ch.read_addr.try_write(i_req+offset);
							++i_req;
					}
					if(!dep_graph_ch.read_data.empty()){
						ap_uint<512> tmp;
						dep_graph_ch.read_data.try_read(tmp);
						dep_graph_q.write(tmp);
						++i_resp;
					}
				}
				offset+=N_chunk;
			}
		}
	}

void read_dep_graph_ptr(int pe_i, int total_N,
	tapa::async_mmap<int>& dep_graph_ptr,
	tapa::ostream<int>& dep_graph_ptr_q){
		int N = 0;
		for (int i_req = 0, i_resp = 0; i_resp < 1;) {
			if(i_req < 1 && !dep_graph_ptr.read_addr.full()){
					dep_graph_ptr.read_addr.try_write(i_req);
					++i_req;
			}
			if(!dep_graph_ptr.read_data.empty()){
				int tmp;
				dep_graph_ptr.read_data.try_read(tmp);
				N = tmp;
				++i_resp;
			}
		}
		for (int i_req = 0, i_resp = 0; i_resp < N;) {
			if(i_req < N && !dep_graph_ptr.read_addr.full()){
					dep_graph_ptr.read_addr.try_write(i_req+1);
					++i_req;
			}
			if(!dep_graph_ptr.read_data.empty()){
				int tmp;
				dep_graph_ptr.read_data.try_read(tmp);
				dep_graph_ptr_q.write(tmp);
				++i_resp;
			}
		}
	}

void X_Merger(int pe_i, int N, tapa::istream<float>& x_first, tapa::istream<float>& x_second, tapa::istream<float>& x_bypass, tapa::ostream<float>& x_out, tapa::istream<int>& N_in){
	int level = N / (WINDOW_SIZE * NUM_CH);
	int last = (N % (WINDOW_SIZE * NUM_CH))%WINDOW_SIZE == 0 ? (N % (WINDOW_SIZE * NUM_CH))/WINDOW_SIZE - 1:(N % (WINDOW_SIZE * NUM_CH))/WINDOW_SIZE;
	for(int round = 0; round <= level;round++){
#pragma HLS loop_flatten off
		if(round == level && pe_i > last){
			for(int i = 0; i < N % (WINDOW_SIZE * NUM_CH); i++){
				x_out.write(x_bypass.read());
			}
		} else {
			for(int i = 0; i < pe_i * WINDOW_SIZE; i++){
				x_out.write(x_first.read());
			}

			const int num_x = N_in.read();
			for(int i = 0; i < num_x; i++){
				x_out.write(x_second.read());
			}
		}
	}
}

void X_bypass(int pe_i, int N, tapa::istream<float>& fifo_x_in, tapa::ostream<float>& fifo_x_out, tapa::ostream<float>& fifo_x_bypass){
	int level = N / (WINDOW_SIZE * NUM_CH);
	int last = (N % (WINDOW_SIZE * NUM_CH))%WINDOW_SIZE == 0 ? (N % (WINDOW_SIZE * NUM_CH))/WINDOW_SIZE - 1:(N % (WINDOW_SIZE * NUM_CH))/WINDOW_SIZE;
	if(pe_i == 0){
		fifo_x_in.read();
	}else {
		for(int round = 0; round <= level ;round++){
#pragma HLS loop_flatten off
			if(round < level || pe_i <= last){
				for(int j = 0; j < WINDOW_SIZE*pe_i; j++){
					fifo_x_out.write(fifo_x_in.read());
				}
			}else{
				for(int j = 0; j < N % (WINDOW_SIZE * NUM_CH); j++){
					fifo_x_bypass.write(fifo_x_in.read());
				}
			}
		}
	}
}

void read_len( int N, 
	tapa::ostreams<int, NUM_CH>& N_val){
	int bound = (N%WINDOW_SIZE == 0) ? N/WINDOW_SIZE : N/WINDOW_SIZE+1;
	for(int i = 0; i < bound; i++){
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
		int len = ((N-i*WINDOW_SIZE) < WINDOW_SIZE) ? N%WINDOW_SIZE : WINDOW_SIZE;
		int ind = i%NUM_CH;
		N_val[ind].write(len);
	}
}

void fill_zero(tapa::ostream<float>& fifo_out){
	fifo_out.try_write(0.0);
}

void request_X(tapa::async_mmap<float_v16>& mmap, tapa::istreams<int, NUM_CH>& block_id, tapa::ostreams<float_v16, NUM_CH>& fifo_x_out, tapa::istream<bool>& fin_write){
// 	float_v16 local_cache_x[WINDOW_SIZE/4];
// 	int cache_index[4];

// #pragma HLS array_partition variable=local_cache_x cyclic factor=8
// #pragma HLS array_partition variable=cache_index complete

	int block = 0;
	int block_done = 0;

// init_cache_tag:
// 	for(int i = 0; i < 4; i++) {
// #pragma HLS pipeline II=1
// 		cache_index[i] = -1;
// 	}

handle_request:
	for(;;){
#pragma HLS loop_flatten off
		if(!fin_write.empty()){
			bool last = fin_write.read(nullptr);
			if(last){
				block_done++;
			}else{
				// for(int i = 0; i < 4; i++) {
				// 	cache_index[i] = -1;
				// }
				block_done = 0;
			}
		}
scan:
		for(int i = 0; i < NUM_CH; i++){
			if(!block_id[i].empty()){
				block = block_id[i].peek(nullptr);
				if(block_done <= block) continue;
				else {
					block_id[i].read(nullptr);
	// 				if(cache_index[block%4] == block){
	// 					// read from bram
	// load_cache:
	// 					for(int j = (block%4)*WINDOW_SIZE_div_16; j < (block%4+1)*WINDOW_SIZE_div_16;){
	// #pragma HLS pipeline II=1
	// #pragma HLS loop_tripcount min=0 max=1024
	// 						if(!fifo_x_out[i].full()){
	// 							fifo_x_out[i].try_write(local_cache_x[j]);
	// 							j++;
	// 						}
	// 					}
	// 				} else {
						// cache_index[block%4] = block;
	load_x_to_cache:
						for(int i_req = 0, i_resp = 0; i_resp < WINDOW_SIZE_div_16;){
	#pragma HLS pipeline II=1
	#pragma HLS loop_tripcount min=0 max=1024
							if(i_req < WINDOW_SIZE/16 && !mmap.read_addr.full()){
								mmap.read_addr.try_write(i_req+block*WINDOW_SIZE_div_16);
								++i_req;
							}
							if(!mmap.read_data.empty() && !fifo_x_out[i].full()){
								float_v16 tmp;
								mmap.read_data.try_read(tmp);
								fifo_x_out[i].try_write(tmp);
								// local_cache_x[(block%4)*WINDOW_SIZE_div_16+i_resp] = tmp;
								++i_resp;
							}
						}
					// }
				}
			}
		}
	}
}

// void SolverMiddleware( int pe_i, int N,
// 	tapa::mmap<ap_uint<512>> csr_edge_list_ch,
// 	tapa::mmap<int> csr_edge_list_ptr,
// 	tapa::mmap<float_v16> f,
// 	tapa::mmap<ap_uint<512>> dep_graph_ch,
// 	tapa::mmap<int> dep_graph_ptr,
// 	tapa::istream<float>& x_q_in,
// 	tapa::ostream<float>& x_q_out,
// 	tapa::istream<int>& N_val,
// 	tapa::ostream<int>& block_id,
// 	tapa::istream<float>& req_x){
		
// 		tapa::stream<ap_uint<512>> dep_graph_ch_q("dep_graph_ch_q");
// 		tapa::stream<int> dep_graph_ptr_q("dep_graph_ptr_q");
// 		tapa::stream<float_v16> f_q("f");
// 		tapa::stream<ap_uint<512>> spmv_val("spmv_val");
// 		tapa::stream<int> spmv_inst("spmv_inst");
// 		tapa::streams<int, 3> N_sub("n_sub");
// 		tapa::stream<float> y("y");
// 		tapa::stream<float, WINDOW_SIZE * NUM_CH> x_next("x_next");
// 		tapa::stream<float, WINDOW_SIZE * NUM_CH> x_apt("x_apt");
// 		tapa::stream<float> x_bypass("x_bypass");

// 		tapa::task()
// 			.invoke<tapa::join>(read_float_vec, pe_i, N, f, N_val, N_sub, f_q)
// 			.invoke<tapa::join>(read_A_and_split, pe_i, N, csr_edge_list_ch, csr_edge_list_ptr, spmv_val, spmv_inst)
// 			.invoke<tapa::join>(read_dep_graph, pe_i, N, dep_graph_ch, dep_graph_ptr, dep_graph_ch_q, dep_graph_ptr_q)
// 			.invoke<tapa::join>(X_bypass, pe_i, N, x_q_in, x_apt, x_bypass)
// 			.invoke<tapa::join>(PEG_Xvec, pe_i, N, spmv_inst, spmv_val, y, N_sub, N_sub, block_id, req_x)
// 			.invoke<tapa::detach>(solve, dep_graph_ch_q, dep_graph_ptr_q, f_q, y, x_next, N_sub, N_sub)
// 			.invoke<tapa::join>(X_Merger, pe_i, N, x_apt, x_next, x_bypass, x_q_out, N_sub);
// 	}

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> csr_edge_list_ch,
			tapa::mmaps<int, NUM_CH> csr_edge_list_ptr,
			tapa::mmaps<ap_uint<512>, NUM_CH> dep_graph_ch,
			tapa::mmaps<int, NUM_CH> dep_graph_ptr,
			tapa::mmaps<float_v16, NUM_CH> f, 
			tapa::mmap<float_v16> x, 
			int N // # of dimension
			){

	tapa::streams<int, NUM_CH> N_val("n_val");
	tapa::streams<float, NUM_CH+1> x_q("x_pipe");
	tapa::stream<bool> fin_write("fin_write");
	tapa::streams<int, NUM_CH> block_id("block_id");
	tapa::streams<float_v16, NUM_CH> req_x("req_x");

	tapa::streams<ap_uint<512>, NUM_CH> dep_graph_ch_q("dep_graph_ch_q");
	tapa::streams<int, NUM_CH> dep_graph_ptr_q("dep_graph_ptr_q");
	tapa::streams<int, NUM_CH> dep_graph_inst("dep_graph_inst");
	tapa::streams<float_v16, NUM_CH> f_q("f");
	tapa::streams<ap_uint<512>, NUM_CH> spmv_val("spmv_val");
	tapa::streams<int, NUM_CH> spmv_inst("spmv_inst");
	tapa::streams<int, NUM_CH> spmv_inst_2("spmv_inst_2");
	tapa::streams<int, NUM_CH> N_sub_1("n_sub_1");
	tapa::streams<int, NUM_CH> N_sub_2("n_sub_2");
	tapa::streams<int, NUM_CH> N_sub_3("n_sub_3");
	tapa::streams<int, NUM_CH> N_sub_4("n_sub_4");
	tapa::streams<float_v16, NUM_CH> y("y");
	tapa::streams<float_v16, NUM_CH> y_update("y_update");
	tapa::streams<float, NUM_CH, WINDOW_SIZE * NUM_CH> x_next("x_next");
	tapa::streams<float, NUM_CH, WINDOW_SIZE * NUM_CH> x_apt("x_apt");
	tapa::streams<float, NUM_CH> x_bypass("x_bypass");
	tapa::stream<float_v16> x_vec("x_vec");
	tapa::streams<MultXVec, NUM_CH> fifo_aXVec("fifo_aXVec");

	tapa::task()
		.invoke<tapa::join>(read_len, N, N_val)
		.invoke<tapa::join>(fill_zero, x_q)
		.invoke<tapa::detach>(request_X, x, block_id, req_x, fin_write)
		// .invoke<tapa::join, NUM_CH>(SolverMiddleware, tapa::seq(), N, csr_edge_list_ch, csr_edge_list_ptr, f, dep_graph_ch, dep_graph_ptr, x_q, x_q, N_val, block_id, req_x)
		.invoke<tapa::join, NUM_CH>(read_float_vec, tapa::seq(), N, f, N_val, N_sub_1, f_q)
		.invoke<tapa::join, NUM_CH>(read_A_and_split, tapa::seq(), N, csr_edge_list_ch, csr_edge_list_ptr, spmv_val, spmv_inst)
		.invoke<tapa::join, NUM_CH>(read_dep_graph_ptr, tapa::seq(), N, dep_graph_ptr, dep_graph_ptr_q)
		.invoke<tapa::join, NUM_CH>(read_dep_graph, tapa::seq(), N, dep_graph_ch, dep_graph_ptr_q, dep_graph_ch_q, dep_graph_inst)
		.invoke<tapa::join, NUM_CH>(X_bypass, tapa::seq(), N, x_q, x_apt, x_bypass)
		.invoke<tapa::join, NUM_CH>(PEG_Xvec, tapa::seq(), N, spmv_inst, spmv_inst_2, spmv_val, fifo_aXVec, block_id, req_x)
		.invoke<tapa::join, NUM_CH>(PEG_YVec, tapa::seq(), N, spmv_inst_2, fifo_aXVec, y, N_sub_1, N_sub_2)
		.invoke<tapa::detach, NUM_CH>(Yvec_minus_f, f_q, y, y_update, N_sub_2, N_sub_3)
		.invoke<tapa::detach, NUM_CH>(solve, dep_graph_ch_q, dep_graph_inst, y_update, x_next, N_sub_3, N_sub_4)
		.invoke<tapa::join, NUM_CH>(X_Merger, tapa::seq(), N, x_apt, x_next, x_bypass, x_q, N_sub_4)
		.invoke<tapa::join>(float_to_float_vec, N, x_q, x_vec)
		.invoke<tapa::join>(write_x, x_vec, fin_write, x, N);
}
