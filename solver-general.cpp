#include <ap_int.h>
#include <cstdint>
#include <tapa.h>
// #include <glog/logging.h>

constexpr int WINDOW_SIZE = 512;
constexpr int NUM_CH = 6;

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;

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

void write_x(tapa::istream<float>& x, tapa::ostream<bool>& fin_write, tapa::async_mmap<float>& mmap, int total_N){
	int level = total_N / (WINDOW_SIZE*NUM_CH);
	for(int i = 0; i < level; i++){
		//LOG(INFO) << "process level " << i << " ...";
		int start = WINDOW_SIZE*NUM_CH*i;
		int N = WINDOW_SIZE*NUM_CH;
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
	int residue = total_N %(WINDOW_SIZE * NUM_CH);
	for(int i_req = 0, i_resp = 0; i_resp < residue;){
		if(i_req < residue && !x.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
			mmap.write_addr.try_write(i_req + total_N - residue);
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

void read_float_vec(int pe_i, int total_N, tapa::async_mmap<float>& mmap, tapa::istream<int>& len_in, tapa::ostream<int>& len_out,
                 tapa::ostream<float>& stream) {
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
						float tmp;
						mmap.read_data.try_read(tmp);
						stream.try_write(tmp);
						++i_resp;
				}
		}
		offset+=N;
	}
}

//TODO: replace with serpen after modifying the read width
void PEG_Xvec( int pe_i, int total_N,
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<ap_uint<64>>& spmv_A, // 12-bit row + 20-bit col + 32-bit float
	tapa::istream<float>& fifo_x_in,
	tapa::ostream<float>& fifo_x_out,
	tapa::ostream<float>& fifo_y_out,
	tapa::istream<int>& N_in, tapa::ostream<int>& N_out,
	tapa::ostream<int>& block_id, tapa::istream<float>& req_x){

		int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
		int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

round:
		for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off
			const int num_x = N_in.read();
			N_out.write(num_x);

			float local_res[WINDOW_SIZE];
			float local_x[4][WINDOW_SIZE];

#pragma HLS bind_storage variable=local_x latency=2
#pragma HLS array_partition variable=local_x complete dim=1
#pragma HLS array_partition variable=local_x cyclic factor=8 dim=2
#pragma HLS array_partition variable=local_res cyclic factor=4

			//reset
reset:
			for(int i = 0; i < num_x; i++){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=512
				local_res[i] = 0.0;
			}

load_x:
			for(int ite = 0; ite < pe_i+round*NUM_CH; ite++){
				const int N = fifo_inst_in.read();
				//read x
				bool x_val_succ = false;
				float x_val = 0.0;
				if(ite < round * NUM_CH && N != 0){
					block_id.write(ite);
					for(int i = 0; i < WINDOW_SIZE;){
						x_val = req_x.read(x_val_succ);
						if(x_val_succ){
							for(int k = 0; k < 4; k++){
								local_x[k][i] = x_val;
							}
							x_val_succ = false;
							i++;
						}
					}
				}else if(ite >= round * NUM_CH){
					for(int i = 0; i < WINDOW_SIZE;){
						x_val = fifo_x_in.read(x_val_succ);
						if(x_val_succ){
							fifo_x_out.write(x_val);
							for(int k = 0; k < 4; k++){
								local_x[k][i] = x_val;
							}
							x_val_succ = false;
							i++;
						}
					}
				}

				//read A and compute
				bool a_val_succ = false;

compute:
				for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=512
					ap_uint<96> a = spmv_A.read(a_val_succ);
					if(a_val_succ){
						ap_uint<12> a_row = a(63, 52);
						ap_uint<20> a_col = a(51, 32);
						ap_uint<32> a_val = a(31, 0);
						float a_val_f = tapa::bit_cast<float>(a_val);
						local_res[a_row] += local_x[i%4][a_col - ite * WINDOW_SIZE] * a_val_f;
						a_val_succ = false;
						i++;
					}
				}
			}

write_y:
			//write result
			for(int i = 0; i < num_x; i++){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=512
				fifo_y_out.write(local_res[i]);
			}
		}
		
	}

void solve_v2(tapa::istream<int>& next,
		tapa::ostream<int>& ack,
		tapa::istream<int>& ia,
		tapa::istream<int>& ja,
		tapa::istream<float>& a,
		tapa::istream<float>& f,
		tapa::istream<float>& spmv_in,
		tapa::ostream<float>& x_out, 
		tapa::istream<int>& N_in,
		tapa::ostream<int>& N_out){

for(;;){
#pragma HLS loop_flatten off
	const int N = N_in.read();
	N_out.write(N);

	float local_x[WINDOW_SIZE];
	float local_a[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
	int local_ia[WINDOW_SIZE];
	int local_ja[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
	float cyclic_aggregate[16];

#pragma HLS bind_storage variable=local_a type=RAM_2P impl=URAM
#pragma HLS array_partition cyclic variable=local_x factor=4
#pragma HLS array_partition cyclic variable=local_a factor=4
#pragma HLS array_partition cyclic variable=local_ia factor=4
#pragma HLS array_partition cyclic variable=local_ja factor=12
#pragma HLS array_partition complete variable=cyclic_aggregate

	int ia_val = 0;
	float f_val = 0.f;
	float a_val = 0.f;
	int ja_val = 0;
	float spmv_in_val = 0.0;

	bool ia_succ = false, ja_succ = false, a_succ = false, f_succ = false, spmv_succ = false;

	int prev_ia = 0;

read_ia_and_f:
	for(int i = 0; i < N;) {
#pragma HLS loop_tripcount min=1 max=256
#pragma HLS pipeline II=1
		if(!ia_succ) ia_val = ia.read(ia_succ);
		if(!f_succ) f_val = f.read(f_succ);
		if(ia_succ && f_succ){
			local_ia[i] = ia_val;
			local_x[i] = f_val;
			for(int j = prev_ia; j < ia_val;){
#pragma HLS loop_tripcount min=1 max=40000
#pragma HLS pipeline II=1
				if(!ja_succ) ja_val = ja.read(ja_succ);
				if(!a_succ) a_val = a.read(a_succ);
				if(ja_succ && a_succ){
					local_a[j] = a_val;
					local_ja[j] = ja_val;
					a_succ = ja_succ = false;
					j++;
				}
			}
			prev_ia = ia_val;
			ia_succ = f_succ = false;
			i++;
		}
	}

read_spmv_and_subtract:
	for(int i = 0; i < N;){
#pragma HLS loop_tripcount min=1 max=256
#pragma HLS dependence variable=local_x false
#pragma HLS pipeline II=1
		spmv_in_val = spmv_in.read(spmv_succ);
		if(spmv_succ){
			local_x[i] -= spmv_in_val;
			spmv_succ = false;
			i++;
		}
	}
	
	bool next_succ = false;
	int index = 0;

compute:
        for(int i = 0; i < N;){
#pragma HLS loop_tripcount min=1 max=256
		if(!next_succ) index = next.read(next_succ);
		if(next_succ) {
			int start = (index == 0) ? 0:local_ia[index-1];
			int end = local_ia[index]-1;
reset:
			for(int i = 0; i < 16; i++){
				cyclic_aggregate[i] = 0.0;
			}
accumulate:
			for(int i = start; i < end; ++i){
#pragma HLS loop_tripcount min=1 max=256
#pragma HLS dependence variable=cyclic_aggregate false
#pragma HLS dependence variable=cyclic_aggregate type=inter distance=16 true
#pragma HLS pipeline II=1
				int c = local_ja[i];
				cyclic_aggregate[i%16] += (local_x[c] * local_a[i]);
			}
aggregate:
			for(int i = 1; i < 16; ++i){
#pragma HLS pipeline II=1
				cyclic_aggregate[0] += cyclic_aggregate[i];
			}
			local_x[index] -= cyclic_aggregate[0]; 
			local_x[index] /= local_a[end];
			ack.write(index);
			next_succ = false;
			i++;
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

void analyze(tapa::istream<int>& ia,
		tapa::istream<int>& csc_col_ptr,
		tapa::istream<int>& csc_row_ind,
		tapa::istream<int>& ack,
		tapa::ostream<int>& next,
		tapa::istream<int>& N_in, tapa::ostream<int>& N_out){

	for(;;){
#pragma HLS loop_flatten off
		const int N = N_in.read();
		N_out.write(N);

		int parents[WINDOW_SIZE];
		int local_csc_col_ptr[WINDOW_SIZE];
		int local_csc_row_ind[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
		int next_queue[WINDOW_SIZE];

#pragma HLS bind_storage variable=local_csc_row_ind type=RAM_2P impl=URAM
#pragma HLS array_partition variable=parents cyclic factor=4
#pragma HLS array_partition variable=local_csc_col_ptr cyclic factor=4
#pragma HLS array_partition variable=local_csc_row_ind cyclic factor=4
#pragma HLS array_partition variable=next_queue cyclic factor=4

        int num_nn = 0;
        int ia_val = 0;
		int csc_col_val = 0;
		int csc_row_val = 0;
		int start = 0, end = 0;

        bool ia_succ = false, csc_col_succ = false, csc_row_succ = false;

compute_parents:
        for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=256
                if(!ia_succ) ia_val = ia.read(ia_succ);
                if(ia_succ){
                	parents[i] = ia_val - num_nn - 1;
                	num_nn = ia_val;
                	if(parents[i] == 0) {
						next_queue[end] = i;
						end++;
					}
                	ia_succ = false;
                	i++;
                }
        }


	int prev_csc_col = 0;

read_csc_col:
	for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=256
		if(!csc_col_succ) csc_col_val = csc_col_ptr.read(csc_col_succ);
		if(csc_col_succ){
			local_csc_col_ptr[i] = csc_col_val;
			for(int j = prev_csc_col; j < csc_col_val;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=40000
				if(!csc_row_succ) csc_row_val = csc_row_ind.read(csc_row_succ);
				if(csc_row_succ){
						local_csc_row_ind[j] = csc_row_val;
						csc_row_succ = false;
						j++;
				}

			}
			prev_csc_col = csc_col_val;
			csc_col_succ = false;
			i++;
		}
	}

	int processed = 0;
	bool ack_succ = false;
	int ack_val = 0;

compute:
	while(processed < N){
#pragma HLS loop_tripcount min=1 max=256
		if(start < end && !next.full()){
			next.try_write(next_queue[start++]);
		}
		if(!ack_succ) ack_val = ack.read(ack_succ);
	    if(ack_succ){
			int start = (ack_val == 0) ? 0 : local_csc_col_ptr[ack_val-1];
remove_dependency:
			for(int i = start; i < local_csc_col_ptr[ack_val]; i++){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=256
				int index = local_csc_row_ind[i];
				parents[index]--;
				if(parents[index] == 0) next_queue[end++] = index;
			}
			ack_succ = false;
			processed++;
		}	
	}
		}
}

void PEG_split(int pe_i, int total_N, tapa::istream<ap_uint<64>>& fifo_A_ch,
			tapa::istream<int>& fifo_A_ptr,
			tapa::istream<int>& csc_col_ptr,
			tapa::istream<int>& csc_row_ind,
			tapa::ostream<int>& solver_row_ptr_a,
			tapa::ostream<int>& solver_row_ptr_b,
			tapa::ostream<int>& solver_col_ind,
			tapa::ostream<float>& solver_val,
			/* for csc mapping search*/
			tapa::ostream<int>& solver_col_ptr,
			tapa::ostream<int>& solver_row_ind,
			tapa::istream<int>& N_in,
			tapa::ostream<int>& N_out){

	int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
	int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;

		for(int round = 0; round < bound; round++){
#pragma HLS loop_flatten off
				const int N = N_in.read();
				N_out.write(N);

				float solver_val_arr[WINDOW_SIZE];
				int solver_col_ind_arr[WINDOW_SIZE];
				int solver_row_ind_arr[WINDOW_SIZE];

				const int K = fifo_A_ptr.read();
				int prev_row = -1;
				int row_ptr = 0;
				int num_one_row = 0;
				bool fifo_A_succ = false;
				ap_uint<96> a_entry;
				for(int i = 0; i < K;){
					a_entry = fifo_A_ch.read(fifo_A_succ);
					if(fifo_A_succ){
						ap_uint<12> row = a_entry(63, 52);
						ap_uint<20> col = a_entry(51, 32);
						ap_uint<32> val = a_entry(31, 0);
						float val_f = tapa::bit_cast<float>(val);
						//if(round == 0 && pe_i == 0) LOG(INFO) << "(" << (int)row << "," << (int) col << "): " << val_f;
						if(row != (prev_row & 0xFFF)){
							if(prev_row != -1){
								solver_row_ptr_a.write(row_ptr);
								solver_row_ptr_b.write(row_ptr);
								for(int k = 0; k < num_one_row; k++){
									solver_val.write(solver_val_arr[k]);
									solver_col_ind.write(solver_col_ind_arr[k]);
								}
								num_one_row = 0;
							}
							prev_row = (int) 0 | row;
						}
						solver_val_arr[num_one_row] = val_f;
						solver_col_ind_arr[num_one_row] = tapa::bit_cast<int>(col)-(pe_i+NUM_CH*round)*WINDOW_SIZE;
						num_one_row++;
						row_ptr++;
						fifo_A_succ = false;
						i++;
					}
				}
				solver_row_ptr_a.write(row_ptr);
				solver_row_ptr_b.write(row_ptr);
				for(int k = 0; k < num_one_row; k++){
					solver_val.write(solver_val_arr[k]);
					solver_col_ind.write(solver_col_ind_arr[k]);
				}

				int end_row = (pe_i+round*NUM_CH) * WINDOW_SIZE + N;

				int csc_col_val = 0, csc_row_val = 0;
				int prev = 0;
				bool csc_col_ptr_succ = false, csc_row_ind_succ = false;
				int solver_col_ptr_tmp = 0;
				int num_one_col = 0;
				for(int i = 0; i < N;){
					csc_col_val = csc_col_ptr.read(csc_col_ptr_succ);
					if(csc_col_ptr_succ) {
						for(int j = 0; j < csc_col_val-prev;){
							csc_row_val = csc_row_ind.read(csc_row_ind_succ);
							if(csc_row_ind_succ){
								if(csc_row_val < end_row) {
									solver_row_ind_arr[num_one_col] = csc_row_val-(pe_i+round*NUM_CH)*WINDOW_SIZE;
									num_one_col ++;
									solver_col_ptr_tmp ++;
								}
								csc_row_ind_succ = false;
								j++;
							}
						}
						solver_col_ptr.write(solver_col_ptr_tmp);
						for(int k = 0; k < num_one_col; k++){
							solver_row_ind.write(solver_row_ind_arr[k]);
						}
						num_one_col = 0;
						prev = csc_col_val;
						csc_col_ptr_succ = false;
						i++;
					}
				}
		}
}

void read_A_and_split(int pe_i, int total_N,
	tapa::async_mmap<ap_uint<512>>& csr_edge_list_ch,
	tapa::mmap<int> csr_edge_list_ptr,
	tapa::ostream<ap_uint<64>>& fifo_A_ch,
	tapa::ostream<int>& fifo_A_ptr,
	tapa::ostream<ap_uint<64>>& spmv_val,
	tapa::ostream<int>& spmv_inst){
		int offset = 0;
		int offset_i = 0;
		int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
		int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
round:
		for(int round = 0;round < bound;round++){
#pragma HLS loop_flatten off
traverse_block:
			for(int i = 0; i < pe_i+NUM_CH*round+1; i++){
				const int N = csr_edge_list_ptr[i+offset_i];
				const int num_ite = (N + 7) / 8;
				if(i == pe_i+NUM_CH*round) {
					fifo_A_ptr.write(N);
				}
				else {
					spmv_inst.write(N);
				}
split:
				for (int i_req = 0, i_resp = 0; i_resp < num_ite;) {
					if(i_req < num_ite && !csr_edge_list_ch.read_addr.full()){
							csr_edge_list_ch.read_addr.try_write(i_req+offset);
							++i_req;
					}
					if(!csr_edge_list_ch.read_data.empty()){
						if(i == pe_i+NUM_CH*round){
							ap_uint<512> tmp;
							csr_edge_list_ch.read_data.try_read(tmp);
							for(int index = 0; index < 8; index++){
								ap_uint<64> tmp_o = tmp(index * 64 + 63, index * 64);
								if(tmp_o(63,52) ^ 0xFFF){
									fifo_A_ch.write(tmp_o);
								}
							}
							++i_resp;
						}else{
							ap_uint<512> tmp;
							csr_edge_list_ch.read_data.try_read(tmp);
							for(int index = 0; index < 8; index++){
								ap_uint<64> tmp_o = tmp(index * 64 + 63, index * 64);
								if(tmp_o(63,52) ^ 0xFFF){
									spmv_val.write(tmp_o);
								}
							}
							++i_resp;
						}
					}
				}
				offset+=num_ite;
			}
			offset_i += (pe_i+NUM_CH*round+1);
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
			const int num_x = N_in.read();
			for(int i = 0; i < pe_i * WINDOW_SIZE; i++){
				x_out.write(x_first.read());
			}
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

void read_len( tapa::async_mmap<int>& K_csc, 
	int N, 
	tapa::ostreams<int, NUM_CH>& K_csc_val, 
	tapa::ostreams<int, NUM_CH>& N_val){
	int bound = (N%WINDOW_SIZE == 0) ? N/WINDOW_SIZE : N/WINDOW_SIZE+1;
	for(int i_req = 0, i_resp = 0; i_resp < bound;){
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
		int len = ((N-i_resp*WINDOW_SIZE) < WINDOW_SIZE) ? N%WINDOW_SIZE : WINDOW_SIZE;
		int ind = i_resp%NUM_CH;
		if(i_req < bound && !K_csc.read_addr.full()){
                K_csc.read_addr.try_write(i_req);
                i_req++;
        }
        if(!K_csc.read_data.empty()){
                int tmp;
				K_csc.read_data.try_read(tmp);
                K_csc_val[ind].write(tmp);
				N_val[ind].write(len);
                i_resp++;
        }
	}
}

void fill_zero(tapa::ostream<float>& fifo_out){
	fifo_out.try_write(0.0);
}

void request_X(tapa::async_mmap<float>& mmap, tapa::istreams<int, NUM_CH>& block_id, tapa::ostreams<float, NUM_CH>& fifo_x_out, tapa::istream<bool>& fin_write){
	float local_cache_x[WINDOW_SIZE*4];
	int cache_index[4];

#pragma HLS array_partition variable=local_cache_x cyclic factor=8
#pragma HLS array_partition variable=cache_index complete

	int block = 0;
	bool read_succ = false;
	int level_done = 0;

init_cache_tag:
	for(int i = 0; i < 4; i++) {
#pragma HLS pipeline II=1
		cache_index[i] = -1;
	}

handle_request:
	for(;;){
#pragma HLS loop_flatten off
		if(!fin_write.empty()){
			bool last = fin_write.read(nullptr);
			if(last){
				level_done++;
			}else{
				for(int i = 0; i < 4; i++) {
					cache_index[i] = -1;
				}
				level_done = 0;
			}
		}
scan:
		for(int i = 0; i < NUM_CH; i++){
			block = block_id[i].read(read_succ);
			if(read_succ){
				if(cache_index[block%4] == block){
					// read from bram
load_cache:
					for(int j = 0; j < WINDOW_SIZE;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=0 max=1024
						if(!fifo_x_out[i].full()){
							float x_f = local_cache_x[(block%4)*WINDOW_SIZE+j];
							fifo_x_out[i].try_write(x_f);
							j++;
						}
					}
				} else {
					if(level_done*NUM_CH <= block){
						fin_write.read();
						level_done++;
					}
					cache_index[block%4] = block;
load_x_to_cache:
					for(int i_req = 0, i_resp = 0; i_resp < WINDOW_SIZE;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=0 max=1024
						if(i_req < WINDOW_SIZE && !mmap.read_addr.full()){
							mmap.read_addr.try_write(i_req+block*WINDOW_SIZE);
							++i_req;
						}
						if(!mmap.read_data.empty() && !fifo_x_out[i].full()){
							float tmp;
							mmap.read_data.try_read(tmp);
							fifo_x_out[i].try_write(tmp);
							local_cache_x[(block%4)*WINDOW_SIZE+i_resp] = tmp;
							++i_resp;
						}
					}
				}
				read_succ = false;
			}
		}
	}
}

void SolverMiddleware( int pe_i, int N,
	tapa::mmap<ap_uint<512>> csr_edge_list_ch,
	tapa::mmap<int> csr_edge_list_ptr,
	tapa::mmap<float> f,
	tapa::mmap<int> csc_col_ptr,
	tapa::mmap<int> csc_row_ind,
	tapa::istream<float>& x_q_in,
	tapa::ostream<float>& x_q_out,
	tapa::istream<int>& N_val,
	tapa::istream<int>& K_csc_val,
	tapa::ostream<int>& block_id,
	tapa::istream<float>& req_x){
		
		tapa::stream<ap_uint<64>> fifo_A_ch("fifo_A_ch");
		tapa::stream<int> fifo_A_ptr("fifo_A_ptr");
		tapa::stream<int> csc_col_ptr_q("csc_col_ptr");
		tapa::stream<int> csc_row_ind_q("csc_row_ind");
		tapa::stream<float> f_q("f");
		tapa::stream<ap_uint<64>> spmv_val("spmv_val");
		tapa::stream<int> spmv_inst("spmv_inst");
		tapa::stream<int> solver_row_ptr_a("solver_row_ptr_a");
		tapa::stream<int> solver_row_ptr_b("solver_row_ptr_b");
		tapa::stream<int> solver_col_ind("solver_col_ind");
		tapa::stream<float> solver_val("solver_val");
		/* for csc mapping search*/
		tapa::stream<int> solver_col_ptr("solver_col_ptr");
		tapa::stream<int> solver_row_ind("solver_row_ind");
		tapa::stream<int> next("next");
    	tapa::stream<int> ack("ack");

		tapa::stream<int> K_sub_csc("k_sub_csc");
		tapa::streams<int, 6> N_sub("n_sub");
		tapa::stream<float> y("y");
		tapa::stream<float, WINDOW_SIZE * NUM_CH> x_prev("x_prev");
		tapa::stream<float> x_next("x_next");
		tapa::stream<float> x_apt("x_apt");
		tapa::stream<float> x_bypass("x_bypass");

		tapa::task()
			.invoke<tapa::join>(read_float_vec, pe_i, N, f, N_val, N_sub, f_q)
			.invoke<tapa::join>(read_int_vec, pe_i, N, csc_col_ptr, N_sub, N_sub, csc_col_ptr_q)
		    .invoke<tapa::join>(read_int_vec, pe_i, N, csc_row_ind, K_csc_val, K_sub_csc, csc_row_ind_q)
			.invoke<tapa::detach>(black_hole_int, K_sub_csc)
			.invoke<tapa::join>(read_A_and_split, pe_i, N, csr_edge_list_ch, csr_edge_list_ptr, fifo_A_ch, fifo_A_ptr, spmv_val, spmv_inst)
			.invoke<tapa::join>(X_bypass, pe_i, N, x_q_in, x_apt, x_bypass)
			.invoke<tapa::join>(PEG_Xvec, pe_i, N, spmv_inst, spmv_val, x_apt, x_prev, y, N_sub, N_sub, block_id, req_x)
			.invoke<tapa::join>(PEG_split, pe_i, N,
				fifo_A_ch,
				fifo_A_ptr,
				csc_col_ptr_q,  
				csc_row_ind_q, 
				solver_row_ptr_a,
				solver_row_ptr_b,
				solver_col_ind,
				solver_val,
				solver_col_ptr,
				solver_row_ind,
				N_sub, N_sub)
			.invoke<tapa::detach>(analyze, solver_row_ptr_a, solver_col_ptr, solver_row_ind, ack, next, N_sub, N_sub) // ?
			.invoke<tapa::detach>(solve_v2, next, ack, solver_row_ptr_b, solver_col_ind, solver_val, f_q, y, x_next, N_sub, N_sub)
			.invoke<tapa::join>(X_Merger, pe_i, N, x_prev, x_next, x_bypass, x_q_out, N_sub);
	}

void TrigSolver(tapa::mmaps<ap_uint<512>, NUM_CH> csr_edge_list_ch,
			tapa::mmaps<int, NUM_CH> csr_edge_list_ptr,
			tapa::mmaps<int, NUM_CH> csc_col_ptr, 
			tapa::mmaps<int, NUM_CH> csc_row_ind, 
			tapa::mmaps<float, NUM_CH> f, 
			tapa::mmap<float> x, 
			int N, // # of dimension
			tapa::mmap<int> K_csc){

	tapa::streams<int, NUM_CH> K_csc_val("k_csc_val");
	tapa::streams<int, NUM_CH> N_val("n_val");
	tapa::streams<float, NUM_CH+1> x_q("x_pipe");
	tapa::stream<bool> fin_write("fin_write");
	tapa::streams<int, NUM_CH> block_id("block_id");
	tapa::streams<float, NUM_CH> req_x("req_x");

	tapa::task()
		.invoke<tapa::join>(read_len, K_csc, N, K_csc_val, N_val)
		.invoke<tapa::join>(fill_zero, x_q)
		.invoke<tapa::detach>(request_X, x, block_id, req_x, fin_write)
		.invoke<tapa::join, NUM_CH>(SolverMiddleware, tapa::seq(), N, csr_edge_list_ch, csr_edge_list_ptr, f, csc_col_ptr, csc_row_ind, x_q, x_q, N_val, K_csc_val, block_id, req_x)
		.invoke<tapa::join>(write_x, x_q, fin_write, x, N);
}
