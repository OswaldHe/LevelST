#include <ap_int.h>
#include <cstdint>
#include <tapa.h>
#include <glog/logging.h>

constexpr int WINDOW_SIZE = 256;
constexpr int NUM_CH = 8;

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;
using float_v8 = tapa::vec_t<float, 8>;
using float_v2 = tapa::vec_t<float, 2>;

void read_int_vec(tapa::async_mmap<int>& mmap, tapa::istream<int>& len_in, tapa::ostream<int>& len_out,
                 tapa::ostream<int>& stream) {
	for(int round = 0;;){
		const int N = len_in.read();
		len_out.write(N);
  for (int i_req = 0, i_resp = 0; i_resp < N;) {
        if(i_req < N && !mmap.read_addr.full()){
                mmap.read_addr.try_write(i_req+round);
                ++i_req;
        }
        if(!mmap.read_data.empty() && !stream.full()){
                int tmp;
                mmap.read_data.try_read(tmp);
				stream.try_write(tmp);
                ++i_resp;
        }
  }
		round += N;
	}
}

void duplicate_int_vec(tapa::async_mmap<int>& mmap, tapa::istream<int>& len_in, tapa::ostream<int>& len_out,
                 tapa::ostream<int>& stream_a, tapa::ostream<int>& stream_b){
	const int N = len_in.read();
	len_out.write(N);
  for (int i_req = 0, i_resp = 0; i_resp < N;) {
        if(i_req < N && !mmap.read_addr.full()){
                mmap.read_addr.try_write(i_req);
                ++i_req;
        }
        if(!mmap.read_data.empty() && !stream_a.full() && !stream_b.full()){
                int tmp;
                mmap.read_data.try_read(tmp);
                stream_a.try_write(tmp);
				stream_b.try_write(tmp);
                ++i_resp;
        }
  }

}

inline void write_float_vec(tapa::istream<float> stream, tapa::async_mmap<float>& mmap,
                 int N, int start) {
#pragma HLS inline
  for(int i_req = 0, i_resp = 0; i_resp < N;){
  	if(i_req < N && !stream.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
		mmap.write_addr.try_write(i_req + start);
		mmap.write_data.try_write(stream.read(nullptr));
		++i_req;
	}
	if(!mmap.write_resp.empty()){
		i_resp += unsigned(mmap.write_resp.read(nullptr))+1;
	}
  }
}

void write_x(tapa::istream<float>& x, tapa::ostream<float>& dump, tapa::ostream<bool>& fin_write, tapa::async_mmap<float>& mmap, int N){
	int level = (N%WINDOW_SIZE == 0)?N/WINDOW_SIZE:N/WINDOW_SIZE+1;
	for(int i = 0; i < level/NUM_CH; i++){
		for(int j = 0; j < WINDOW_SIZE*NUM_CH*i; j++){
			dump.write(x.read());
		}
		write_float_vec(x, mmap, WINDOW_SIZE*NUM_CH, WINDOW_SIZE*NUM_CH*i);
		fin_write.write(true);
	}
}

void read_float_vec(tapa::async_mmap<float>& mmap, tapa::istream<int>& len_in, tapa::ostream<int>& len_out,
                 tapa::ostream<float>& stream) {
	for(int round = 0;;){
		const int N = len_in.read();
		len_out.write(N);
  for (int i_req = 0, i_resp = 0; i_resp < N;) {
        if(i_req < N && !mmap.read_addr.full()){
                mmap.read_addr.try_write(i_req+round);
                ++i_req;
        }
        if(!mmap.read_data.empty() && !stream.full()){
                float tmp;
                mmap.read_data.try_read(tmp);
                stream.try_write(tmp);
                ++i_resp;
        }
  }
		round+=N;
	}
}

void solve(tapa::istream<float>& a, 
		tapa::istream<int>& ia,
		tapa::istream<int>& ja,
		tapa::istream<float>& f,
		tapa::ostream<float>& x, int N){
	
	int read = 0;
	int num_nn = 0;
	int diff = 0;
	float f_val = 0.f;
	float a_val = 0.f;
	int ja_val = 0;
	int tmp = 0;
	float local_x[WINDOW_SIZE];

	bool ia_succ = false, f_succ = false, ja_succ = false, a_succ = false;

	while(read < N){
		if(!ia_succ) tmp = ia.read(ia_succ);
		if(!f_succ) f_val = f.read(f_succ);
		if(ia_succ && f_succ){
		diff = tmp - num_nn;
		num_nn = tmp;
		for(int i = 0; i < diff;){
			if(!ja_succ) ja_val = ja.read(ja_succ);
			if(!a_succ) a_val = a.read(a_succ);
			if(a_succ && ja_succ){
				if(i != diff-1){
					f_val -= (local_x[ja_val] * a_val);
				} else {
					local_x[ja_val] = (f_val / a_val);
				}
				a_succ = ja_succ = false;
				i++;
			}
		}
		ia_succ = f_succ = false;
		read++;
		}
	}

	for(int i = 0; i < N; i++){
		x.write(local_x[i]);
	}

}

//TODO: replace with serpen after modifying the read width
void PEG_Xvec( int pe_i,
	tapa::istream<int>& fifo_inst_in,
	tapa::istream<ap_uint<96>>& spmv_A,
	tapa::istream<float>& fifo_x_in,
	tapa::ostream<float>& fifo_x_out,
	tapa::ostream<float>& fifo_y_out,
	tapa::istream<int>& N_in, tapa::ostream<int>& N_out){
		float local_res[WINDOW_SIZE];
		float local_x[WINDOW_SIZE];

		for(int round = 0;; round++){
			const int num_x = N_in.read();
			N_out.write(num_x);
			//reset
			for(int i = 0; i < num_x; i++){
				local_res[i] = 0.0;
			}

			for(int ite = 0; ite < pe_i+round*NUM_CH; ite++){
				const int N = fifo_inst_in.read();
				//read x
				bool x_val_succ = false;
				float x_val = 0.0;
				for(int i = 0; i < WINDOW_SIZE;){
					x_val = fifo_x_in.read(x_val_succ);
					if(x_val_succ){
						fifo_x_out.write(x_val);
						local_x[i] = x_val;
						x_val_succ = false;
						i++;
					}
				}

				//read A and compute
				bool a_val_succ = false;
				for(int i = 0; i < N;){
					ap_uint<96> a = spmv_A.read(a_val_succ);
					if(a_val_succ){
						ap_uint<32> a_row = a(95, 64);
						ap_uint<32> a_col = a(63, 32);
						ap_uint<32> a_val = a(31, 0);
						float a_val_f = tapa::bit_cast<float>(a_val);
						local_res[a_row] += local_x[a_col - ite * WINDOW_SIZE] * a_val_f;
						a_val_succ = false;
						i++;
					}
				}
			}

			//write result
			for(int i = 0; i < num_x; i++){
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
		tapa::ostream<int>& N_out,
		tapa::istream<int>& k_solver_in){
	
	float local_x[WINDOW_SIZE];
	float local_a[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
	int local_ia[WINDOW_SIZE];
	int local_ja[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
	float cyclic_aggregate[16];

#pragma HLS bind_storage type=ram_2p impl=bram latency=1
#pragma HLS array_partition cyclic variable=local_ja factor=16
#pragma HLS array_partition complete variable=cyclic_aggregate

for(;;){
	const int N = N_in.read();
	N_out.write(N);

	int ia_val = 0;
	float f_val = 0.f;
	float a_val = 0.f;
	int ja_val = 0;
	float spmv_in_val = 0.0;
	int K = -1;

	bool ia_succ = false, ja_succ = false, a_succ = false, f_succ = false, spmv_succ = false;

read_ja_and_a:
	for(int i = 0;;){
#pragma HLS loop_tripcount min=1 max=40000
#pragma HLS pipeline II=1
		if(!k_solver_in.empty()){
			K = k_solver_in.read();
		}
		if(K != -1 && i >= K) break;
		if(!ja_succ) ja_val = ja.read(ja_succ);
		if(!a_succ) a_val = a.read(a_succ);
		if(ja_succ && a_succ){
			local_a[i] = a_val;
			local_ja[i] = ja_val;
			a_succ = ja_succ = false;
			i++;
		}
	}

read_ia_and_f:
	for(int i = 0; i < N;) {
#pragma HLS loop_tripcount min=1 max=256
#pragma HLS pipeline II=1
		if(!ia_succ) ia_val = ia.read(ia_succ);
		if(!f_succ) f_val = f.read(f_succ);
		if(ia_succ && f_succ){
			local_ia[i] = ia_val;
			local_x[i] = f_val;
			ia_succ = f_succ = false;
			i++;
		}
	}

read_spmv_and_subtract:
	for(int i = 0; i < N;){
#pragma HLS loop_tripcount min=1 max=256
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
reset:
			for(int i = 0; i < 16; i++){
				cyclic_aggregate[i] = 0.0;
			}
accumulate:
			for(int i = start; i < local_ia[index]-1; i++){
#pragma HLS loop_tripcount min=1 max=256
#pragma HLS dependence variable=cyclic_aggregate false
#pragma HLS dependence variable=cyclic_aggregate type=inter distance=16 true
#pragma HLS pipeline II=1
				int c = local_ja[i];
				cyclic_aggregate[i%16] += (local_x[c] * local_a[i]);
			}
aggregate:
			for(int i = 1; i < 16; i++){
#pragma HLS pipeline II=1
				cyclic_aggregate[0] += cyclic_aggregate[i];
			}
			local_x[index] -= cyclic_aggregate[0]; 
			local_x[index] /= local_a[local_ia[index]-1];
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
		tapa::istream<int>& N_in, tapa::ostream<int>& N_out,
		tapa::istream<int>& k_solver_in, tapa::ostream<int>& k_solver_out){

		for(;;){
		const int N = N_in.read();
		N_out.write(N);

        int num_nn = 0;
        int diff = 0;
        int ia_val = 0;
		int csc_col_val = 0;
		int csc_row_val = 0;
		int parents[WINDOW_SIZE];
		int local_csc_col_ptr[WINDOW_SIZE];
		int local_csc_row_ind[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
		int K = -1;

#pragma HLS array_partition variable=parents cyclic factor=4
#pragma HLS array_partition variable=local_csc_col_ptr cyclic factor=4
#pragma HLS array_partition variable=local_csc_row_ind cyclic factor=4

        bool ia_succ = false, csc_col_succ = false, csc_row_succ = false;

compute_parents:
        for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=256
                if(!ia_succ) ia_val = ia.read(ia_succ);
                if(ia_succ){
                	diff = ia_val - num_nn;
                	num_nn = ia_val;
					parents[i] = diff-1;
                	if(parents[i] == 0) next.write(i);
                	ia_succ = false;
                	i++;
                }
        }

read_csc_row:
	for(int i = 0;;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=40000
		if(!k_solver_in.empty()){
			K = k_solver_in.read();
			k_solver_out.write(K);
		}
		if(K != -1 && i >= K) break;
		if(!csc_row_succ) csc_row_val = csc_row_ind.read(csc_row_succ);
		if(csc_row_succ){
				local_csc_row_ind[i] = csc_row_val;
				csc_row_succ = false;
				i++;
		}

	}

read_csc_col:
	for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=256
		if(!csc_col_succ) csc_col_val = csc_col_ptr.read(csc_col_succ);
		if(csc_col_succ){
			local_csc_col_ptr[i] = csc_col_val;
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
		if(!ack_succ) ack_val = ack.read(ack_succ);
	        if(ack_succ){
			int start = (ack_val == 0) ? 0 : local_csc_col_ptr[ack_val-1];
remove_dependency:
			for(int i = start; i < local_csc_col_ptr[ack_val]; i++){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=256
				int index = local_csc_row_ind[i];
				parents[index]--;
				if(parents[index] == 0) next.write(index);
			}
			ack_succ = false;
			processed++;
		}	
	}
		}
}

void PEG_split(int pe_i, tapa::istream<ap_uint<96>>& fifo_A_ch,
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
			tapa::ostream<int>& K_solver,
			tapa::istream<int>& N_in,
			tapa::ostream<int>& N_out){

		for(int round = 0;;round++){
				const int N = N_in.read();
				N_out.write(N);

				const int K = fifo_A_ptr.read();
				K_solver.write(K);
				int prev_row = -1;
				int row_ptr = 0;
				bool fifo_A_succ = false;
				ap_uint<96> a_entry;
				for(int i = 0; i < K;){
					a_entry = fifo_A_ch.read(fifo_A_succ);
					if(fifo_A_succ){
						ap_uint<32> row = a_entry(95, 64);
						ap_uint<32> col = a_entry(63, 32);
						ap_uint<32> val = a_entry(31, 0);
						float val_f = tapa::bit_cast<float>(val);
						// if(round == 1 && pe_i == 0) LOG(INFO) << "(" << (int)row << "," << (int) col << "): " << val_f;
						if(tapa::bit_cast<int>(row) != prev_row){
							if(prev_row != -1){
								solver_row_ptr_a.write(row_ptr);
								solver_row_ptr_b.write(row_ptr);
							}
							prev_row = tapa::bit_cast<int>(row);
						}
						solver_val.write(val_f);
						solver_col_ind.write(tapa::bit_cast<int>(col)-(pe_i+NUM_CH*round)*WINDOW_SIZE);
						row_ptr++;
						fifo_A_succ = false;
						i++;
					}
				}
				solver_row_ptr_a.write(row_ptr);
				solver_row_ptr_b.write(row_ptr);

				int end_row = (pe_i+round*NUM_CH) * WINDOW_SIZE + N;

				int csc_col_val = 0, csc_row_val = 0;
				int prev = 0;
				bool csc_col_ptr_succ = false, csc_row_ind_succ = false;
				int solver_col_ptr_tmp = 0;
				for(int i = 0; i < N;){
					csc_col_val = csc_col_ptr.read(csc_col_ptr_succ);
					if(csc_col_ptr_succ) {
						for(int j = 0; j < csc_col_val-prev;){
							csc_row_val = csc_row_ind.read(csc_row_ind_succ);
							if(csc_row_ind_succ){
								if(csc_row_val < end_row) {
									solver_row_ind.write(csc_row_val-(pe_i+round*NUM_CH)*WINDOW_SIZE);
									solver_col_ptr_tmp ++;
								}
								csc_row_ind_succ = false;
								j++;
							}
						}
						solver_col_ptr.write(solver_col_ptr_tmp);
						prev = csc_col_val;
						csc_col_ptr_succ = false;
						i++;
					}
				}
		}
}

void read_A_and_split(int pe_i, int total_N,
	tapa::async_mmap<ap_uint<96>>& csr_edge_list_ch,
	tapa::mmap<int> csr_edge_list_ptr,
	tapa::ostream<ap_uint<96>>& fifo_A_ch,
	tapa::ostream<int>& fifo_A_ptr,
	tapa::ostream<ap_uint<96>>& spmv_val,
	tapa::ostream<int>& spmv_inst){
		int offset = 0;
		int offset_i = 0;
		int level = (total_N%WINDOW_SIZE == 0)?total_N/WINDOW_SIZE:total_N/WINDOW_SIZE+1;
		int bound = (level%NUM_CH>pe_i)?level/NUM_CH+1:level/NUM_CH;
		for(int round = 0;round < bound;round++){
			for(int i = 0; i < pe_i+NUM_CH*round+1; i++){
				const int N = csr_edge_list_ptr[i+offset_i];
				if(i == pe_i+NUM_CH*round) {
					fifo_A_ptr.write(N);
				}
				else {
					spmv_inst.write(N);
				}
				for (int i_req = 0, i_resp = 0; i_resp < N;) {
					if(i_req < N && !csr_edge_list_ch.read_addr.full()){
							csr_edge_list_ch.read_addr.try_write(i_req+offset);
							++i_req;
					}
					if(!csr_edge_list_ch.read_data.empty()){
						if(i == pe_i+NUM_CH*round && !fifo_A_ch.full()){
							ap_uint<96> tmp;
							csr_edge_list_ch.read_data.try_read(tmp);
							fifo_A_ch.try_write(tmp);
							++i_resp;
						}else if(i != pe_i+NUM_CH*round && !spmv_val.full()){
							ap_uint<96> tmp;
							csr_edge_list_ch.read_data.try_read(tmp);
							spmv_val.try_write(tmp);
							++i_resp;
						}
					}
				}
				offset+=N;
			}
			offset_i += (pe_i+NUM_CH*round+1);
		}
	}

void X_Merger(int pe_i, int N, tapa::istream<float>& x_first, tapa::istream<float>& x_second, tapa::ostream<float>& x_out, tapa::ostream<float>& x_res, tapa::istream<int>& N_in){
	int level = (N%WINDOW_SIZE == 0)?N/WINDOW_SIZE:N/WINDOW_SIZE+1;
	bool is_last = ((level % NUM_CH == (pe_i+1)%NUM_CH) && (pe_i != NUM_CH-1));
	int bound = (level % NUM_CH == 0) ? (level / NUM_CH) - 1 : level/NUM_CH; 
	for(int round = 0;;round++){
		const int num_x = N_in.read();
		for(int i = 0; i < (pe_i+round*NUM_CH) * WINDOW_SIZE; i++){
			if(is_last && round == bound){
				x_res.write(x_first.read());
			}else{
				x_out.write(x_first.read());
			}
		}
		for(int i = 0; i < num_x; i++){
			if(is_last && round == bound){
				x_res.write(x_second.read());
			}else{
				x_out.write(x_second.read());
			}
		}
	}
}

void read_len( tapa::mmap<int> K_csc, 
	int N, 
	tapa::ostreams<int, NUM_CH>& K_csc_val, 
	tapa::ostreams<int, NUM_CH>& N_val){
	int bound = (N%WINDOW_SIZE == 0) ? N/WINDOW_SIZE : N/WINDOW_SIZE+1;
	for(int i = 0; i < bound; i++){
		int len = ((N-i*WINDOW_SIZE) < WINDOW_SIZE) ? N%WINDOW_SIZE : WINDOW_SIZE;
		K_csc_val[i%NUM_CH].write(K_csc[i]);
		N_val[i%NUM_CH].write(len);
	}
}

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

void broadcast(tapa::istream<int>& N, tapa::ostream<int>& N_sub){
	for(;;){
		N_sub.write(N.read());
	}
}

void read_X(tapa::async_mmap<float>& mmap, tapa::istream<bool>& fin_write, tapa::ostream<float>& fifo_x, int N){
	for(int round = 0; round < N/WINDOW_SIZE; round++){
		const int N = (round+1)*WINDOW_SIZE*NUM_CH;
		fin_write.read();
		for (int i_req = 0, i_resp = 0; i_resp < N;) {
			if(i_req < N && !mmap.read_addr.full()){
					mmap.read_addr.try_write(i_req);
					++i_req;
			}
			if(!mmap.read_data.empty() && !fifo_x.full()){
					float tmp;
					mmap.read_data.try_read(tmp);
					fifo_x.try_write(tmp);
					++i_resp;
			}
		}
	}
}

void write_X_residue(tapa::istreams<float, NUM_CH>& x_res, tapa::ostream<float>& dump, tapa::async_mmap<float>& x, int N){
	int level = (N%WINDOW_SIZE == 0)?N/WINDOW_SIZE:N/WINDOW_SIZE+1;
	int last = ((level%NUM_CH)+7)%NUM_CH;
	int dump_size = N - (N %(WINDOW_SIZE*NUM_CH));
	if(dump_size != N){
		for(int i = 0; i < dump_size; i++){
			dump.write(x_res[last].read());
		}
		write_float_vec(x_res[last], x, N-dump_size, dump_size);
	}
}


void Timer(tapa::istream<bool>& q_done, tapa::mmap<int> cycle_count){
        int i = 0;
        while(true){
                ++i;
                if(!q_done.empty()){
                        q_done.read();
                        break;
                }
        }
        cycle_count[0] = i;
}

void SolverMiddleware( int pe_i, int N,
	tapa::mmap<ap_uint<96>> csr_edge_list_ch,
	tapa::mmap<int> csr_edge_list_ptr,
	tapa::mmap<float> f,
	tapa::mmap<int> csc_col_ptr,
	tapa::mmap<int> csc_row_ind,
	tapa::istream<float>& x_q_in,
	tapa::ostream<float>& x_q_out,
	tapa::ostream<float>& x_res,
	tapa::istream<int>& N_val,
	tapa::istream<int>& K_csc_val){
		
		tapa::stream<ap_uint<96>> fifo_A_ch("fifo_A_ch");
		tapa::stream<int> fifo_A_ptr("fifo_A_ptr");
		tapa::stream<int> csc_col_ptr_q("csc_col_ptr");
		tapa::stream<int> csc_row_ind_q("csc_row_ind");
		tapa::stream<float> f_q("f");
		tapa::stream<ap_uint<96>> spmv_val("spmv_val");
		tapa::stream<int> spmv_inst("spmv_inst");
		tapa::stream<int, WINDOW_SIZE> solver_row_ptr_a("solver_row_ptr_a");
		tapa::stream<int, WINDOW_SIZE> solver_row_ptr_b("solver_row_ptr_b");
		tapa::stream<int, WINDOW_SIZE> solver_col_ind("solver_col_ind");
		tapa::stream<float, WINDOW_SIZE> solver_val("solver_val");
		/* for csc mapping search*/
		tapa::stream<int, WINDOW_SIZE> solver_col_ptr("solver_col_ptr");
		tapa::stream<int, WINDOW_SIZE> solver_row_ind("solver_row_ind");
		tapa::stream<int, WINDOW_SIZE> next("next");
    	tapa::stream<int, WINDOW_SIZE> ack("ack");

		tapa::streams<int, 2> K_solver("k_solver");
		tapa::streams<int, 3> K_sub_csc("k_sub_csc");
		tapa::streams<int, 7> N_sub("n_sub");
		tapa::stream<float> y("y");
		tapa::stream<float> x_prev("x_prev");
		tapa::stream<float> x_next("x_next");

		tapa::task()
			.invoke<tapa::detach>(broadcast, N_val, N_sub)
			.invoke<tapa::detach>(broadcast, K_csc_val, K_sub_csc)
			.invoke<tapa::detach>(read_float_vec, f, N_sub, N_sub, f_q)
			.invoke<tapa::detach>(read_int_vec, csc_col_ptr, N_sub, N_sub, csc_col_ptr_q)
		    .invoke<tapa::detach>(read_int_vec, csc_row_ind, K_sub_csc, K_sub_csc, csc_row_ind_q)
			.invoke<tapa::detach>(black_hole_int, K_sub_csc)
			.invoke<tapa::detach>(read_A_and_split, pe_i, N, csr_edge_list_ch, csr_edge_list_ptr, fifo_A_ch, fifo_A_ptr, spmv_val, spmv_inst)
			.invoke<tapa::detach>(PEG_Xvec, pe_i, spmv_inst, spmv_val, x_q_in, x_prev, y, N_sub, N_sub)
			.invoke<tapa::detach>(PEG_split, pe_i,
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
				K_solver,
				N_sub, N_sub)
			// delete after implementation
			.invoke<tapa::detach>(analyze, solver_row_ptr_a, solver_col_ptr, solver_row_ind, ack, next, N_sub, N_sub, K_solver, K_solver) // ?
			.invoke<tapa::detach>(solve_v2, next, ack, solver_row_ptr_b, solver_col_ind, solver_val, f_q, y, x_next, N_sub, N_sub, K_solver)
			.invoke<tapa::detach>(X_Merger, pe_i, N, x_prev, x_next, x_q_out, x_res, N_sub);
			// .invoke<tapa::detach>(black_hole_int, solver_row_ptr_a)
			// .invoke<tapa::detach>(black_hole_int, solver_row_ptr_b)
			// .invoke<tapa::detach>(black_hole_int, solver_col_ind)
			// .invoke<tapa::detach>(black_hole_float, solver_val)
			// .invoke<tapa::detach>(black_hole_int, solver_col_ptr)
			// .invoke<tapa::detach>(black_hole_int, solver_row_ind)
			// .invoke<tapa::detach>(black_hole_float, f_q);
			
			// .invoke(analyze, solver_row_ptr_a, solver_col_ptr, solver_row_ind, ack, next, N, K_solver_val, K_solver_val)
			// .invoke(solve_v2, next, ack, solver_row_ptr_b, solver_col_ind, solver_val, f_q, x_q, N, K_solver_val);
	}

void TrigSolver(tapa::mmaps<ap_uint<96>, NUM_CH> csr_edge_list_ch,
			tapa::mmaps<int, NUM_CH> csr_edge_list_ptr,
			tapa::mmaps<int, NUM_CH> csc_col_ptr, 
			tapa::mmaps<int, NUM_CH> csc_row_ind, 
			tapa::mmaps<float, NUM_CH> f, 
			tapa::mmap<float> x, 
			int N, // # of dimension
			tapa::mmap<int> K_csc,
			tapa::mmap<int> cycle_count){

	tapa::streams<int, NUM_CH> K_csc_val("k_csc_val");
	tapa::streams<int, NUM_CH> N_val("n_val");
	tapa::streams<float, NUM_CH+1> x_q("x_pipe");
	tapa::streams<float, NUM_CH> x_res("x_res");
	tapa::stream<bool> fin_write("fin_write");
	tapa::stream<float> dump("dump");
	tapa::stream<float> dump_res("dump_res");

	tapa::task()
		.invoke(read_len, K_csc, N, K_csc_val, N_val)
		.invoke<tapa::detach>(read_X, x, fin_write, x_q, N)
		.invoke<tapa::detach, NUM_CH>(SolverMiddleware, tapa::seq(), N, csr_edge_list_ch, csr_edge_list_ptr, f, csc_col_ptr, csc_row_ind, x_q, x_q, x_res, N_val, K_csc_val)
		.invoke<tapa::join>(write_x, x_q, dump, fin_write, x, N)
		.invoke<tapa::join>(write_X_residue, x_res, dump_res, x, N)
		.invoke<tapa::detach>(black_hole_float, dump)
		.invoke<tapa::detach>(black_hole_float, dump_res);
		//.invoke(Timer, q_done, cycle_count);	
}
