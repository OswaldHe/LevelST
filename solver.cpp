#include <cstdint>
#include <tapa.h>
#include <ap_int.h>
#include <glog/logging.h>

constexpr int WINDOW_SIZE = 256;

using float_v16 = tapa::vec_t<float, 16>;
using int_v16 = tapa::vec_t<int, 16>;

void read_int_vec(tapa::async_mmap<int_v16>& mmap, int N,
                 tapa::ostream<int_v16>& stream) {
	int cap = (N%16 == 0) ? N/16 : (N/16+1);
  for (int i_req = 0, i_resp = 0; i_resp < cap;) {
        if(i_req < cap && !mmap.read_addr.full()){
                mmap.read_addr.try_write(i_req);
                ++i_req;
        }
        if(!mmap.read_data.empty() && !stream.full()){
                int_v16 tmp;
                mmap.read_data.try_read(tmp);
				stream.try_write(tmp);
				++i_resp;	
        }
  }
}

void duplicate_int_vec(tapa::async_mmap<int_v16>& mmap, int N,
                 tapa::ostream<int_v16>& stream_a, tapa::ostream<int_v16>& stream_b){
	int cap = (N%16 == 0) ? N/16 : (N/16+1);
  for (int i_req = 0, i_resp = 0; i_resp < cap;) {
        if(i_req < cap && !mmap.read_addr.full()){
                mmap.read_addr.try_write(i_req);
                ++i_req;
        }
        if(!mmap.read_data.empty() && !stream_a.full() && !stream_b.full()){
                int_v16 tmp;
                mmap.read_data.try_read(tmp);
				stream_a.try_write(tmp);
				stream_b.try_write(tmp);
				++i_resp;
                
        }
  }

}

void write_float_vec(tapa::istream<float>& stream, tapa::async_mmap<float>& mmap,
                 int N, tapa::ostream<bool>& q_done) {
  for(int i_req = 0, i_resp = 0; i_resp < N;){
  	if(i_req < N && !stream.empty() && !mmap.write_addr.full() && !mmap.write_data.full()){
		mmap.write_addr.try_write(i_req);
		mmap.write_data.try_write(stream.read(nullptr));
		++i_req;
	}
	if(!mmap.write_resp.empty()){
		i_resp += unsigned(mmap.write_resp.read(nullptr))+1;
	}
  }
  q_done.write(true);
}

void read_float_vec(tapa::async_mmap<float_v16>& mmap, int N,
                 tapa::ostream<float_v16>& stream) {
   int cap = (N%16 == 0) ? N/16 : (N/16+1);
  for (int i_req = 0, i_resp = 0; i_resp < cap;) {
        if(i_req < cap && !mmap.read_addr.full()){
                mmap.read_addr.try_write(i_req);
                ++i_req;
        }
        if(!mmap.read_data.empty() && !stream.full()){
                float_v16 tmp;
                mmap.read_data.try_read(tmp);
				stream.try_write(tmp);
                ++i_resp;
        }
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

void solve_v2(tapa::istream<int>& next,
		tapa::ostream<int>& ack,
		tapa::istream<int_v16>& ia,
		tapa::istream<int_v16>& ja,
		tapa::istream<float_v16>& a,
		tapa::istream<float_v16>& f,
		tapa::ostream<float>& x, int N, int K){
	
	float local_x[WINDOW_SIZE];
	float local_a[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
	int local_ia[WINDOW_SIZE];
	int local_ja[(WINDOW_SIZE+1)*WINDOW_SIZE/2];
	float cyclic_aggregate[16];

#pragma HLS bind_storage type=ram_2p impl=bram latency=1
#pragma HLS array_partition cyclic variable=local_x factor=16
#pragma HLS array_partition cyclic variable=local_a factor=16
#pragma HLS array_partition cyclic variable=local_ia factor=4
#pragma HLS array_partition cyclic variable=local_ja factor=16
#pragma HLS array_partition complete variable=cyclic_aggregate

	int_v16 ia_val;
	float_v16 f_val;
	float_v16 a_val;
	int_v16 ja_val;

	bool ia_succ = false, ja_succ = false, a_succ = false, f_succ = false;

read_ia_and_f:
	for(int i = 0; i < N;) {
#pragma HLS loop_tripcount min=1 max=16
#pragma HLS pipeline II=1
		if(!ia_succ) ia_val = ia.read(ia_succ);
		if(!f_succ) f_val = f.read(f_succ);
		if(ia_succ && f_succ){
			for(int j = 0; j < 16 && i < N; j++){
				local_ia[i] = ia_val[j];	
				local_x[i] = f_val[j];
				i++;
			}
			ia_succ = f_succ = false;
		}
	}

read_ja_and_a:
	for(int i = 0; i < K;){
#pragma HLS loop_tripcount min=1 max=2000
#pragma HLS pipeline II=1
		if(!ja_succ) ja_val = ja.read(ja_succ);
		if(!a_succ) a_val = a.read(a_succ);
		if(ja_succ && a_succ){
			for(int j = 0; j < 16 && i < K; j++){
				local_a[i] = a_val[j];
				local_ja[i] = ja_val[j];
				i++;
			}
			a_succ = ja_succ = false;
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
		x.write(local_x[i]);
	}

}

void analyze(tapa::istream<int_v16>& ia,
		tapa::istream<int_v16>& csc_col_ptr,
		tapa::istream<int_v16>& csc_row_ind,
		tapa::istream<int>& ack,
		tapa::ostream<int>& next,
		int N, int K){

        int num_nn = 0;
        int diff = 0;
        int_v16 ia_val;
	int_v16 csc_col_val;
	int_v16 csc_row_val;
	int parents[WINDOW_SIZE];
	int local_csc_col_ptr[WINDOW_SIZE];
	int local_csc_row_ind[(WINDOW_SIZE+1)*WINDOW_SIZE/2];

#pragma HLS array_partition variable=parents cyclic factor=4
#pragma HLS array_partition variable=local_csc_col_ptr cyclic factor=4
#pragma HLS array_partition variable=local_csc_row_ind cyclic factor=4

        bool ia_succ = false, csc_col_succ = false, csc_row_succ = false;

compute_parents:
        for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=16
                if(!ia_succ) ia_val = ia.read(ia_succ);
                if(ia_succ){
					for(int j = 0; j < 16 && i < N; j++){
                		diff = ia_val[j] - num_nn;
                		num_nn = ia_val[j];
						parents[i] = diff-1;
                		if(parents[i] == 0) next.write(i);
                		i++;
					}
					ia_succ = false;
                }
        }

read_csc_col:
	for(int i = 0; i < N;){
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=16
		if(!csc_col_succ) csc_col_val = csc_col_ptr.read(csc_col_succ);
		if(csc_col_succ){
			for(int j = 0; j < 16 && i < N; j++){
				local_csc_col_ptr[i] = csc_col_val[j];
				i++;
			}
			csc_col_succ = false;
		}
	}

read_csc_row:
	for(int i = 0; i < K;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1 max=2000
		if(!csc_row_succ) csc_row_val = csc_row_ind.read(csc_row_succ);
        if(csc_row_succ){
			for(int j = 0; j < 16 && i < K; j++){
                local_csc_row_ind[i] = csc_row_val[j];
                i++;
			}
			csc_row_succ = false;
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

void TrigSolver(tapa::mmap<float_v16> A, tapa::mmap<int_v16> IA, tapa::mmap<int_v16> JA, tapa::mmap<int_v16> csc_col_ptr, tapa::mmap<int_v16> csc_row_ind, tapa::mmap<float_v16> f, tapa::mmap<float> x, int N, int K, tapa::mmap<int> cycle_count){
	// sparse matrix input in CSR format
	tapa::stream<float_v16, 16> a_q("A");
	tapa::stream<int_v16, 16> ia_qa("IA_A");
	tapa::stream<int_v16, 16> ia_qb("IA_B");
	tapa::stream<int_v16, 16> ja_q("JA");
	tapa::stream<int_v16, 16> csc_col_ptr_q("csc_col_ptr");
	tapa::stream<int_v16, 16> csc_row_ind_q("csc_row_ind");
	tapa::stream<float_v16, 16> f_q("f");
	tapa::stream<float> x_q("x");
	tapa::stream<bool> q_done("q_d");
	tapa::stream<int, WINDOW_SIZE> next("next");
    tapa::stream<int, WINDOW_SIZE> ack("ack");

	tapa::task()
		.invoke(read_float_vec, A, K, a_q)
		.invoke(duplicate_int_vec, IA, N, ia_qa, ia_qb)
		.invoke(read_int_vec, JA, K, ja_q)
	    .invoke(read_float_vec, f, N, f_q)
		.invoke(read_int_vec, csc_col_ptr, N, csc_col_ptr_q)
		.invoke(read_int_vec, csc_row_ind, K, csc_row_ind_q)
		.invoke(analyze, ia_qa, csc_col_ptr_q, csc_row_ind_q, ack, next, N, K)
		.invoke(solve_v2, next, ack, ia_qb, ja_q, a_q, f_q, x_q, N, K)
		//.invoke(solve, a_q, ia_q, ja_q, f_q, x_q, N)
		.invoke(write_float_vec, x_q, x, N, q_done)
		.invoke(Timer, q_done, cycle_count);	
}
