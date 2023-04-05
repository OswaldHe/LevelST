#include <vector>
#include <iostream>
#include <tapa.h>
#include "mmio.h"

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

#ifndef SPARSE_HELPER
#define SPARSE_HELPER

template <typename data_t>
struct rcv{
    int r;
    int c;
    data_t v;
};

enum MATRIX_FORMAT {CSR, CSC};

template <typename data_t>
struct edge{
    int col;
    int row;
    data_t attr;
    
    edge(int d = -1, int s = -1, data_t v = 0): col(d), row(s), attr(v) {}
    
    edge& operator=(const edge& rhs) {
        col = rhs.col;
        row = rhs.row;
        attr = rhs.attr;
        return *this;
    }
};

template <typename data_t>
int cmp_by_row_column(const void *aa,
                      const void *bb) {
    rcv<data_t> * a = (rcv<data_t> *) aa;
    rcv<data_t> * b = (rcv<data_t> *) bb;
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    return 0;
}

template <typename data_t>
int cmp_by_column_row(const void *aa,
                      const void *bb) {
    rcv<data_t> * a = (rcv<data_t> *) aa;
    rcv<data_t> * b = (rcv<data_t> *) bb;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    return 0;
}

template <typename data_t>
void sort_by_fn(int nnz_s,
                vector<int> & cooRowIndex,
                vector<int> & cooColIndex,
                vector<data_t> & cooVal,
                int (* cmp_func)(const void *, const void *)) {
    auto rcv_arr = new rcv<data_t>[nnz_s];
    
    for(int i = 0; i < nnz_s; ++i) {
        rcv_arr[i].r = cooRowIndex[i];
        rcv_arr[i].c = cooColIndex[i];
        rcv_arr[i].v = cooVal[i];
    }
    
    qsort(rcv_arr, nnz_s, sizeof(rcv<data_t>), cmp_func);
    
    for(int i = 0; i < nnz_s; ++i) {
        cooRowIndex[i] = rcv_arr[i].r;
        cooColIndex[i] = rcv_arr[i].c;
        cooVal[i] = rcv_arr[i].v;
    }
    
    delete [] rcv_arr;
}

void mm_init_read(FILE * f,
                  char * filename,
                  MM_typecode & matcode,
                  int & m,
                  int & n,
                  int & nnz) {
    //if ((f = fopen(filename, "r")) == NULL) {
    //        cout << "Could not open " << filename << endl;
    //        return 1;
    //}
    
    if (mm_read_banner(f, &matcode) != 0) {
        cout << "Could not process Matrix Market banner for " << filename << endl;
        exit(1);
    }
    
    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) != 0) {
        cout << "Could not read Matrix Market format for " << filename << endl;
        exit(1);
    }
}

void load_S_matrix(FILE * f_A,
                   int nnz_mmio,
                   int & nnz,
                   vector<int> & cooRowIndex,
                   vector<int> & cooColIndex,
                   vector<float> & cooVal,
                   MM_typecode & matcode) {
    
    if (mm_is_complex(matcode)) {
        cout << "Redaing in a complex matrix, not supported yet!" << endl;
        exit(1);
    }
    
    if (!mm_is_symmetric(matcode)) {
        cout << "It's an NS matrix.\n";
    } else {
        cout << "It's an S matrix.\n";
    }
    
    int r_idx, c_idx;
    float value;
    int idx = 0;
    
    for (int i = 0; i < nnz_mmio; ++i) {
        if (mm_is_pattern(matcode)) {
            fscanf(f_A, "%d %d\n", &r_idx, &c_idx);
            value = 1.0;
        }else {
            fscanf(f_A, "%d %d %lf\n", &r_idx, &c_idx, &value);
        }
        
        //unsigned int * tmpPointer_v = reinterpret_cast<unsigned int*>(&value);
        //unsigned int uint_v = *tmpPointer_v;
        
        uint64_t * tmpPointer_v = reinterpret_cast<uint64_t*>(&value);
        uint64_t uint_v = *tmpPointer_v;
        
        if (uint_v != 0) {
            if (r_idx < 1 || c_idx < 1) { // report error
                cout << "idx = " << idx << " [" << r_idx - 1 << ", " << c_idx - 1 << "] = " << value << endl;
                exit(1);
            }
            
            cooRowIndex[idx] = r_idx - 1;
            cooColIndex[idx] = c_idx - 1;
            cooVal[idx] = value;
            idx++;
            
            if (mm_is_symmetric(matcode)) {
                if (r_idx != c_idx) {
                    cooRowIndex[idx] = c_idx - 1;
                    cooColIndex[idx] = r_idx - 1;
                    cooVal[idx] = value;
                    idx++;
                }
            }
        }
    }
    nnz = idx;
}

void read_suitsparse_matrix_FP64(char * filename_A,
                                 vector<int> & elePtr,
                                 vector<int> & eleIndex,
                                 vector<float> & eleVal,
                                 int & M,
                                 int & K,
                                 int & nnz,
                                 const MATRIX_FORMAT mf=CSR) {
    int nnz_mmio;
    MM_typecode matcode;
    FILE * f_A;
    
    if ((f_A = fopen(filename_A, "r")) == NULL) {
        cout << "Could not open " << filename_A << endl;
        exit(1);
    }
    
    mm_init_read(f_A, filename_A, matcode, M, K, nnz_mmio);
    
    if (!mm_is_coordinate(matcode)) {
        cout << "The input matrix file " << filename_A << "is not a coordinate file!" << endl;
        exit(1);
    }
    
    int nnz_alloc = (mm_is_symmetric(matcode))? (nnz_mmio * 2): nnz_mmio;
    cout << "Matrix A -- #row: " << M << " #col: " << K << endl;
    
    vector<int> cooRowIndex(nnz_alloc);
    vector<int> cooColIndex(nnz_alloc);
    //eleIndex.resize(nnz_alloc);
    eleVal.resize(nnz_alloc);
    
    cout << "Loading input matrix A from " << filename_A << "\n";
    
    load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, eleVal, matcode);
    
    fclose(f_A);
    
    if (mf == CSR) {
        sort_by_fn(nnz, cooRowIndex, cooColIndex, eleVal, cmp_by_row_column<float>);
    }else if (mf == CSC) {
        sort_by_fn(nnz, cooRowIndex, cooColIndex, eleVal, cmp_by_column_row<float>);
    }else {
        cout << "Unknow format!\n";
        exit(1);
    }
    
    // convert to CSR/CSC format
    int M_K = (mf == CSR)? M : K;
    elePtr.resize(M_K+1);
    vector<int> counter(M_K, 0);
    
    if (mf == CSR) {
        for (int i = 0; i < nnz; i++) {
            counter[cooRowIndex[i]]++;
        }
    }else if (mf == CSC) {
        for (int i = 0; i < nnz; i++) {
            counter[cooColIndex[i]]++;
        }
    }else {
        cout << "Unknow format!\n";
        exit(1);
    }
    
    int t = 0;
    for (int i = 0; i < M_K; i++) {
        t += counter[i];
    }
    
    elePtr[0] = 0;
    for (int i = 1; i <= M_K; i++) {
        elePtr[i] = elePtr[i - 1] + counter[i - 1];
    }
    
    eleIndex.resize(nnz);
    if (mf == CSR) {
        for (int i = 0; i < nnz; ++i) {
            eleIndex[i] = cooColIndex[i];
        }
    }else if (mf == CSC){
        for (int i = 0; i < nnz; ++i) {
            eleIndex[i] = cooRowIndex[i];
        }
    }
    
    if (mm_is_symmetric(matcode)) {
        //eleIndex.resize(nnz);
        eleVal.resize(nnz);
    }
}

template <typename data_t>
void cpu_spmv_CSR(const int M,
                  const int K,
                  const int NNZ,
                  const data_t ALPHA,
                  const vector<int> & CSRRowPtr,
                  const vector<int> & CSRColIndex,
                  const vector<data_t> & CSRVal,
                  const vector<data_t> & vec_X,
                  const data_t BETA,
                  vector<data_t> & vec_Y) {
    // A: sparse matrix, M x K
    // X: dense vector, K x 1
    // Y: dense vecyor, M x 1
    // output vec_Y = ALPHA * mat_A * vec_X + BETA * vec_Y
    // dense matrices: column major
    
    for (int i = 0; i < M; ++i) {
        data_t psum = 0;
        for (int j = CSRRowPtr[i]; j < CSRRowPtr[i+1]; ++j) {
            psum += CSRVal[j] * vec_X[CSRColIndex[j]];
        }
        vec_Y[i] = ALPHA * psum + BETA * vec_Y[i];
    }
}

template <typename data_t>
void generate_edge_list_for_one_PE(const vector<edge<data_t>> & tmp_edge_list,
                                   vector<edge<data_t>> & edge_list,
                                   const int base_col_index,
                                   const int i_start,
                                   const int NUM_Row,
                                   const int NUM_PE,
                                   const int DEP_DIST_LOAD_STORE = 10){
    
    edge<data_t> e_empty = {-1, -1, 0.0};
    //vector<edge> scheduled_edges(NUM_Row);
    //std::fill(scheduled_edges.begin(), scheduled_edges.end(), e_empty);
    vector<edge<data_t>> scheduled_edges;
    
    //const int DEP_DIST_LOAD_STORE = 7;
    
    vector<int> cycles_rows(NUM_Row, -DEP_DIST_LOAD_STORE);
    int e_dst, e_src;
    float e_attr;
    for (unsigned int pp = 0; pp < tmp_edge_list.size(); ++pp) {
        e_src = tmp_edge_list[pp].col - base_col_index;
        //e_dst = tmp_edge_list[pp].row / 2 / NUM_PE;
        e_dst = tmp_edge_list[pp].row / NUM_PE;
        e_attr = tmp_edge_list[pp].attr;
        auto cycle = cycles_rows[e_dst] + DEP_DIST_LOAD_STORE;
        
        bool taken = true;
        while (taken){
            if (cycle >= ((int)scheduled_edges.size()) ) {
                scheduled_edges.resize(cycle + 1, e_empty);
            }
            auto e = scheduled_edges[cycle];
            if (e.row != -1)
                cycle++;
            else
                taken = false;
        }
        scheduled_edges[cycle].col = e_src;
        //scheduled_edges[cycle].row = e_dst * 2 + (tmp_edge_list[pp].row % 2);
        scheduled_edges[cycle].row = e_dst;
        scheduled_edges[cycle].attr = e_attr;
        cycles_rows[e_dst] = cycle;
    }
    
    int scheduled_edges_size = scheduled_edges.size();
    if (scheduled_edges_size > 0) {
        //edge_list.resize(i_start + scheduled_edges_size + DEP_DIST_LOAD_STORE - 1, e_empty);
        edge_list.resize(i_start + scheduled_edges_size, e_empty);
        for (int i = 0; i < scheduled_edges_size; ++i) {
            edge_list[i + i_start] = scheduled_edges[i];
        }
    }
}

template <typename data_t>
void generate_edge_list_for_all_PEs(const vector<int> & CSCColPtr,
                                    const vector<int> & CSCRowIndex,
                                    const vector<data_t> & CSCVal,
                                    const int NUM_PE,
                                    const int NUM_ROW,
                                    const int NUM_COLUMN,
                                    const int WINDOE_SIZE,
                                    vector<vector<edge<data_t>> > & edge_list_pes,
                                    vector<int> & edge_list_ptr,
                                    const int DEP_DIST_LOAD_STORE = 10) {
    edge_list_pes.resize(NUM_PE);
    edge_list_ptr.resize((NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE + 1, 0);
    
    vector<vector<edge<data_t>> > tmp_edge_list_pes(NUM_PE);
    for (int i = 0; i < (NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE; ++i) {
        for (int p = 0; p < NUM_PE; ++p) {
            tmp_edge_list_pes[p].resize(0);
        }
        
        //fill tmp_edge_lsit_pes
        for (int col =  WINDOE_SIZE * i; col < min(WINDOE_SIZE * (i + 1), NUM_COLUMN); ++col) {
            for (int j = CSCColPtr[col]; j < CSCColPtr[col+1]; ++j) {
                //int p = (CSCRowIndex[j] / 2) % NUM_PE;
                int p = CSCRowIndex[j] % NUM_PE;
                int pos = tmp_edge_list_pes[p].size();
                tmp_edge_list_pes[p].resize(pos + 1);
                tmp_edge_list_pes[p][pos] = edge<data_t>(col, CSCRowIndex[j], CSCVal[j]);
            }
        }
        
        //form the scheduled edge list for each PE
        for (int p = 0; p < NUM_PE; ++p) {
            int i_start = edge_list_pes[p].size();
            int base_col_index = i * WINDOE_SIZE;
            generate_edge_list_for_one_PE(tmp_edge_list_pes[p],
                                          edge_list_pes[p],
                                          base_col_index,
                                          i_start,
                                          NUM_ROW,
                                          NUM_PE,
                                          DEP_DIST_LOAD_STORE);
        }
        
        //insert bubules to align edge list
        int max_len = 0;
        for (int p = 0; p < NUM_PE; ++p) {
            max_len = max((int) edge_list_pes[p].size(), max_len);
        }
        for (int p = 0; p < NUM_PE; ++p) {
            edge_list_pes[p].resize(max_len, edge<data_t>(-1,-1,0.0));
        }
        
        //pointer
        edge_list_ptr[i+1] = max_len;
    }
    
}

void edge_list_64bit_fp64(const vector<vector<edge<double>> > & edge_list_pes,
                          const vector<int> & edge_list_ptr,
                          vector<vector<ap_uint<128>, tapa::aligned_allocator<ap_uint<128>> > > & sparse_A_fpga_vec,
                          const int NUM_CH_SPARSE = 16) {
    
    int sparse_A_fpga_column_size = 4 * edge_list_ptr[edge_list_ptr.size()-1];
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 255)/256) * 256; //4 KB
    
    for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
        sparse_A_fpga_vec[cc].resize(sparse_A_fpga_chunk_size, 0);
    }
    
    // col(12 bits) + row (20 bits) + value (32 bits)
    // ->
    // col(14 bits) + row (18 bits) + value (32 bits)
    for (int i = 0; i < edge_list_ptr[edge_list_ptr.size()-1]; ++i) {
        for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
            for (int j = 0; j < 4; ++j) {
                edge<double> e = edge_list_pes[j + cc * 4][i];
                ap_uint<128> x = 0;
                if (e.row == -1) {
                    /*
                     x = (ap_uint<96>) 0x3FFFF; //0xFFFFF; //x = 0x3FFFFF;
                     x = x << 64;
                     */
                    x(81, 64) = (ap_uint<18>)0x3FFFF;
                } else {
                    /*
                     ap_uint<96> x_col = (ap_uint<96>) e.col;
                     x_col = (x_col & 0x3FFF) << (64 + 18); // x_col = (x_col & 0xFFF) << (32 + 20); //x_col = (x_col & 0x3FF) << (32 + 22);
                     ap_uint<96> x_row = (ap_uint<96>) e.row;
                     x_row = (x_row & 0x3FFFF) << 64; //x_row = (x_row & 0xFFFFF) << 32; //x_row = (x_row & 0x3FFFFF) << 32;
                     */
                    x(95, 82) = (ap_uint<14>)(e.col & 0x3FFF);
                    x(81, 64) = (ap_uint<18>)(e.row & 0x3FFFF);
                    
                    double x_double = e.attr;
                    //float x_float = 1.0;
                    /*
                     ap_uint<64> x_double_in_int = *((ap_uint<64>*)(&x_double));
                     ap_uint<96> x_double_val_96 = ((ap_uint<96>) x_double_in_int);
                     x_double_val_96 = x_double_val_96 & 0xFFFFFFFFFFFFFFFF;
                     
                     x = x_col | x_row | x_double_val_96;
                     */
                    x(63,  0) = tapa::bit_cast<ap_uint<64>>(e.attr);
                }
                if (NUM_CH_SPARSE == 16) {
                    int pe_idx = j + cc * 4;
                    // ch= 0: pe  0
                    // ch= 1: pe  8
                    // ch= 2: pe  1
                    // ch= 3: pe  9
                    // ch= 4: pe  2
                    // ch= 5: pe 10
                    // ch= 6: pe  3
                    // ch= 7: pe 11
                    // ch= 8: pe  4
                    // ch= 9: pe 12
                    // ch=10: pe  5
                    // ch=11: pe 13
                    // ch=12: pe  6
                    // ch=13: pe 14
                    // ch=14: pe  7
                    // ch=15: pe 15
                    
                    int pix_m16 = pe_idx % 16;
                    sparse_A_fpga_vec[(pix_m16 % 8) * 2 + pix_m16 / 8][(pe_idx % 64) / 16 + i * 4] = x;
                } else {
                    cout << "UPDATE me\n";
                    exit(1);
                }
            }
        }
    }
}

void edge_list_64bit_fp32(const vector<vector<edge<float>> > & edge_list_pes,
                          const vector<int> & edge_list_ptr,
                          vector<vector<unsigned long, tapa::aligned_allocator<unsigned long> > > & sparse_A_fpga_vec,
                          const int NUM_CH_SPARSE = 8) {
    
    int sparse_A_fpga_column_size = 8 * edge_list_ptr[edge_list_ptr.size()-1] * 4 / 4;
    int sparse_A_fpga_chunk_size = ((sparse_A_fpga_column_size + 511)/512) * 512;
    
    for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
        sparse_A_fpga_vec[cc].resize(sparse_A_fpga_chunk_size, 0);
    }
    
    // col(12 bits) + row (20 bits) + value (32 bits)
    // ->
    // col(14 bits) + row (18 bits) + value (32 bits)
    for (int i = 0; i < edge_list_ptr[edge_list_ptr.size()-1]; ++i) {
        for (int cc = 0; cc < NUM_CH_SPARSE; ++cc) {
            for (int j = 0; j < 8; ++j) {
                edge<float> e = edge_list_pes[j + cc * 8][i];
                unsigned long x = 0;
                if (e.row == -1) {
                    x = 0x3FFFF; //0xFFFFF; //x = 0x3FFFFF;
                    x = x << 32;
                } else {
                    unsigned long x_col = e.col;
                    x_col = (x_col & 0x3FFF) << (32 + 18); // x_col = (x_col & 0xFFF) << (32 + 20); //x_col = (x_col & 0x3FF) << (32 + 22);
                    unsigned long x_row = e.row;
                    x_row = (x_row & 0x3FFFF) << 32; //x_row = (x_row & 0xFFFFF) << 32; //x_row = (x_row & 0x3FFFFF) << 32;
                    
                    float x_float = e.attr;
                    //float x_float = 1.0;
                    unsigned int x_float_in_int = *((unsigned int*)(&x_float));
                    unsigned long x_float_val_64 = ((unsigned long) x_float_in_int);
                    x_float_val_64 = x_float_val_64 & 0xFFFFFFFF;
                    
                    x = x_col | x_row | x_float_val_64;
                }
                if (NUM_CH_SPARSE == 24) {
                    int pe_idx = j + cc * 8;
                    // ch= 0: pe  0
                    // ch= 1: pe  4
                    // ch= 2: pe  8
                    // ch= 3: pe 12
                    // ch= 4: pe 16
                    // ch= 5: pe 20
                    // ch= 6: pe  1
                    // ch= 7: pe  5
                    // ch= 8: pe  9
                    // ch= 9: pe 13
                    // ch=10: pe 17
                    // ch=11: pe 21
                    // ch=12: pe  2
                    // ch=13: pe  6
                    // ch=14: pe 10
                    // ch=15: pe 14
                    // ch=16: pe 18
                    // ch=17: pe 22
                    // ch=18: pe  3
                    // ch=19: pe  7
                    // ch=20: pe 11
                    // ch=21: pe 15
                    // ch=22: pe 19
                    // ch=23: pe 23
                    
                    int pix_m24 = pe_idx % 24;
                    sparse_A_fpga_vec[(pix_m24 % 4) * 6 + pix_m24 / 4][(pe_idx % 192) / 24 + i * 8] = x;
                } else if (NUM_CH_SPARSE == 16) {
                    int pe_idx = j + cc * 8;
                    // ch= 0: pe  0
                    // ch= 1: pe  8
                    // ch= 2: pe  1
                    // ch= 3: pe  9
                    // ch= 4: pe  2
                    // ch= 5: pe 10
                    // ch= 6: pe  3
                    // ch= 7: pe 11
                    // ch= 8: pe  4
                    // ch= 9: pe 12
                    // ch=10: pe  5
                    // ch=11: pe 13
                    // ch=12: pe  6
                    // ch=13: pe 14
                    // ch=14: pe  7
                    // ch=15: pe 15
                    
                    int pix_m16 = pe_idx % 16;
                    sparse_A_fpga_vec[(pix_m16 % 8) * 2 + pix_m16 / 8][(pe_idx % 128) / 16 + i * 8] = x;
                } else {
                    cout << "UPDATE me\n";
                    exit(1);
                }
            }
        }
    }
}

template <typename data_t>
void CSC_2_CSR(int M,
               int K,
               int NNZ,
               const vector<int> & csc_col_Ptr,
               const vector<int> & csc_row_Index,
               const vector<data_t> & cscVal,
               vector<int> & csr_row_Ptr,
               vector<int> & csr_col_Index,
               vector<data_t> & csrVal) {
    csr_row_Ptr.resize(M + 1, 0);
    csrVal.resize(NNZ, 0.0);
    csr_col_Index.resize(NNZ, 0);
    
    for (int i = 0; i < NNZ; ++i) {
        csr_row_Ptr[csc_row_Index[i] + 1]++;
    }
    
    for (int i = 0; i < M; ++i) {
        csr_row_Ptr[i + 1] += csr_row_Ptr[i];
    }
    
    vector<int> row_nz(M, 0);
    for (int i = 0; i < K; ++i) {
        for (int j = csc_col_Ptr[i]; j < csc_col_Ptr[i + 1]; ++j) {
            int r = csc_row_Index[j];
            int c = i;
            auto v = cscVal[j];
            
            int pos = csr_row_Ptr[r] + row_nz[r];
            csrVal[pos] = v;
            csr_col_Index[pos] = c;
            row_nz[r]++;
        }
    }
}

template <typename data_t>
void extract_lower_triangular_matrix(int M,
               int K,
               int NNZ,
               const vector<int> & csr_row_Ptr,
               const vector<int> & csr_col_Index,
               const vector<data_t> & csrVal,
               aligned_vector<int>& low_trig_row_ptr,
               aligned_vector<int>& low_trig_col_ind,
               aligned_vector<data_t>& low_trig_val) {
    
    low_trig_row_ptr.push_back(0);
    int col_ptr = 0;
    for(int i = 1; i <= M; i++){
        for(int j = csr_row_Ptr[i-1]; j < csr_row_Ptr[i]; j++){
            if(csr_col_Index[j] < i){
                low_trig_col_ind.push_back(csr_col_Index[j]);
                low_trig_val.push_back(csrVal[j]);
                col_ptr++;
            } else {
                break;
            }
        }
        low_trig_row_ptr.push_back(col_ptr);
    }
    low_trig_row_ptr.push_back(col_ptr);
}

#endif