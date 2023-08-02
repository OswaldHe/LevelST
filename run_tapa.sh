tapac \
    -o solver.xilinx_u280_gen3x16_xdma_1_202211_1.hw.xo \
    --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
     --top TrigSolver \
    --work-dir solver.xilinx_u280_gen3x16_xdma_1_202211_1.hw.xo.tapa \
    --connectivity hbm_config.ini \
    --read-only-args csr_edge_list_ch* \
    --read-only-args dep_graph_ch* \
    --enable-hbm-binding-adjustment \
    --enable-synth-util \
    --run-floorplan-dse \
    --min-area-limit 0.58 \
    --min-slr-width-limit 5000 \
    --max-slr-width-limit 19000 \
    --max-parallel-synth-jobs 16 \
    --floorplan-output solver.tcl \
    solver-general.cpp 
