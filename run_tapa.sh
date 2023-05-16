tapac \
    -o solver.xilinx_u280_xdma_201920_3.hw.xo \
    --platform xilinx_u280_xdma_201920_3 \
    --top TrigSolver \
    --work-dir solver.xilinx_u280_xdma_201920_3.hw.xo.tapa \
    --connectivity hbm_config.ini \
    --read-only-args csr_edge_list_ch* \
    --read-only-args dep_graph_ch* \
    --enable-hbm-binding-adjustment \
    --enable-synth-util \
    --run-floorplan-dse \
    --min-area-limit 0.58 \
    --min-slr-width-limit 5000 \
    --max-slr-width-limit 18000 \
    --max-parallel-synth-jobs 12 \
    --floorplan-output solver.tcl \
    solver-general.cpp 