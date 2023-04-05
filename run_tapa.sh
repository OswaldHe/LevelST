tapac \
    -o solver.xilinx_u280_xdma_201920_3.hw.xo \
    --platform xilinx_u280_xdma_201920_3 \
    --top TrigSolver \
    --work-dir solver.xilinx_u280_xdma_201920_3.hw.xo.tapa \
    --connectivity hbm_config.ini \
    --enable-hbm-binding-adjustment \
    --enable-synth-util \
    --run-floorplan-dse \
    --min-area-limit 0.58 \
    --min-slr-width-limit 5000 \
    --max-parallel-synth-jobs 16 \
    --floorplan-output solver.tcl \
    solver-general.cpp 