# AIR kernel argument spec, consumed by air-rewrite.
# One "kernel" block per exported kernel; arguments in IR parameter order.
#   buffer  <index> <read|write|read_write> <air.arg_type_name> <size> <align> <arg_name>
#   builtin <index> <air builtin name>      <air.arg_type_name>            <arg_name>
kernel vadd
buffer  0 read       float 4 4 a
buffer  1 read       float 4 4 b
buffer  2 read_write float 4 4 c
builtin 3 air.thread_position_in_grid uint gid
kernel reduce
buffer  0 read       float 4 4 inbuf
buffer  1 read_write float 4 4 out
buffer  2 read_write uint  4 4 counter
builtin 3 air.threadgroup_position_in_grid uint tgid
builtin 4 air.thread_position_in_threadgroup uint tid
builtin 5 air.thread_index_in_simdgroup uint lane
builtin 6 air.simdgroup_index_in_threadgroup uint sgid
kernel parsef
buffer  0 read       char 1 1 text
buffer  1 read       uint 4 4 lengths
buffer  2 read_write float 4 4 out
builtin 3 air.thread_position_in_grid uint gid
