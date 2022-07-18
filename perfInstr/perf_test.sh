#!/bin/bash

               ctl_dir=/tmp/

               ctl_fifo=${ctl_dir}perf_ctl.fifo
               test -p ${ctl_fifo} && unlink ${ctl_fifo}
               mkfifo ${ctl_fifo}
               exec {ctl_fd}<>${ctl_fifo}

               ctl_ack_fifo=${ctl_dir}perf_ctl_ack.fifo
               test -p ${ctl_ack_fifo} && unlink ${ctl_ack_fifo}
               mkfifo ${ctl_ack_fifo}
               exec {ctl_fd_ack}<>${ctl_ack_fifo}

               perf stat -D -1    \
                         --control fd:${ctl_fd},${ctl_fd_ack} \
                         ${1}

               exec {ctl_fd_ack}>&-
               unlink ${ctl_ack_fifo}

               exec {ctl_fd}>&-
               unlink ${ctl_fifo}

               wait -n ${perf_pid}
               exit $?

