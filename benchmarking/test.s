# GNU C++14 (GCC) version 7.0.0 20160506 (experimental) [trunk revision 235962] (x86_64-apple-darwin15.4.0)
#	compiled by GNU C version 7.0.0 20160506 (experimental) [trunk revision 235962], GMP version 5.0.5, MPFR version 3.1.1, MPC version 0.8.1, isl version none
# warning: MPC header version 0.8.1 differs from library version 0.9.
# GGC heuristics: --param ggc-min-expand=30 --param ggc-min-heapsize=4096
# options passed:  -D__DYNAMIC__ test.cc -fPIC -mmacosx-version-min=10.11.6
# -mtune=core2 -g0 -O3 -Wall -Wextra -Wpedantic -std=c++14 -fverbose-asm
# options enabled:  -Wnonportable-cfstrings -fPIC
# -faggressive-loop-optimizations -falign-labels
# -fasynchronous-unwind-tables -fauto-inc-dec -fbranch-count-reg
# -fcaller-saves -fchkp-check-incomplete-type -fchkp-check-read
# -fchkp-check-write -fchkp-instrument-calls -fchkp-narrow-bounds
# -fchkp-optimize -fchkp-store-bounds -fchkp-use-static-bounds
# -fchkp-use-static-const-bounds -fchkp-use-wrappers
# -fcombine-stack-adjustments -fcommon -fcompare-elim -fcprop-registers
# -fcrossjumping -fcse-follow-jumps -fdefer-pop
# -fdelete-null-pointer-checks -fdevirtualize -fdevirtualize-speculatively
# -fearly-inlining -feliminate-unused-debug-types -fexceptions
# -fexpensive-optimizations -fforward-propagate -ffunction-cse -fgcse
# -fgcse-after-reload -fgcse-lm -fgnu-unique -fguess-branch-probability
# -fhoist-adjacent-loads -fident -fif-conversion -fif-conversion2
# -findirect-inlining -finline -finline-atomics -finline-functions
# -finline-functions-called-once -finline-small-functions -fipa-cp
# -fipa-cp-alignment -fipa-cp-clone -fipa-icf -fipa-icf-functions
# -fipa-icf-variables -fipa-profile -fipa-pure-const -fipa-ra
# -fipa-reference -fipa-sra -fira-hoist-pressure -fira-share-save-slots
# -fira-share-spill-slots -fisolate-erroneous-paths-dereference -fivopts
# -fkeep-static-consts -fleading-underscore -flifetime-dse -flra-remat
# -flto-odr-type-merging -fmath-errno -fmerge-constants
# -fmerge-debug-strings -fmove-loop-invariants -fnext-runtime
# -fobjc-abi-version= -fomit-frame-pointer -foptimize-sibling-calls
# -foptimize-strlen -fpartial-inlining -fpeephole -fpeephole2 -fplt
# -fpredictive-commoning -fprefetch-loop-arrays -free -freg-struct-return
# -freorder-blocks -freorder-functions -frerun-cse-after-loop
# -fsched-critical-path-heuristic -fsched-dep-count-heuristic
# -fsched-group-heuristic -fsched-interblock -fsched-last-insn-heuristic
# -fsched-rank-heuristic -fsched-spec -fsched-spec-insn-heuristic
# -fsched-stalled-insns-dep -fschedule-fusion -fschedule-insns2
# -fsemantic-interposition -fshow-column -fshrink-wrap -fsigned-zeros
# -fsplit-ivs-in-unroller -fsplit-paths -fsplit-wide-types -fssa-backprop
# -fssa-phiopt -fstdarg-opt -fstrict-aliasing -fstrict-overflow
# -fstrict-volatile-bitfields -fsync-libcalls -fthread-jumps
# -ftoplevel-reorder -ftrapping-math -ftree-bit-ccp -ftree-builtin-call-dce
# -ftree-ccp -ftree-ch -ftree-coalesce-vars -ftree-copy-prop -ftree-cselim
# -ftree-dce -ftree-dominator-opts -ftree-dse -ftree-forwprop -ftree-fre
# -ftree-loop-distribute-patterns -ftree-loop-if-convert -ftree-loop-im
# -ftree-loop-ivcanon -ftree-loop-optimize -ftree-loop-vectorize
# -ftree-parallelize-loops= -ftree-partial-pre -ftree-phiprop -ftree-pre
# -ftree-pta -ftree-reassoc -ftree-scev-cprop -ftree-sink
# -ftree-slp-vectorize -ftree-slsr -ftree-sra -ftree-switch-conversion
# -ftree-tail-merge -ftree-ter -ftree-vrp -funit-at-a-time -funswitch-loops
# -funwind-tables -fverbose-asm -fzero-initialized-in-bss -gstrict-dwarf
# -m128bit-long-double -m64 -m80387 -malign-stringops -matt-stubs
# -mconstant-cfstrings -mfancy-math-387 -mfp-ret-in-387 -mfxsr -mieee-fp
# -mlong-double-80 -mmmx -mno-sse4 -mpush-args -mred-zone -msse -msse2
# -msse3 -mstv -mvzeroupper

	.text
	.align 4,0x90
	.globl __Z8test_intv
__Z8test_intv:
LFB1042:
	movl	$2, %eax	#, tmp91
	movl	%eax, %edx	# tmp91, x
	addl	%edx, %eax	# x, tmp94
	ret
LFE1042:
	.align 4,0x90
	.globl __Z10test_arrayv
__Z10test_arrayv:
LFB1043:
	subq	$32, %rsp	#,
LCFI0:
	movdqa	LC0(%rip), %xmm0	#, tmp99
	movl	$2, -88(%rsp)	#, MEM[(int &)&x + 32]
	movaps	%xmm0, -120(%rsp)	# tmp99, MEM[(int &)&x]
	movaps	%xmm0, -104(%rsp)	# tmp99, MEM[(int &)&x + 16]
	movl	$2, -84(%rsp)	#, MEM[(int &)&x + 36]
	movaps	%xmm0, -72(%rsp)	# tmp99, MEM[(int &)&y]
	movaps	%xmm0, -56(%rsp)	# tmp99, MEM[(int &)&y + 16]
	movl	$2, -40(%rsp)	#, MEM[(int &)&y + 32]
	movl	$2, -36(%rsp)	#, MEM[(int &)&y + 36]
	movl	-88(%rsp), %eax	# x, x
	addl	-40(%rsp), %eax	# y, tmp110
	movdqa	-120(%rsp), %xmm0	# MEM[(value_type &)&x], MEM[(value_type &)&x]
	paddd	-72(%rsp), %xmm0	# MEM[(value_type &)&y], vect__4.52
	movaps	%xmm0, -24(%rsp)	# vect__4.52, MEM[(value_type &)&z]
	movdqa	-104(%rsp), %xmm0	# MEM[(value_type &)&x + 16], MEM[(value_type &)&x + 16]
	paddd	-56(%rsp), %xmm0	# MEM[(value_type &)&y + 16], vect__4.52
	movl	%eax, 8(%rsp)	# tmp110, z
	movl	-84(%rsp), %eax	# x, x
	addl	-36(%rsp), %eax	# y, tmp116
	movaps	%xmm0, -8(%rsp)	# vect__4.52, MEM[(value_type &)&z + 16]
	movl	%eax, 12(%rsp)	# tmp116, z
	addq	$32, %rsp	#,
LCFI1:
	ret
LFE1043:
	.align 4,0x90
	.globl __Z13test_longlongv
__Z13test_longlongv:
LFB1047:
	movl	$2, %eax	#, tmp91
	movq	%rax, %rdx	# tmp91, x
	addq	%rdx, %rax	# x, tmp94
	ret
LFE1047:
	.align 4,0x90
	.globl __Z15test_class_typev
__Z15test_class_typev:
LFB1053:
	movl	$2, %eax	#, D.28319
	movl	%eax, %edx	# D.28319, x
	addl	%edx, %eax	# x, tmp95
	ret
LFE1053:
	.align 4,0x90
	.globl __Z9check_dcev
__Z9check_dcev:
LFB1054:
	ret
LFE1054:
	.align 4,0x90
	.globl __Z16check_const_foldv
__Z16check_const_foldv:
LFB1055:
	ret
LFE1055:
	.literal16
	.align 4
LC0:
	.long	2
	.long	2
	.long	2
	.long	2
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x1
	.ascii "zR\0"
	.byte	0x1
	.byte	0x78
	.byte	0x10
	.byte	0x1
	.byte	0x10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x90
	.byte	0x1
	.align 3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB1042-.
	.set L$set$2,LFE1042-LFB1042
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1043-.
	.set L$set$4,LFE1043-LFB1043
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB1043
	.long L$set$5
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$7,LEFDE5-LASFDE5
	.long L$set$7
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB1047-.
	.set L$set$8,LFE1047-LFB1047
	.quad L$set$8
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$9,LEFDE7-LASFDE7
	.long L$set$9
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB1053-.
	.set L$set$10,LFE1053-LFB1053
	.quad L$set$10
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$11,LEFDE9-LASFDE9
	.long L$set$11
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB1054-.
	.set L$set$12,LFE1054-LFB1054
	.quad L$set$12
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$13,LEFDE11-LASFDE11
	.long L$set$13
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB1055-.
	.set L$set$14,LFE1055-LFB1055
	.quad L$set$14
	.byte	0
	.align 3
LEFDE11:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
