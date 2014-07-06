	.text
	.align 4,0x90
	.globl __Z2nnR3Loc
__Z2nnR3Loc:
LFB228:
	movl	12(%rdi), %ecx
	leaq	_loc(%rip), %rax
	vmovss	LC0(%rip), %xmm2
	vmovss	(%rdi), %xmm3
	leaq	16384+_loc(%rip), %rdx
	vmovss	8(%rdi), %xmm1
	.align 4,0x90
L2:
	vsubss	(%rax), %xmm3, %xmm0
	vandps	%xmm2, %xmm0, %xmm0
	vcomiss	%xmm0, %xmm1
	jbe	L3
	movl	4(%rax), %ecx
	vmovaps	%xmm0, %xmm1
L3:
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	L2
	vmovss	%xmm1, 8(%rdi)
	movl	%ecx, 12(%rdi)
	ret
LFE228:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
__GLOBAL__sub_I_NearN_omp4.cc:
LFB230:
	leaq	_loc(%rip), %rax
	leaq	16384+_loc(%rip), %rdx
	.align 4
L9:
	movl	$0x4b18967f, 8(%rax)
	addq	$16, %rax
	movl	$-1, -4(%rax)
	cmpq	%rdx, %rax
	jne	L9
	rep; ret
LFE230:
	.globl _loc
	.zerofill __DATA,__pu_bss5,_loc,16384,5
	.literal16
	.align 4
LC0:
	.long	2147483647
	.long	0
	.long	0
	.long	0
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
	.quad	LFB228-.
	.set L$set$2,LFE228-LFB228
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB230-.
	.set L$set$4,LFE230-LFB230
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_NearN_omp4.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
