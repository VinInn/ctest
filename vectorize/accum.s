	.text
	.align 4,0x90
	.globl __Z3accv
__Z3accv:
LFB0:
	vmovaps	_a(%rip), %ymm0
	leaq	32+_a(%rip), %rax
	leaq	4096+_a(%rip), %rdx
	.align 4,0x90
L2:
	vmulps	(%rax), %ymm0, %ymm0
	addq	$32, %rax
	vmovaps	%ymm0, -32(%rax)
	cmpq	%rdx, %rax
	jne	L2
	vzeroupper
	ret
LFE0:
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,4096,5
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
	.quad	LFB0-.
	.set L$set$2,LFE0-LFB0
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
