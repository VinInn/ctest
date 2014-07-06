	.text
	.align 4,0x90
	.globl __Z7adjDiffv
__Z7adjDiffv:
LFB0:
	vmovss	4+_sum(%rip), %xmm0
	leaq	32+_sum(%rip), %rax
	leaq	32+_diff(%rip), %rdx
	vsubss	_sum(%rip), %xmm0, %xmm1
	leaq	4096+_sum(%rip), %rcx
	vmovss	%xmm1, 4+_diff(%rip)
	vmovss	8+_sum(%rip), %xmm1
	vsubss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 8+_diff(%rip)
	vmovss	12+_sum(%rip), %xmm0
	vsubss	%xmm1, %xmm0, %xmm1
	vmovss	%xmm1, 12+_diff(%rip)
	vmovss	16+_sum(%rip), %xmm1
	vsubss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 16+_diff(%rip)
	vmovss	20+_sum(%rip), %xmm0
	vsubss	%xmm1, %xmm0, %xmm1
	vmovss	%xmm1, 20+_diff(%rip)
	vmovss	24+_sum(%rip), %xmm1
	vsubss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, 24+_diff(%rip)
	vmovss	28+_sum(%rip), %xmm0
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 28+_diff(%rip)
	.align 4,0x90
L3:
	vmovups	-4(%rax), %xmm0
	addq	$32, %rax
	addq	$32, %rdx
	vmovaps	-32(%rax), %ymm1
	vinsertf128	$0x1, -20(%rax), %ymm0, %ymm0
	vsubps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, -32(%rdx)
	cmpq	%rcx, %rax
	jne	L3
	vzeroupper
	ret
LFE0:
	.align 4,0x90
	.globl __Z8adjDiff1v
__Z8adjDiff1v:
LFB1:
	leaq	_diff(%rip), %rax
	leaq	4096+_diff(%rip), %rdx
	.align 4,0x90
L7:
	vmovups	4(%rax), %xmm0
	addq	$32, %rax
	vmovaps	-32(%rax), %ymm1
	vinsertf128	$0x1, -12(%rax), %ymm0, %ymm0
	vsubps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, -32(%rax)
	cmpq	%rdx, %rax
	jne	L7
	vzeroupper
	ret
LFE1:
	.globl _diff
	.zerofill __DATA,__pu_bss5,_diff,4100,5
	.globl _sum
	.zerofill __DATA,__pu_bss5,_sum,4096,5
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
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1-.
	.set L$set$4,LFE1-LFB1
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
