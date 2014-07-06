	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z4sum1PKfS0_S0_.arch_haswell
__Z4sum1PKfS0_S0_.arch_haswell:
LFB1:
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	.align 4,0x90
L2:
	vmovss	(%rdi,%rax), %xmm1
	vmovss	(%rdx,%rax), %xmm2
	vfmadd132ss	(%rsi,%rax), %xmm2, %xmm1
	addq	$4, %rax
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	$4096, %rax
	jne	L2
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB2:
	.text
LHOTB2:
	.align 4,0x90
	.globl __Z4sum1PKfS0_S0_.arch_nehalem
__Z4sum1PKfS0_S0_.arch_nehalem:
LFB2:
	xorl	%eax, %eax
	pxor	%xmm0, %xmm0
	.align 4,0x90
L6:
	movss	(%rdi,%rax), %xmm1
	mulss	(%rsi,%rax), %xmm1
	addss	(%rdx,%rax), %xmm1
	addq	$4, %rax
	cmpq	$4096, %rax
	addss	%xmm1, %xmm0
	jne	L6
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __Z4sum0PKfS0_S0_
__Z4sum0PKfS0_S0_:
LFB3:
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	.align 4,0x90
L10:
	vmovss	(%rdi,%rax), %xmm1
	vmovss	(%rdx,%rax), %xmm2
	vfmadd132ss	(%rsi,%rax), %xmm2, %xmm1
	addq	$4, %rax
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	$4096, %rax
	jne	L10
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
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
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB2-.
	.set L$set$4,LFE2-LFB2
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3-.
	.set L$set$6,LFE3-LFB3
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
