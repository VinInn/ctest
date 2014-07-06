	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z2gov
__Z2gov:
LFB0:
	movss	60+_x(%rip), %xmm3
	movl	$1000, %esi
	leaq	_y(%rip), %rcx
	leaq	_x(%rip), %rdx
	.align 4,0x90
L2:
	movaps	%xmm3, %xmm2
	xorl	%eax, %eax
	pxor	%xmm1, %xmm1
	shufps	$0, %xmm2, %xmm2
	.align 4,0x90
L3:
	movaps	(%rcx,%rax), %xmm0
	mulps	(%rdx,%rax), %xmm0
	addq	$16, %rax
	cmpq	$4096, %rax
	addps	%xmm2, %xmm0
	addps	%xmm0, %xmm1
	jne	L3
	haddps	%xmm1, %xmm1
	subl	$1, %esi
	haddps	%xmm1, %xmm1
	movaps	%xmm1, %xmm0
	addss	%xmm0, %xmm3
	movss	%xmm3, 60+_x(%rip)
	jne	L2
	pxor	%xmm0, %xmm0
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.globl _q
	.zerofill __DATA,__pu_bss2,_q,4,2
	.globl _y
	.zerofill __DATA,__pu_bss6,_y,4096,6
	.globl _x
	.zerofill __DATA,__pu_bss6,_x,4096,6
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
