	.text
	.align 4,0x90
	.globl __Z5sumO1v
__Z5sumO1v:
LFB0:
	pushq	%rbp
LCFI0:
	movq	%rsp, %rbp
LCFI1:
	andq	$-32, %rsp
	addq	$16, %rsp
	leaq	-48(%rsp), %rax
	leaq	-16(%rsp), %rcx
	movq	%rax, %rdx
	.align 4,0x90
L3:
	movl	$0x00000000, (%rdx)
	addq	$4, %rdx
	cmpq	%rcx, %rdx
	jne	L3
	vmovaps	-48(%rsp), %ymm0
	leaq	_a(%rip), %rsi
	xorl	%edx, %edx
	leaq	_b(%rip), %rcx
	.align 4,0x90
L7:
	vmovaps	(%rsi,%rdx), %ymm1
	vfmadd231ps	(%rcx,%rdx), %ymm1, %ymm0
	addq	$32, %rdx
	cmpq	$4096, %rdx
	jne	L7
	leaq	-16(%rsp), %rdx
	vmovaps	%ymm0, -48(%rsp)
	vxorps	%xmm0, %xmm0, %xmm0
	.align 4,0x90
L6:
	vaddss	(%rax), %xmm0, %xmm0
	addq	$4, %rax
	cmpq	%rdx, %rax
	jne	L6
	vzeroupper
	leave
LCFI2:
	ret
LFE0:
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,4096,5
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
	.byte	0x4
	.set L$set$3,LCFI0-LFB0
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
