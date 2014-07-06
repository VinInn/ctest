	.text
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB3:
	movaps	LC0(%rip), %xmm1
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	leaq	_b(%rip), %rdx
	.align 4,0x90
L3:
	movaps	(%rcx,%rax), %xmm0
	mulps	%xmm0, %xmm0
	mulps	%xmm1, %xmm0
	movaps	%xmm0, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$2048, %rax
	jne	L3
	leaq	_a(%rip), %rsi
	movl	$2048, %edx
	leaq	2048+_b(%rip), %rdi
	jmp	_memcpy
LFE3:
	.align 4,0x90
	.globl __Z4bar2v
__Z4bar2v:
LFB4:
	movss	LC1(%rip), %xmm3
	leaq	_a(%rip), %rax
	leaq	_b(%rip), %rdx
	leaq	2048+_a(%rip), %rcx
	.align 4,0x90
L7:
	movss	2048(%rax), %xmm0
	addq	$4, %rax
	addq	$4, %rdx
	movss	-4(%rax), %xmm2
	mulss	%xmm0, %xmm0
	movaps	%xmm0, %xmm1
	mulss	%xmm3, %xmm1
	mulss	%xmm2, %xmm0
	mulss	%xmm2, %xmm1
	movss	%xmm0, 2044(%rdx)
	mulss	%xmm2, %xmm1
	movss	%xmm1, -4(%rdx)
	cmpq	%rcx, %rax
	jne	L7
	rep; ret
LFE4:
	.align 4,0x90
	.globl __Z4bar3v
__Z4bar3v:
LFB5:
	movaps	LC0(%rip), %xmm3
	leaq	_b(%rip), %rax
	leaq	2048+_a(%rip), %rdx
	leaq	2048+_b(%rip), %rcx
	.align 4,0x90
L9:
	movaps	(%rdx), %xmm2
	addq	$16, %rax
	addq	$16, %rdx
	movaps	-2064(%rdx), %xmm0
	movaps	%xmm2, %xmm1
	mulps	%xmm2, %xmm1
	mulps	%xmm0, %xmm0
	mulps	%xmm3, %xmm1
	mulps	%xmm0, %xmm1
	mulps	%xmm2, %xmm0
	movaps	%xmm1, -16(%rax)
	movaps	%xmm0, 2032(%rax)
	cmpq	%rcx, %rax
	jne	L9
	rep; ret
LFE5:
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,4096,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,4096,5
	.literal16
	.align 4
LC0:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.literal4
	.align 2
LC1:
	.long	1056964608
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
	.quad	LFB3-.
	.set L$set$2,LFE3-LFB3
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB4-.
	.set L$set$4,LFE4-LFB4
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB5-.
	.set L$set$6,LFE5-LFB5
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
