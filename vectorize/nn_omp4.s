	.text
	.align 4,0x90
	.globl __Z8selfLessRff
__Z8selfLessRff:
LFB4084:
	minss	(%rdi), %xmm0
	movss	%xmm0, (%rdi)
	ret
LFE4084:
	.align 4,0x90
	.globl __Z6nearNSi
__Z6nearNSi:
LFB4087:
	movq	$0, -40(%rsp)
	movslq	%edi, %rdi
	movabsq	$-4294967296, %rcx
	movq	-40(%rsp), %rax
	leaq	_distNN(%rip), %r8
	movabsq	$9187343235540844544, %rdx
	movq	$0, -32(%rsp)
	movss	(%r8,%rdi,4), %xmm7
	leaq	_index(%rip), %rsi
	movq	$0, -24(%rsp)
	movq	$0, -16(%rsp)
	andq	%rcx, %rax
	movdqa	-24(%rsp), %xmm3
	orq	$2139095039, %rax
	movl	%eax, %eax
	orq	%rdx, %rax
	movq	%rax, -40(%rsp)
	movq	-32(%rsp), %rax
	andq	%rcx, %rax
	leaq	_phi(%rip), %rcx
	orq	$2139095039, %rax
	movl	%eax, %eax
	movss	(%rcx,%rdi,4), %xmm6
	orq	%rdx, %rax
	movq	%rax, -32(%rsp)
	leaq	_eta(%rip), %rdx
	xorl	%eax, %eax
	movss	(%rdx,%rdi,4), %xmm5
	shufps	$0, %xmm6, %xmm6
	movaps	-40(%rsp), %xmm2
	shufps	$0, %xmm5, %xmm5
	.align 4,0x90
L3:
	movaps	(%rcx,%rax), %xmm1
	movaps	(%rdx,%rax), %xmm0
	subps	%xmm6, %xmm1
	movdqa	(%rsi,%rax), %xmm4
	addq	$16, %rax
	subps	%xmm5, %xmm0
	cmpq	$4096, %rax
	mulps	%xmm1, %xmm1
	mulps	%xmm0, %xmm0
	addps	%xmm0, %xmm1
	movaps	%xmm2, %xmm0
	cmpleps	%xmm1, %xmm0
	minps	%xmm1, %xmm2
	pblendvb	%xmm0, %xmm3, %xmm4
	movdqa	%xmm4, %xmm3
	jne	L3
	movaps	%xmm2, -40(%rsp)
	movaps	%xmm7, %xmm0
	movq	-40(%rsp), %rax
	movdqa	%xmm4, -24(%rsp)
	movl	-12(%rsp), %edx
	movd	%eax, %xmm7
	shrq	$32, %rax
	minss	%xmm7, %xmm0
	movd	%eax, %xmm7
	movq	-32(%rsp), %rax
	minss	%xmm7, %xmm0
	movd	%eax, %xmm7
	shrq	$32, %rax
	minss	%xmm7, %xmm0
	movd	%eax, %xmm7
	leaq	_nn(%rip), %rax
	movl	%edx, (%rax,%rdi,4)
	minss	%xmm7, %xmm0
	movss	%xmm0, (%r8,%rdi,4)
	ret
LFE4087:
	.globl _index
	.zerofill __DATA,__pu_bss5,_index,4096,5
	.globl _nn
	.zerofill __DATA,__pu_bss5,_nn,4096,5
	.globl _distNN
	.zerofill __DATA,__pu_bss5,_distNN,4096,5
	.globl _phi
	.zerofill __DATA,__pu_bss5,_phi,4096,5
	.globl _eta
	.zerofill __DATA,__pu_bss5,_eta,4096,5
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
	.quad	LFB4084-.
	.set L$set$2,LFE4084-LFB4084
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB4087-.
	.set L$set$4,LFE4087-LFB4087
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
