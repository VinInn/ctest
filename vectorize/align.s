	.text
	.align 4,0x90
	.globl __Z3fooR1VS0_S0_
__Z3fooR1VS0_S0_:
LFB6:
	movq	(%rdi), %rcx
	testq	%rcx, %rcx
	je	L21
	movq	(%rdx), %rdx
	leaq	8(%rcx), %rax
	negq	%rax
	movq	(%rsi), %rsi
	shrq	$3, %rax
	leaq	1(%rax), %r9
	leaq	32(%rcx), %rax
	leaq	32(%rdx), %rdi
	cmpq	%rdx, %rax
	setbe	%r8b
	cmpq	%rdi, %rcx
	setae	%dil
	orl	%edi, %r8d
	cmpq	%rsi, %rax
	leaq	32(%rsi), %rax
	setbe	%dil
	cmpq	%rax, %rcx
	setae	%al
	orl	%eax, %edi
	testb	%dil, %r8b
	je	L3
	cmpq	$6, %r9
	jbe	L3
	movq	%r9, %r10
	xorl	%eax, %eax
	xorl	%edi, %edi
	shrq	$2, %r10
	leaq	0(,%r10,4), %r8
L9:
	vmovupd	(%rsi,%rax), %xmm1
	addq	$1, %rdi
	vmovupd	(%rdx,%rax), %xmm0
	vinsertf128	$0x1, 16(%rsi,%rax), %ymm1, %ymm1
	vinsertf128	$0x1, 16(%rdx,%rax), %ymm0, %ymm0
	vaddpd	%ymm0, %ymm1, %ymm0
	vmovupd	%xmm0, (%rcx,%rax)
	vextractf128	$0x1, %ymm0, 16(%rcx,%rax)
	addq	$32, %rax
	cmpq	%rdi, %r10
	ja	L9
	leaq	(%rcx,%r8,8), %rcx
	cmpq	%r8, %r9
	je	L20
	movslq	%r8d, %rax
	cmpq	$-8, %rcx
	vmovsd	(%rsi,%rax,8), %xmm0
	vaddsd	(%rdx,%rax,8), %xmm0, %xmm0
	leal	1(%r8), %eax
	vmovsd	%xmm0, (%rcx)
	je	L20
	cltq
	addl	$2, %r8d
	cmpq	$-16, %rcx
	vmovsd	(%rsi,%rax,8), %xmm0
	vaddsd	(%rdx,%rax,8), %xmm0, %xmm0
	vmovsd	%xmm0, 8(%rcx)
	je	L20
	movslq	%r8d, %r8
	vmovsd	(%rsi,%r8,8), %xmm0
	vaddsd	(%rdx,%r8,8), %xmm0, %xmm0
	vmovsd	%xmm0, 16(%rcx)
	vzeroupper
	ret
	.align 4,0x90
L20:
	vzeroupper
L21:
	rep; ret
	.align 4,0x90
L3:
	salq	$3, %r9
	xorl	%eax, %eax
	.align 4,0x90
L11:
	vmovsd	(%rsi,%rax), %xmm0
	vaddsd	(%rdx,%rax), %xmm0, %xmm0
	vmovsd	%xmm0, (%rcx,%rax)
	addq	$8, %rax
	cmpq	%r9, %rax
	jne	L11
	rep; ret
LFE6:
	.align 4,0x90
	.globl __Z3barR2VAS0_S0_
__Z3barR2VAS0_S0_:
LFB7:
	movq	(%rdi), %rcx
	testq	%rcx, %rcx
	je	L43
	movq	(%rdx), %rdx
	leaq	8(%rcx), %rax
	negq	%rax
	movq	(%rsi), %rsi
	shrq	$3, %rax
	leaq	1(%rax), %r9
	leaq	32(%rdx), %rdi
	leaq	32(%rcx), %rax
	cmpq	%rdi, %rcx
	setae	%r8b
	cmpq	%rdx, %rax
	setbe	%dil
	orl	%edi, %r8d
	leaq	32(%rsi), %rdi
	cmpq	%rdi, %rcx
	setae	%dil
	cmpq	%rsi, %rax
	setbe	%al
	orl	%eax, %edi
	testb	%dil, %r8b
	je	L25
	cmpq	$5, %r9
	jbe	L25
	movq	%r9, %r10
	xorl	%eax, %eax
	xorl	%edi, %edi
	shrq	$2, %r10
	leaq	0(,%r10,4), %r8
L31:
	vmovapd	(%rsi,%rax), %ymm0
	addq	$1, %rdi
	vaddpd	(%rdx,%rax), %ymm0, %ymm0
	vmovapd	%ymm0, (%rcx,%rax)
	addq	$32, %rax
	cmpq	%rdi, %r10
	ja	L31
	leaq	(%rcx,%r8,8), %rcx
	cmpq	%r8, %r9
	je	L42
	movslq	%r8d, %rax
	cmpq	$-8, %rcx
	vmovsd	(%rsi,%rax,8), %xmm0
	vaddsd	(%rdx,%rax,8), %xmm0, %xmm0
	leal	1(%r8), %eax
	vmovsd	%xmm0, (%rcx)
	je	L42
	cltq
	addl	$2, %r8d
	cmpq	$-16, %rcx
	vmovsd	(%rsi,%rax,8), %xmm0
	vaddsd	(%rdx,%rax,8), %xmm0, %xmm0
	vmovsd	%xmm0, 8(%rcx)
	je	L42
	movslq	%r8d, %r8
	vmovsd	(%rsi,%r8,8), %xmm0
	vaddsd	(%rdx,%r8,8), %xmm0, %xmm0
	vmovsd	%xmm0, 16(%rcx)
	vzeroupper
	ret
	.align 4,0x90
L42:
	vzeroupper
L43:
	rep; ret
	.align 4,0x90
L25:
	salq	$3, %r9
	xorl	%eax, %eax
	.align 4,0x90
L33:
	vmovsd	(%rsi,%rax), %xmm0
	vaddsd	(%rdx,%rax), %xmm0, %xmm0
	vmovsd	%xmm0, (%rcx,%rax)
	addq	$8, %rax
	cmpq	%r9, %rax
	jne	L33
	rep; ret
LFE7:
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
	.quad	LFB6-.
	.set L$set$2,LFE6-LFB6
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB7-.
	.set L$set$4,LFE7-LFB7
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
