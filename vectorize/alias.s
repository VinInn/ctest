	.text
	.align 4,0x90
	.globl __Z3sumv
__Z3sumv:
LFB0:
	leaq	4096+_mem(%rip), %rax
	leaq	8192+_mem(%rip), %rdx
	.align 4,0x90
L3:
	movaps	(%rax), %xmm0
	addq	$16, %rax
	addps	4080(%rax), %xmm0
	movaps	%xmm0, -4112(%rax)
	cmpq	%rdx, %rax
	jne	L3
	rep; ret
LFE0:
	.align 4,0x90
	.globl __Z4sumNi
__Z4sumNi:
LFB1:
	leaq	_mem(%rip), %rdx
	movslq	%edi, %rax
	testl	%edi, %edi
	leaq	(%rdx,%rax,4), %rcx
	leal	(%rdi,%rdi), %eax
	cltq
	leaq	(%rdx,%rax,4), %rsi
	je	L5
	leaq	16(%rcx), %rax
	leaq	16+_mem(%rip), %r8
	cmpq	%rdx, %rax
	setbe	%al
	cmpq	%r8, %rcx
	setae	%r9b
	orl	%r9d, %eax
	cmpl	$6, %edi
	seta	%r9b
	testb	%r9b, %al
	je	L7
	leaq	16(%rsi), %rax
	cmpq	%rdx, %rax
	setbe	%r9b
	cmpq	%r8, %rsi
	setae	%al
	orb	%al, %r9b
	je	L7
	movl	%edi, %r11d
	xorl	%eax, %eax
	xorl	%r8d, %r8d
	shrl	$2, %r11d
	leal	0(,%r11,4), %r10d
L11:
	movups	(%rcx,%rax), %xmm0
	leaq	_mem(%rip), %r9
	addl	$1, %r8d
	movups	(%rsi,%rax), %xmm1
	addps	%xmm1, %xmm0
	movaps	%xmm0, (%rdx,%rax)
	addq	$16, %rax
	cmpl	%r11d, %r8d
	jb	L11
	cmpl	%r10d, %edi
	je	L5
	movslq	%r10d, %rax
	movss	(%rcx,%rax,4), %xmm0
	addss	(%rsi,%rax,4), %xmm0
	movss	%xmm0, (%r9,%rax,4)
	leal	1(%r10), %eax
	cmpl	%eax, %edi
	je	L5
	cltq
	addl	$2, %r10d
	movss	(%rcx,%rax,4), %xmm0
	cmpl	%r10d, %edi
	addss	(%rsi,%rax,4), %xmm0
	movss	%xmm0, (%r9,%rax,4)
	je	L5
	movslq	%r10d, %r10
	movss	(%rcx,%r10,4), %xmm0
	addss	(%rsi,%r10,4), %xmm0
	movss	%xmm0, (%r9,%r10,4)
	ret
	.align 4,0x90
L7:
	xorl	%eax, %eax
	.align 4,0x90
L13:
	movss	(%rcx,%rax,4), %xmm0
	addss	(%rsi,%rax,4), %xmm0
	movss	%xmm0, (%rdx,%rax,4)
	addq	$1, %rax
	cmpl	%eax, %edi
	jne	L13
L5:
	rep; ret
LFE1:
	.globl _mem
	.zerofill __DATA,__pu_bss5,_mem,12288,5
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
