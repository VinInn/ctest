	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB13:
	.text
LHOTB13:
	.align 4,0x90
	.globl __Z4normP3XYZS0_i
__Z4normP3XYZS0_i:
LFB226:
	testl	%edx, %edx
	jle	L1
	leal	-4(%rdx), %eax
	leal	-1(%rdx), %r8d
	shrl	$2, %eax
	addl	$1, %eax
	leal	0(,%rax,4), %ecx
	cmpl	$2, %r8d
	jbe	L9
	movdqa	LC0(%rip), %xmm6
	movq	%rdi, %r9
	movq	%rsi, %r8
	xorl	%r10d, %r10d
	movdqa	LC1(%rip), %xmm15
	movdqa	LC2(%rip), %xmm14
	movdqa	LC3(%rip), %xmm13
	movdqa	LC4(%rip), %xmm12
	movdqa	LC5(%rip), %xmm11
	movaps	LC6(%rip), %xmm10
	movaps	LC7(%rip), %xmm9
	movdqa	LC8(%rip), %xmm8
	movdqa	LC9(%rip), %xmm7
L4:
	movups	(%r9), %xmm0
	addl	$1, %r10d
	addq	$48, %r9
	addq	$48, %r8
	movups	-32(%r9), %xmm1
	movups	-16(%r9), %xmm5
	movdqa	%xmm0, %xmm3
	movdqa	%xmm1, %xmm2
	pshufb	%xmm6, %xmm3
	pshufb	%xmm15, %xmm2
	por	%xmm3, %xmm2
	movaps	%xmm5, %xmm3
	movaps	%xmm2, %xmm4
	shufps	$100, %xmm5, %xmm3
	blendps	$8, %xmm3, %xmm4
	movdqa	%xmm1, %xmm2
	pshufb	%xmm11, %xmm1
	movaps	%xmm4, %xmm3
	movdqa	%xmm0, %xmm4
	pshufb	%xmm13, %xmm2
	pshufb	%xmm12, %xmm0
	pshufb	%xmm14, %xmm4
	por	%xmm2, %xmm4
	movaps	%xmm5, %xmm2
	por	%xmm1, %xmm0
	shufps	$164, %xmm5, %xmm2
	blendps	$8, %xmm2, %xmm4
	movaps	%xmm4, %xmm2
	movaps	%xmm0, %xmm4
	movaps	%xmm3, %xmm0
	mulps	%xmm3, %xmm0
	movaps	%xmm2, %xmm1
	shufps	$196, %xmm5, %xmm4
	mulps	%xmm2, %xmm1
	addps	%xmm1, %xmm0
	movaps	%xmm4, %xmm1
	mulps	%xmm4, %xmm1
	addps	%xmm1, %xmm0
	rsqrtps	%xmm0, %xmm1
	mulps	%xmm1, %xmm0
	mulps	%xmm1, %xmm0
	mulps	%xmm9, %xmm1
	addps	%xmm10, %xmm0
	mulps	%xmm1, %xmm0
	mulps	%xmm0, %xmm3
	mulps	%xmm0, %xmm2
	mulps	%xmm4, %xmm0
	movaps	%xmm3, %xmm1
	unpcklps	%xmm2, %xmm1
	shufps	$132, %xmm1, %xmm1
	movaps	%xmm0, %xmm4
	shufps	$196, %xmm0, %xmm4
	blendps	$4, %xmm4, %xmm1
	movdqa	%xmm3, %xmm4
	pshufb	%xmm6, %xmm3
	movups	%xmm1, -48(%r8)
	movdqa	%xmm2, %xmm1
	pshufb	%xmm8, %xmm4
	pshufb	LC10(%rip), %xmm2
	por	%xmm3, %xmm2
	movaps	%xmm2, %xmm3
	pshufb	%xmm7, %xmm1
	por	%xmm4, %xmm1
	blendps	$2, %xmm0, %xmm1
	shufps	$230, %xmm0, %xmm0
	blendps	$9, %xmm0, %xmm3
	movups	%xmm1, -32(%r8)
	movups	%xmm3, -16(%r8)
	cmpl	%r10d, %eax
	ja	L4
	cmpl	%ecx, %edx
	je	L14
L3:
	movss	LC12(%rip), %xmm3
	movslq	%ecx, %rax
	movss	LC11(%rip), %xmm4
	leaq	(%rax,%rax,2), %rax
	salq	$2, %rax
	leaq	(%rdi,%rax), %r8
	movss	(%r8), %xmm6
	movss	4(%r8), %xmm5
	movaps	%xmm6, %xmm0
	movss	8(%r8), %xmm2
	mulss	%xmm6, %xmm0
	movaps	%xmm5, %xmm1
	mulss	%xmm5, %xmm1
	leaq	(%rsi,%rax), %r8
	addss	%xmm1, %xmm0
	movaps	%xmm2, %xmm1
	mulss	%xmm2, %xmm1
	addss	%xmm1, %xmm0
	rsqrtss	%xmm0, %xmm1
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	%xmm3, %xmm1
	addss	%xmm4, %xmm0
	mulss	%xmm1, %xmm0
	mulss	%xmm0, %xmm6
	mulss	%xmm0, %xmm5
	mulss	%xmm2, %xmm0
	movss	%xmm6, (%r8)
	movss	%xmm5, 4(%r8)
	movss	%xmm0, 8(%r8)
	leal	1(%rcx), %r8d
	cmpl	%r8d, %edx
	jle	L1
	leaq	12(%rax), %r8
	addl	$2, %ecx
	leaq	(%rdi,%r8), %r9
	addq	%rsi, %r8
	movss	(%r9), %xmm6
	movss	4(%r9), %xmm5
	movaps	%xmm6, %xmm0
	movss	8(%r9), %xmm7
	mulss	%xmm6, %xmm0
	movaps	%xmm5, %xmm2
	mulss	%xmm5, %xmm2
	movaps	%xmm7, %xmm1
	mulss	%xmm7, %xmm1
	addss	%xmm0, %xmm2
	addss	%xmm1, %xmm2
	rsqrtss	%xmm2, %xmm1
	mulss	%xmm1, %xmm2
	mulss	%xmm1, %xmm2
	mulss	%xmm3, %xmm1
	addss	%xmm4, %xmm2
	mulss	%xmm1, %xmm2
	mulss	%xmm2, %xmm6
	movaps	%xmm2, %xmm0
	mulss	%xmm2, %xmm5
	mulss	%xmm7, %xmm0
	movss	%xmm6, (%r8)
	movss	%xmm5, 4(%r8)
	movss	%xmm0, 8(%r8)
	cmpl	%ecx, %edx
	jle	L1
	addq	$24, %rax
	addq	%rax, %rdi
	addq	%rax, %rsi
	movss	(%rdi), %xmm6
	movss	4(%rdi), %xmm5
	movaps	%xmm6, %xmm0
	movss	8(%rdi), %xmm2
	mulss	%xmm6, %xmm0
	movaps	%xmm5, %xmm1
	mulss	%xmm5, %xmm1
	addss	%xmm1, %xmm0
	movaps	%xmm2, %xmm1
	mulss	%xmm2, %xmm1
	addss	%xmm1, %xmm0
	rsqrtss	%xmm0, %xmm1
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm3
	mulss	%xmm1, %xmm0
	addss	%xmm4, %xmm0
	mulss	%xmm3, %xmm0
	mulss	%xmm0, %xmm6
	mulss	%xmm0, %xmm5
	mulss	%xmm2, %xmm0
	movss	%xmm6, (%rsi)
	movss	%xmm5, 4(%rsi)
	movss	%xmm0, 8(%rsi)
	ret
	.align 4,0x90
L1:
	ret
	.align 4,0x90
L14:
	ret
	.align 4,0x90
L9:
	xorl	%ecx, %ecx
	jmp	L3
LFE226:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE13:
	.text
LHOTE13:
	.literal16
	.align 4
LC0:
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.align 4
LC1:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 4
LC2:
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.align 4
LC3:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 4
LC4:
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.align 4
LC5:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 4
LC6:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 4
LC7:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.align 4
LC8:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 4
LC9:
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.align 4
LC10:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.literal4
	.align 2
LC11:
	.long	3225419776
	.align 2
LC12:
	.long	3204448256
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
	.quad	LFB226-.
	.set L$set$2,LFE226-LFB226
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
