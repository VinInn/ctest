	.text
	.align 4,0x90
	.globl __Z7computev
__Z7computev:
LFB3420:
	movaps	LC0(%rip), %xmm4
	xorl	%eax, %eax
	xorps	%xmm5, %xmm5
	movaps	LC25(%rip), %xmm14
	leaq	_a(%rip), %rcx
	movaps	LC26(%rip), %xmm13
	leaq	_b(%rip), %rdx
	movaps	LC27(%rip), %xmm12
	movaps	LC28(%rip), %xmm11
	movaps	LC29(%rip), %xmm10
	movaps	LC36(%rip), %xmm9
	movaps	LC37(%rip), %xmm8
	movaps	LC38(%rip), %xmm7
	movaps	LC39(%rip), %xmm6
	.align 4,0x90
L3:
	movaps	(%rcx,%rax), %xmm3
	movaps	%xmm4, %xmm15
	movdqa	LC2(%rip), %xmm1
	movaps	%xmm3, %xmm0
	subps	%xmm3, %xmm15
	addps	%xmm4, %xmm0
	mulps	LC20(%rip), %xmm3
	mulps	%xmm0, %xmm15
	movdqa	%xmm15, %xmm2
	pand	%xmm15, %xmm1
	por	LC3(%rip), %xmm1
	psrad	$22, %xmm2
	pand	LC1(%rip), %xmm2
	movdqa	%xmm2, %xmm0
	pslld	$23, %xmm0
	psubd	%xmm0, %xmm1
	movaps	LC4(%rip), %xmm0
	subps	%xmm4, %xmm1
	mulps	%xmm1, %xmm0
	addps	LC5(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC6(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC7(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC8(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC9(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC10(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC11(%rip), %xmm0
	mulps	%xmm1, %xmm0
	movdqa	%xmm15, %xmm1
	psrad	$23, %xmm1
	pand	LC12(%rip), %xmm1
	paddd	LC13(%rip), %xmm1
	paddd	%xmm2, %xmm1
	movaps	%xmm5, %xmm2
	cvtdq2ps	%xmm1, %xmm1
	mulps	LC14(%rip), %xmm1
	addps	%xmm1, %xmm0
	xorps	LC15(%rip), %xmm0
	rsqrtps	%xmm0, %xmm15
	movaps	%xmm0, %xmm1
	cmpneqps	%xmm0, %xmm2
	subps	LC16(%rip), %xmm1
	andps	%xmm2, %xmm15
	movaps	%xmm15, %xmm2
	mulps	%xmm0, %xmm2
	cmpltps	LC21(%rip), %xmm0
	mulps	%xmm2, %xmm15
	mulps	LC18(%rip), %xmm2
	addps	LC17(%rip), %xmm15
	mulps	%xmm2, %xmm15
	movaps	LC22(%rip), %xmm2
	mulps	%xmm1, %xmm2
	subps	LC19(%rip), %xmm15
	addps	LC23(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC24(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	%xmm14, %xmm2
	mulps	%xmm1, %xmm2
	addps	%xmm13, %xmm2
	mulps	%xmm1, %xmm2
	addps	%xmm12, %xmm2
	mulps	%xmm1, %xmm2
	addps	%xmm11, %xmm2
	mulps	%xmm1, %xmm2
	addps	%xmm10, %xmm2
	mulps	%xmm1, %xmm2
	movaps	LC31(%rip), %xmm1
	mulps	%xmm15, %xmm1
	addps	LC30(%rip), %xmm2
	addps	LC32(%rip), %xmm1
	mulps	%xmm15, %xmm1
	addps	LC33(%rip), %xmm1
	mulps	%xmm15, %xmm1
	addps	LC34(%rip), %xmm1
	mulps	%xmm15, %xmm1
	addps	LC35(%rip), %xmm1
	mulps	%xmm15, %xmm1
	addps	%xmm9, %xmm1
	mulps	%xmm15, %xmm1
	addps	%xmm8, %xmm1
	mulps	%xmm15, %xmm1
	addps	%xmm7, %xmm1
	mulps	%xmm15, %xmm1
	addps	%xmm6, %xmm1
	blendvps	%xmm0, %xmm2, %xmm1
	mulps	%xmm1, %xmm3
	movaps	%xmm3, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$32768, %rax
	jne	L3
	rep; ret
LFE3420:
	.align 4,0x90
	.globl __Z8computeBv
__Z8computeBv:
LFB3421:
	leaq	_a(%rip), %r9
	subq	$8080, %rsp
LCFI0:
	leaq	-120(%rsp), %rsi
	movaps	LC0(%rip), %xmm10
	movq	%r9, %rax
	leaq	_b(%rip), %r10
	movq	%rsi, %rcx
	movdqa	LC1(%rip), %xmm9
	leaq	32768+_a(%rip), %rdi
	movq	%r10, %rdx
	.align 4,0x90
L7:
	movaps	(%rax), %xmm15
	movaps	%xmm10, %xmm8
	movaps	%xmm10, %xmm7
	subps	16(%rax), %xmm7
	movaps	%xmm10, %xmm6
	movaps	%xmm10, %xmm5
	movaps	%xmm15, %xmm0
	subps	%xmm15, %xmm8
	movdqa	LC2(%rip), %xmm4
	addps	%xmm10, %xmm0
	movdqa	LC2(%rip), %xmm3
	addq	$64, %rax
	subps	-32(%rax), %xmm6
	addq	$16, %rcx
	addq	$64, %rdx
	subps	-16(%rax), %xmm5
	mulps	%xmm0, %xmm8
	movaps	-48(%rax), %xmm0
	movdqa	LC2(%rip), %xmm2
	addps	%xmm10, %xmm0
	movdqa	LC2(%rip), %xmm1
	movdqa	%xmm8, %xmm14
	pand	%xmm8, %xmm4
	psrad	$23, %xmm8
	mulps	%xmm0, %xmm7
	movaps	-32(%rax), %xmm0
	psrad	$22, %xmm14
	pand	%xmm9, %xmm14
	por	LC3(%rip), %xmm4
	pand	LC12(%rip), %xmm8
	paddd	LC13(%rip), %xmm8
	addps	%xmm10, %xmm0
	movdqa	%xmm7, %xmm13
	pand	%xmm7, %xmm3
	psrad	$23, %xmm7
	mulps	%xmm0, %xmm6
	movaps	-16(%rax), %xmm0
	psrad	$22, %xmm13
	pand	%xmm9, %xmm13
	por	LC3(%rip), %xmm3
	pand	LC12(%rip), %xmm7
	addps	%xmm10, %xmm0
	paddd	LC13(%rip), %xmm7
	paddd	%xmm14, %xmm8
	cvtdq2ps	%xmm8, %xmm8
	mulps	LC14(%rip), %xmm8
	movdqa	%xmm6, %xmm12
	pand	%xmm6, %xmm2
	paddd	%xmm13, %xmm7
	mulps	%xmm0, %xmm5
	movdqa	%xmm14, %xmm0
	por	LC3(%rip), %xmm2
	pslld	$23, %xmm0
	cvtdq2ps	%xmm7, %xmm7
	mulps	LC14(%rip), %xmm7
	psubd	%xmm0, %xmm4
	psrad	$22, %xmm12
	movdqa	%xmm13, %xmm0
	pslld	$23, %xmm0
	pand	%xmm9, %xmm12
	movdqa	%xmm5, %xmm11
	psubd	%xmm0, %xmm3
	movdqa	%xmm12, %xmm0
	pslld	$23, %xmm0
	pand	%xmm5, %xmm1
	por	LC3(%rip), %xmm1
	psrad	$22, %xmm11
	psubd	%xmm0, %xmm2
	pand	%xmm9, %xmm11
	movdqa	%xmm11, %xmm0
	subps	%xmm10, %xmm4
	pslld	$23, %xmm0
	psubd	%xmm0, %xmm1
	movaps	LC4(%rip), %xmm0
	subps	%xmm10, %xmm3
	subps	%xmm10, %xmm2
	mulps	%xmm4, %xmm0
	subps	%xmm10, %xmm1
	psrad	$23, %xmm5
	pand	LC12(%rip), %xmm5
	paddd	LC13(%rip), %xmm5
	psrad	$23, %xmm6
	pand	LC12(%rip), %xmm6
	paddd	LC13(%rip), %xmm6
	addps	LC5(%rip), %xmm0
	paddd	%xmm11, %xmm5
	cvtdq2ps	%xmm5, %xmm5
	mulps	LC14(%rip), %xmm5
	paddd	%xmm12, %xmm6
	cvtdq2ps	%xmm6, %xmm6
	mulps	LC14(%rip), %xmm6
	mulps	%xmm4, %xmm0
	addps	LC6(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC7(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC8(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC9(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC10(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC11(%rip), %xmm0
	mulps	%xmm4, %xmm0
	movaps	LC4(%rip), %xmm4
	mulps	%xmm3, %xmm4
	addps	%xmm8, %xmm0
	addps	LC5(%rip), %xmm4
	xorps	LC15(%rip), %xmm0
	mulps	%xmm3, %xmm4
	addps	LC6(%rip), %xmm4
	mulps	%xmm3, %xmm4
	addps	LC7(%rip), %xmm4
	mulps	%xmm3, %xmm4
	addps	LC8(%rip), %xmm4
	mulps	%xmm3, %xmm4
	addps	LC9(%rip), %xmm4
	mulps	%xmm3, %xmm4
	addps	LC10(%rip), %xmm4
	mulps	%xmm3, %xmm4
	addps	LC11(%rip), %xmm4
	mulps	%xmm3, %xmm4
	movaps	LC4(%rip), %xmm3
	mulps	%xmm2, %xmm3
	addps	%xmm7, %xmm4
	addps	LC5(%rip), %xmm3
	xorps	LC15(%rip), %xmm4
	mulps	%xmm2, %xmm3
	addps	LC6(%rip), %xmm3
	mulps	%xmm2, %xmm3
	addps	LC7(%rip), %xmm3
	mulps	%xmm2, %xmm3
	addps	LC8(%rip), %xmm3
	mulps	%xmm2, %xmm3
	addps	LC9(%rip), %xmm3
	mulps	%xmm2, %xmm3
	addps	LC10(%rip), %xmm3
	mulps	%xmm2, %xmm3
	addps	LC11(%rip), %xmm3
	mulps	%xmm2, %xmm3
	movaps	LC4(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	%xmm6, %xmm3
	movaps	LC21(%rip), %xmm6
	addps	LC5(%rip), %xmm2
	xorps	LC15(%rip), %xmm3
	cmpltps	%xmm3, %xmm6
	mulps	%xmm1, %xmm2
	pand	%xmm9, %xmm6
	pshufb	LC40(%rip), %xmm6
	addps	LC6(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC7(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC8(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC9(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC10(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC11(%rip), %xmm2
	mulps	%xmm1, %xmm2
	movaps	LC21(%rip), %xmm1
	cmpltps	%xmm4, %xmm1
	addps	%xmm5, %xmm2
	movaps	LC21(%rip), %xmm5
	pand	%xmm9, %xmm1
	pshufb	LC41(%rip), %xmm1
	cmpltps	%xmm0, %xmm5
	xorps	LC15(%rip), %xmm2
	pand	%xmm9, %xmm5
	pshufb	LC40(%rip), %xmm5
	por	%xmm5, %xmm1
	pshufb	LC42(%rip), %xmm1
	movaps	LC21(%rip), %xmm5
	cmpltps	%xmm2, %xmm5
	pand	%xmm9, %xmm5
	pshufb	LC41(%rip), %xmm5
	por	%xmm6, %xmm5
	pshufb	LC43(%rip), %xmm5
	por	%xmm5, %xmm1
	movdqa	%xmm1, -16(%rcx)
	subps	LC16(%rip), %xmm0
	movaps	LC22(%rip), %xmm1
	subps	LC16(%rip), %xmm4
	mulps	%xmm0, %xmm1
	subps	LC16(%rip), %xmm3
	mulps	LC20(%rip), %xmm15
	subps	LC16(%rip), %xmm2
	addps	LC23(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC24(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC25(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC26(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC27(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC28(%rip), %xmm1
	mulps	%xmm0, %xmm1
	addps	LC29(%rip), %xmm1
	mulps	%xmm0, %xmm1
	movaps	LC22(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC30(%rip), %xmm1
	addps	LC23(%rip), %xmm0
	mulps	%xmm1, %xmm15
	mulps	%xmm4, %xmm0
	movaps	%xmm15, -64(%rdx)
	movaps	-48(%rax), %xmm1
	addps	LC24(%rip), %xmm0
	mulps	LC20(%rip), %xmm1
	mulps	%xmm4, %xmm0
	addps	LC25(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC26(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC27(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC28(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC29(%rip), %xmm0
	mulps	%xmm4, %xmm0
	addps	LC30(%rip), %xmm0
	mulps	%xmm0, %xmm1
	movaps	LC22(%rip), %xmm0
	mulps	%xmm3, %xmm0
	movaps	%xmm1, -48(%rdx)
	movaps	-32(%rax), %xmm1
	addps	LC23(%rip), %xmm0
	mulps	LC20(%rip), %xmm1
	mulps	%xmm3, %xmm0
	addps	LC24(%rip), %xmm0
	mulps	%xmm3, %xmm0
	addps	LC25(%rip), %xmm0
	mulps	%xmm3, %xmm0
	addps	LC26(%rip), %xmm0
	mulps	%xmm3, %xmm0
	addps	LC27(%rip), %xmm0
	mulps	%xmm3, %xmm0
	addps	LC28(%rip), %xmm0
	mulps	%xmm3, %xmm0
	addps	LC29(%rip), %xmm0
	mulps	%xmm3, %xmm0
	addps	LC30(%rip), %xmm0
	mulps	%xmm0, %xmm1
	movaps	%xmm1, -32(%rdx)
	movaps	LC22(%rip), %xmm0
	movaps	-16(%rax), %xmm1
	mulps	%xmm2, %xmm0
	mulps	LC20(%rip), %xmm1
	addps	LC23(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC24(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC25(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC26(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC27(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC28(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC29(%rip), %xmm0
	mulps	%xmm2, %xmm0
	addps	LC30(%rip), %xmm0
	mulps	%xmm0, %xmm1
	movaps	%xmm1, -16(%rdx)
	cmpq	%rdi, %rax
	jne	L7
	movss	LC44(%rip), %xmm3
	xorl	%eax, %eax
	movss	LC54(%rip), %xmm4
	movss	LC45(%rip), %xmm12
	movss	LC46(%rip), %xmm11
	movss	LC47(%rip), %xmm10
	movss	LC48(%rip), %xmm9
	movss	LC49(%rip), %xmm8
	movss	LC50(%rip), %xmm7
	movss	LC51(%rip), %xmm6
	movss	LC52(%rip), %xmm5
	.align 4,0x90
L10:
	cmpb	$0, (%rsi,%rax)
	je	L8
	movss	(%r9,%rax,4), %xmm2
	movaps	%xmm3, %xmm0
	movaps	%xmm2, %xmm1
	subss	%xmm2, %xmm0
	addss	%xmm3, %xmm1
	mulss	LC64(%rip), %xmm2
	mulss	%xmm1, %xmm0
	movd	%xmm0, %ecx
	movd	%xmm0, %edx
	movd	%xmm0, %edi
	sarl	$22, %ecx
	andl	$8388607, %edx
	sarl	$23, %edi
	andl	$1, %ecx
	orl	$1065353216, %edx
	movzbl	%dil, %edi
	movl	%ecx, %r8d
	sall	$23, %r8d
	subl	%r8d, %edx
	movd	%edx, %xmm1
	subss	%xmm3, %xmm1
	leal	-127(%rcx,%rdi), %edx
	movaps	%xmm1, %xmm0
	mulss	%xmm12, %xmm0
	addss	%xmm11, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm10, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm9, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm8, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm6, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm5, %xmm0
	mulss	%xmm1, %xmm0
	cvtsi2ss	%edx, %xmm1
	mulss	LC53(%rip), %xmm1
	addss	%xmm1, %xmm0
	movss	LC56(%rip), %xmm1
	xorps	%xmm4, %xmm0
	sqrtss	%xmm0, %xmm0
	subss	LC55(%rip), %xmm0
	mulss	%xmm0, %xmm1
	addss	LC57(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC58(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC59(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC60(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC61(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC62(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC63(%rip), %xmm1
	mulss	%xmm0, %xmm1
	addss	LC65(%rip), %xmm1
	mulss	%xmm1, %xmm2
	movss	%xmm2, (%r10,%rax,4)
L8:
	addq	$1, %rax
	cmpq	$8192, %rax
	jne	L10
	addq	$8080, %rsp
LCFI1:
	ret
LFE3421:
	.align 4,0x90
	.globl __Z8computeVv
__Z8computeVv:
LFB3424:
	leaq	_foo(%rip), %rax
	leaq	_vb(%rip), %rcx
	leaq	_va(%rip), %rdx
	leaq	2048+_foo(%rip), %r8
	jmp	L17
	.align 4,0x90
L13:
	movaps	(%rdx), %xmm1
	movaps	%xmm1, %xmm5
	movaps	%xmm1, %xmm0
	movaps	%xmm1, %xmm6
	shufps	$85, %xmm1, %xmm5
	sqrtss	%xmm0, %xmm0
	movaps	%xmm5, %xmm2
	sqrtss	%xmm2, %xmm2
	unpckhps	%xmm1, %xmm6
	shufps	$255, %xmm1, %xmm1
	sqrtss	%xmm1, %xmm1
	insertps	$0xe, %xmm0, %xmm0
	insertps	$16, %xmm2, %xmm0
	movaps	%xmm6, %xmm2
	sqrtss	%xmm2, %xmm2
	insertps	$32, %xmm2, %xmm0
	insertps	$48, %xmm1, %xmm0
	movaps	%xmm0, (%rcx)
L15:
	addq	$1, %rax
	addq	$16, %rcx
	addq	$16, %rdx
	cmpq	%r8, %rax
	je	L18
L17:
	cmpb	$0, (%rax)
	je	L13
	movaps	(%rdx), %xmm11
	movaps	LC0(%rip), %xmm2
	movaps	LC0(%rip), %xmm0
	subps	%xmm11, %xmm2
	movss	LC44(%rip), %xmm14
	addps	%xmm11, %xmm0
	movss	LC45(%rip), %xmm13
	movss	LC46(%rip), %xmm10
	movss	LC47(%rip), %xmm9
	mulps	%xmm0, %xmm2
	movss	LC49(%rip), %xmm7
	movss	LC48(%rip), %xmm8
	movss	LC50(%rip), %xmm6
	movss	LC51(%rip), %xmm5
	movaps	%xmm2, -56(%rsp)
	movss	LC52(%rip), %xmm4
	movl	-56(%rsp), %esi
	movss	LC53(%rip), %xmm3
	movl	%esi, %edi
	movl	%esi, %r9d
	andl	$8388607, %esi
	sarl	$22, %edi
	orl	$1065353216, %esi
	sarl	$23, %r9d
	andl	$1, %edi
	movzbl	%r9b, %r9d
	movl	%edi, %r10d
	sall	$23, %r10d
	subl	%r10d, %esi
	movd	%esi, %xmm1
	subss	%xmm14, %xmm1
	leal	-127(%rdi,%r9), %esi
	movaps	%xmm1, %xmm0
	mulss	%xmm13, %xmm0
	addss	%xmm10, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm9, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm8, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm6, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm5, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm4, %xmm0
	mulss	%xmm1, %xmm0
	cvtsi2ss	%esi, %xmm1
	extractps	$1, %xmm2, %esi
	movl	%esi, %edi
	movl	%esi, %r9d
	andl	$8388607, %esi
	sarl	$22, %edi
	orl	$1065353216, %esi
	sarl	$23, %r9d
	andl	$1, %edi
	movzbl	%r9b, %r9d
	mulss	%xmm3, %xmm1
	movl	%edi, %r10d
	sall	$23, %r10d
	subl	%r10d, %esi
	addss	%xmm1, %xmm0
	movaps	%xmm15, %xmm1
	movd	%esi, %xmm15
	subss	%xmm14, %xmm15
	leal	-127(%rdi,%r9), %esi
	movss	%xmm0, %xmm1
	movaps	%xmm15, %xmm0
	mulss	%xmm13, %xmm0
	addss	%xmm10, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm9, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm8, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm6, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm5, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm4, %xmm0
	mulss	%xmm15, %xmm0
	cvtsi2ss	%esi, %xmm15
	extractps	$2, %xmm2, %esi
	movl	%esi, %edi
	movl	%esi, %r9d
	andl	$8388607, %esi
	sarl	$22, %edi
	orl	$1065353216, %esi
	sarl	$23, %r9d
	andl	$1, %edi
	movzbl	%r9b, %r9d
	mulss	%xmm3, %xmm15
	movl	%edi, %r10d
	sall	$23, %r10d
	subl	%r10d, %esi
	addss	%xmm15, %xmm0
	movd	%esi, %xmm15
	subss	%xmm14, %xmm15
	leal	-127(%rdi,%r9), %esi
	insertps	$16, %xmm0, %xmm1
	movaps	%xmm15, %xmm0
	mulss	%xmm13, %xmm0
	addss	%xmm10, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm9, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm8, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm6, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm5, %xmm0
	mulss	%xmm15, %xmm0
	addss	%xmm4, %xmm0
	mulss	%xmm15, %xmm0
	cvtsi2ss	%esi, %xmm15
	extractps	$3, %xmm2, %esi
	movl	%esi, %edi
	movl	%esi, %r9d
	andl	$8388607, %esi
	sarl	$22, %edi
	orl	$1065353216, %esi
	sarl	$23, %r9d
	andl	$1, %edi
	movzbl	%r9b, %r9d
	mulss	%xmm3, %xmm15
	movl	%edi, %r10d
	sall	$23, %r10d
	subl	%r10d, %esi
	addss	%xmm15, %xmm0
	xorps	%xmm15, %xmm15
	insertps	$32, %xmm0, %xmm1
	movd	%esi, %xmm0
	subss	%xmm14, %xmm0
	leal	-127(%rdi,%r9), %esi
	movaps	%xmm0, %xmm14
	movaps	%xmm13, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm10, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm9, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm8, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm6, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm5, %xmm0
	mulss	%xmm14, %xmm0
	addss	%xmm4, %xmm0
	cvtsi2ss	%esi, %xmm4
	mulss	%xmm14, %xmm0
	mulss	%xmm3, %xmm4
	addss	%xmm4, %xmm0
	insertps	$48, %xmm0, %xmm1
	movaps	LC21(%rip), %xmm0
	xorps	LC15(%rip), %xmm1
	cmpltps	%xmm1, %xmm0
	movmskps	%xmm0, %esi
	testl	%esi, %esi
	jne	L14
	movss	LC66(%rip), %xmm13
	movaps	%xmm1, %xmm3
	movss	LC67(%rip), %xmm0
	subss	%xmm13, %xmm3
	movss	LC68(%rip), %xmm10
	movss	LC69(%rip), %xmm9
	movss	LC70(%rip), %xmm8
	movaps	%xmm3, %xmm2
	movss	LC71(%rip), %xmm7
	mulss	%xmm0, %xmm2
	movss	LC72(%rip), %xmm6
	movss	LC73(%rip), %xmm5
	movss	LC74(%rip), %xmm4
	addss	%xmm10, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm9, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm8, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm7, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm6, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm5, %xmm2
	mulss	%xmm3, %xmm2
	addss	%xmm4, %xmm2
	mulss	%xmm3, %xmm2
	movss	LC75(%rip), %xmm3
	addss	%xmm3, %xmm2
	movss	%xmm2, %xmm12
	movaps	%xmm1, %xmm2
	shufps	$85, %xmm1, %xmm2
	movaps	%xmm2, %xmm14
	subss	%xmm13, %xmm14
	movaps	%xmm14, %xmm2
	mulss	%xmm0, %xmm2
	addss	%xmm10, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm9, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm8, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm7, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm6, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm5, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm4, %xmm2
	mulss	%xmm14, %xmm2
	addss	%xmm3, %xmm2
	insertps	$16, %xmm2, %xmm12
	movaps	%xmm1, %xmm2
	unpckhps	%xmm1, %xmm2
	movaps	%xmm2, %xmm14
	shufps	$255, %xmm1, %xmm1
	subss	%xmm13, %xmm14
	subss	%xmm13, %xmm1
	movaps	%xmm14, %xmm2
	mulss	%xmm0, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm10, %xmm2
	addss	%xmm10, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm9, %xmm2
	addss	%xmm9, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm8, %xmm2
	addss	%xmm8, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm7, %xmm2
	addss	%xmm7, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm6, %xmm2
	addss	%xmm6, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm5, %xmm2
	addss	%xmm5, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm4, %xmm2
	addss	%xmm4, %xmm0
	mulss	%xmm14, %xmm2
	mulss	%xmm1, %xmm0
	addss	%xmm3, %xmm2
	addss	%xmm3, %xmm0
	insertps	$32, %xmm2, %xmm12
	insertps	$48, %xmm0, %xmm12
	movaps	LC20(%rip), %xmm0
	mulps	%xmm12, %xmm0
	mulps	%xmm11, %xmm0
	movaps	%xmm0, (%rcx)
	jmp	L15
	.align 4,0x90
L14:
	movdqa	%xmm2, %xmm3
	movdqa	LC2(%rip), %xmm1
	psrad	$22, %xmm3
	pand	LC1(%rip), %xmm3
	movdqa	%xmm3, %xmm0
	pand	%xmm2, %xmm1
	pslld	$23, %xmm0
	por	LC3(%rip), %xmm1
	psubd	%xmm0, %xmm1
	movaps	LC4(%rip), %xmm0
	psrad	$23, %xmm2
	subps	LC0(%rip), %xmm1
	pand	LC12(%rip), %xmm2
	paddd	LC13(%rip), %xmm2
	paddd	%xmm3, %xmm2
	mulps	%xmm1, %xmm0
	cvtdq2ps	%xmm2, %xmm2
	mulps	LC14(%rip), %xmm2
	addps	LC5(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC6(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC7(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC8(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC9(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC10(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	LC11(%rip), %xmm0
	mulps	%xmm1, %xmm0
	addps	%xmm2, %xmm0
	movaps	%xmm15, %xmm2
	xorps	LC15(%rip), %xmm0
	rsqrtps	%xmm0, %xmm3
	movaps	%xmm0, %xmm1
	cmpneqps	%xmm0, %xmm2
	subps	LC16(%rip), %xmm1
	andps	%xmm2, %xmm3
	movaps	%xmm3, %xmm2
	mulps	%xmm0, %xmm2
	cmpltps	LC21(%rip), %xmm0
	mulps	%xmm2, %xmm3
	mulps	LC18(%rip), %xmm2
	addps	LC17(%rip), %xmm3
	mulps	%xmm2, %xmm3
	movaps	LC22(%rip), %xmm2
	mulps	%xmm1, %xmm2
	subps	LC19(%rip), %xmm3
	addps	LC23(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC24(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC25(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC26(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC27(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC28(%rip), %xmm2
	mulps	%xmm1, %xmm2
	addps	LC29(%rip), %xmm2
	mulps	%xmm1, %xmm2
	movaps	LC31(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC30(%rip), %xmm2
	addps	LC32(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC33(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC34(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC35(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC36(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC37(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC38(%rip), %xmm1
	mulps	%xmm3, %xmm1
	addps	LC39(%rip), %xmm1
	blendvps	%xmm0, %xmm2, %xmm1
	mulps	%xmm11, %xmm1
	mulps	LC20(%rip), %xmm1
	movaps	%xmm1, (%rcx)
	jmp	L15
L18:
	rep; ret
LFE3424:
	.align 4,0x90
	.globl __Z5countv
__Z5countv:
LFB3425:
	movss	LC44(%rip), %xmm2
	leaq	_a(%rip), %rsi
	xorl	%eax, %eax
	movss	LC45(%rip), %xmm11
	leaq	32768+_a(%rip), %r9
	movss	LC46(%rip), %xmm10
	movss	LC47(%rip), %xmm9
	movss	LC48(%rip), %xmm8
	movss	LC49(%rip), %xmm7
	movss	LC50(%rip), %xmm6
	movss	LC51(%rip), %xmm5
	movss	LC52(%rip), %xmm4
	movss	LC53(%rip), %xmm3
	.align 4,0x90
L21:
	movss	(%rsi), %xmm1
	movaps	%xmm2, %xmm0
	subss	%xmm1, %xmm0
	addss	%xmm2, %xmm1
	mulss	%xmm1, %xmm0
	movd	%xmm0, %ecx
	movd	%xmm0, %edx
	movd	%xmm0, %edi
	sarl	$22, %ecx
	andl	$8388607, %edx
	sarl	$23, %edi
	andl	$1, %ecx
	orl	$1065353216, %edx
	movzbl	%dil, %edi
	movl	%ecx, %r8d
	leal	-127(%rcx,%rdi), %ecx
	sall	$23, %r8d
	subl	%r8d, %edx
	movd	%edx, %xmm1
	subss	%xmm2, %xmm1
	leal	1(%rax), %edx
	movaps	%xmm1, %xmm0
	mulss	%xmm11, %xmm0
	addss	%xmm10, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm9, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm8, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm6, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm5, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm4, %xmm0
	mulss	%xmm1, %xmm0
	cvtsi2ss	%ecx, %xmm1
	mulss	%xmm3, %xmm1
	addss	%xmm1, %xmm0
	comiss	LC76(%rip), %xmm0
	cmova	%edx, %eax
	addq	$4, %rsi
	cmpq	%r9, %rsi
	jne	L21
	rep; ret
LFE3425:
	.align 4,0x90
	.globl __Z4fillv
__Z4fillv:
LFB3855:
	movss	_rgen(%rip), %xmm8
	leaq	_a(%rip), %rsi
	pxor	%xmm2, %xmm2
	movss	4+_rgen(%rip), %xmm7
	leaq	32768+_a(%rip), %r9
	movq	2496+_eng(%rip), %rax
	leaq	_eng(%rip), %r10
	subss	%xmm8, %xmm7
	leaq	896+_eng(%rip), %r8
	movsd	LC81(%rip), %xmm9
	movdqa	LC78(%rip), %xmm6
	leaq	2492+_eng(%rip), %rdx
	movdqa	LC79(%rip), %xmm5
	mulss	LC77(%rip), %xmm7
	movdqa	LC1(%rip), %xmm4
	movdqa	LC80(%rip), %xmm3
	jmp	L33
	.align 4,0x90
L35:
	movl	(%r10,%rax,4), %ecx
	leaq	1(%rax), %rdi
	.align 4,0x90
L24:
	movl	%ecx, %eax
	addq	$4, %rsi
	shrl	$11, %eax
	xorl	%ecx, %eax
	movl	%eax, %ecx
	sall	$7, %ecx
	andl	$-1658038656, %ecx
	xorl	%eax, %ecx
	movl	%ecx, %eax
	sall	$15, %eax
	andl	$-272236544, %eax
	xorl	%ecx, %eax
	movl	%eax, %ecx
	shrl	$18, %ecx
	xorl	%ecx, %eax
	cvtsi2ssq	%rax, %xmm0
	mulss	%xmm7, %xmm0
	addss	%xmm8, %xmm0
	unpcklps	%xmm0, %xmm0
	cvtps2pd	%xmm0, %xmm0
	addsd	%xmm0, %xmm0
	subsd	%xmm9, %xmm0
	movddup	%xmm0, %xmm1
	cvtpd2ps	%xmm1, %xmm1
	movss	%xmm1, -4(%rsi)
	cmpq	%r9, %rsi
	je	L32
	movq	%rdi, %rax
L33:
	cmpq	$623, %rax
	jbe	L35
	leaq	_eng(%rip), %rax
	.align 4,0x90
L27:
	movdqu	4(%rax), %xmm1
	addq	$16, %rax
	movdqa	-16(%rax), %xmm0
	movdqu	1572(%rax), %xmm10
	pand	%xmm6, %xmm1
	pand	%xmm5, %xmm0
	por	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pand	%xmm4, %xmm0
	pcmpeqd	%xmm2, %xmm0
	psrld	$1, %xmm1
	pandn	%xmm3, %xmm0
	pxor	%xmm10, %xmm0
	pxor	%xmm1, %xmm0
	movdqa	%xmm0, -16(%rax)
	cmpq	%r8, %rax
	jne	L27
	movl	900+_eng(%rip), %edi
	movl	896+_eng(%rip), %ecx
	movl	%edi, %eax
	andl	$-2147483648, %edi
	andl	$2147483647, %eax
	andl	$-2147483648, %ecx
	orl	%eax, %ecx
	movl	%ecx, %eax
	shrl	%ecx
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2484+_eng(%rip), %eax
	xorl	%ecx, %eax
	movl	904+_eng(%rip), %ecx
	movl	%eax, 896+_eng(%rip)
	movl	%ecx, %eax
	andl	$-2147483648, %ecx
	andl	$2147483647, %eax
	orl	%eax, %edi
	movl	%edi, %eax
	shrl	%edi
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2488+_eng(%rip), %eax
	xorl	%edi, %eax
	movl	%eax, 900+_eng(%rip)
	movl	908+_eng(%rip), %eax
	andl	$2147483647, %eax
	orl	%eax, %ecx
	movl	%ecx, %eax
	shrl	%ecx
	andl	$1, %eax
	negl	%eax
	andl	$-1727483681, %eax
	xorl	2492+_eng(%rip), %eax
	xorl	%ecx, %eax
	movl	%eax, 904+_eng(%rip)
	leaq	908+_eng(%rip), %rax
	.align 4,0x90
L26:
	movdqa	4(%rax), %xmm1
	addq	$16, %rax
	movdqu	-16(%rax), %xmm0
	pand	%xmm6, %xmm1
	pand	%xmm5, %xmm0
	por	%xmm0, %xmm1
	movdqa	%xmm1, %xmm0
	pand	%xmm4, %xmm0
	pcmpeqd	%xmm2, %xmm0
	psrld	$1, %xmm1
	pandn	%xmm3, %xmm0
	pxor	-924(%rax), %xmm0
	pxor	%xmm1, %xmm0
	movdqu	%xmm0, -16(%rax)
	cmpq	%rdx, %rax
	jne	L26
	movl	_eng(%rip), %ecx
	movl	2492+_eng(%rip), %eax
	movl	%ecx, %edi
	andl	$-2147483648, %eax
	andl	$2147483647, %edi
	orl	%eax, %edi
	movl	%edi, %eax
	andl	$1, %edi
	shrl	%eax
	negl	%edi
	xorl	1584+_eng(%rip), %eax
	andl	$-1727483681, %edi
	xorl	%edi, %eax
	movl	$1, %edi
	movl	%eax, 2492+_eng(%rip)
	jmp	L24
L32:
	movq	%rdi, 2496+_eng(%rip)
	ret
LFE3855:
	.align 4,0x90
	.globl __Z3sumv
__Z3sumv:
LFB3856:
	leaq	_b(%rip), %rax
	xorps	%xmm0, %xmm0
	leaq	32768+_b(%rip), %rdx
	.align 4,0x90
L38:
	addps	(%rax), %xmm0
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	L38
	haddps	%xmm0, %xmm0
	haddps	%xmm0, %xmm0
	ret
LFE3856:
	.cstring
LC83:
	.ascii " \0"
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB3858:
	pushq	%r15
LCFI2:
	pushq	%r14
LCFI3:
	pushq	%r13
LCFI4:
	pushq	%r12
LCFI5:
	pushq	%rbp
LCFI6:
	pushq	%rbx
LCFI7:
	subq	$40, %rsp
LCFI8:
	cmpl	$3, %edi
	jg	L42
	movl	$2048, %edx
	movl	$-1, %esi
	leaq	_foo(%rip), %rdi
	call	_memset
L42:
	call	__Z4fillv
	xorl	%r15d, %r15d
	xorl	%r14d, %r14d
	call	__Z7computev
	leaq	32768+_b(%rip), %rbx
	xorl	%r13d, %r13d
	call	__Z5countv
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	movl	%eax, %esi
	call	__ZNSolsEi
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	xorps	%xmm1, %xmm1
	movl	$10000, 28(%rsp)
	movss	%xmm1, 24(%rsp)
	movss	%xmm1, 12(%rsp)
	movss	%xmm1, 8(%rsp)
	.align 4
L41:
	call	__Z4fillv
	rdtsc
	movq	%rax, %rbp
	salq	$32, %rdx
	orq	%rdx, %rbp
	call	__Z7computev
	rdtsc
	xorps	%xmm0, %xmm0
	salq	$32, %rdx
	movq	%rdx, %r12
	orq	%rax, %r12
	subq	%rbp, %r12
	leaq	(%r12,%r13), %rax
	movq	%rax, 16(%rsp)
	movq	%rax, %r13
	leaq	_b(%rip), %rax
	.align 4
L44:
	addps	(%rax), %xmm0
	addq	$16, %rax
	cmpq	%rbx, %rax
	jne	L44
	haddps	%xmm0, %xmm0
	haddps	%xmm0, %xmm0
	addss	8(%rsp), %xmm0
	movss	%xmm0, 8(%rsp)
	rdtsc
	movq	%rax, %r12
	salq	$32, %rdx
	orq	%rdx, %r12
	call	__Z8computeBv
	rdtsc
	xorps	%xmm0, %xmm0
	salq	$32, %rdx
	movq	%rdx, %rbp
	orq	%rax, %rbp
	leaq	_b(%rip), %rax
	subq	%r12, %rbp
	leaq	0(%rbp,%r14), %r12
	movq	%r12, %r14
	.align 4
L46:
	addps	(%rax), %xmm0
	addq	$16, %rax
	cmpq	%rax, %rbx
	jne	L46
	haddps	%xmm0, %xmm0
	leaq	_a(%rip), %rsi
	movl	$32768, %edx
	leaq	_va(%rip), %rdi
	haddps	%xmm0, %xmm0
	addss	12(%rsp), %xmm0
	movss	%xmm0, 12(%rsp)
	call	_memcpy
	rdtsc
	movq	%rax, %rbp
	salq	$32, %rdx
	orq	%rdx, %rbp
	call	__Z8computeVv
	rdtsc
	leaq	_vb(%rip), %rsi
	leaq	_b(%rip), %rdi
	salq	$32, %rdx
	orq	%rax, %rdx
	subq	%rbp, %rdx
	leaq	(%rdx,%r15), %rax
	movl	$32768, %edx
	movq	%rax, %rbp
	movq	%rax, %r15
	call	_memcpy
	leaq	_b(%rip), %rax
	xorps	%xmm0, %xmm0
	.align 4
L48:
	addps	(%rax), %xmm0
	addq	$16, %rax
	cmpq	%rbx, %rax
	jne	L48
	subl	$1, 28(%rsp)
	haddps	%xmm0, %xmm0
	haddps	%xmm0, %xmm0
	addss	24(%rsp), %xmm0
	movss	%xmm0, 24(%rsp)
	jne	L41
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	movss	8(%rsp), %xmm0
	cvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC83(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	cvtsi2sdq	16(%rsp), %xmm0
	movq	%rax, %rdi
	mulsd	LC84(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	movss	12(%rsp), %xmm0
	cvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC83(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	cvtsi2sdq	%r12, %xmm0
	movq	%rax, %rdi
	mulsd	LC84(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	movss	24(%rsp), %xmm0
	cvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC83(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	cvtsi2sdq	%rbp, %xmm0
	movq	%rax, %rdi
	mulsd	LC84(%rip), %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	addq	$40, %rsp
LCFI9:
	xorl	%eax, %eax
	popq	%rbx
LCFI10:
	popq	%rbp
LCFI11:
	popq	%r12
LCFI12:
	popq	%r13
LCFI13:
	popq	%r14
LCFI14:
	popq	%r15
LCFI15:
	ret
LFE3858:
	.align 4
__GLOBAL__sub_I_enfinv1.cpp:
LFB4151:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI16:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	call	___cxa_atexit
	movl	$5489, %edx
	movl	$1, %ecx
	movl	$5489, _eng(%rip)
	leaq	_eng(%rip), %r8
	movl	$440509467, %edi
	.align 4
L53:
	movl	%edx, %eax
	shrl	$30, %eax
	xorl	%edx, %eax
	movl	%ecx, %edx
	shrl	$4, %edx
	imull	$1812433253, %eax, %esi
	movl	%edx, %eax
	mull	%edi
	movl	%ecx, %eax
	shrl	$2, %edx
	imull	$624, %edx, %edx
	subl	%edx, %eax
	movl	%eax, %edx
	addl	%esi, %edx
	movl	%edx, (%r8,%rcx,4)
	addq	$1, %rcx
	cmpq	$624, %rcx
	jne	L53
	movl	$5489, %edx
	movl	$1, %ecx
	movl	$440509467, %edi
	movq	$624, 2496+_eng(%rip)
	leaq	_eng2(%rip), %r8
	movl	$5489, _eng2(%rip)
	.align 4
L55:
	movl	%edx, %eax
	shrl	$30, %eax
	xorl	%edx, %eax
	movl	%ecx, %edx
	shrl	$4, %edx
	imull	$1812433253, %eax, %esi
	movl	%edx, %eax
	mull	%edi
	movl	%ecx, %eax
	shrl	$2, %edx
	imull	$624, %edx, %edx
	subl	%edx, %eax
	movl	%eax, %edx
	addl	%esi, %edx
	movl	%edx, (%r8,%rcx,4)
	addq	$1, %rcx
	cmpq	$624, %rcx
	jne	L55
	movq	$624, 2496+_eng2(%rip)
	movl	$0x00000000, _rgen(%rip)
	movl	$0x3f800000, 4+_rgen(%rip)
	addq	$8, %rsp
LCFI17:
	ret
LFE4151:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_enfinv1.cpp
	.globl _rgen
	.zerofill __DATA,__pu_bss4,_rgen,8,4
	.globl _eng2
	.zerofill __DATA,__pu_bss5,_eng2,2504,5
	.globl _eng
	.zerofill __DATA,__pu_bss5,_eng,2504,5
	.static_data
__ZStL8__ioinit:
	.space	1
	.globl _foo
	.zerofill __DATA,__pu_bss5,_foo,2048,5
	.globl _vb
	.zerofill __DATA,__pu_bss5,_vb,32768,5
	.globl _va
	.zerofill __DATA,__pu_bss5,_va,32768,5
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,32768,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,32768,5
	.literal16
	.align 4
LC0:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC1:
	.long	1
	.long	1
	.long	1
	.long	1
	.align 4
LC2:
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.align 4
LC3:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC4:
	.long	3179186122
	.long	3179186122
	.long	3179186122
	.long	3179186122
	.align 4
LC5:
	.long	1041113284
	.long	1041113284
	.long	1041113284
	.long	1041113284
	.align 4
LC6:
	.long	3191002492
	.long	3191002492
	.long	3191002492
	.long	3191002492
	.align 4
LC7:
	.long	1045333864
	.long	1045333864
	.long	1045333864
	.long	1045333864
	.align 4
LC8:
	.long	3196041016
	.long	3196041016
	.long	3196041016
	.long	3196041016
	.align 4
LC9:
	.long	1051369742
	.long	1051369742
	.long	1051369742
	.long	1051369742
	.align 4
LC10:
	.long	3204448304
	.long	3204448304
	.long	3204448304
	.long	3204448304
	.align 4
LC11:
	.long	1065353222
	.long	1065353222
	.long	1065353222
	.long	1065353222
	.align 4
LC12:
	.long	255
	.long	255
	.long	255
	.long	255
	.align 4
LC13:
	.long	-127
	.long	-127
	.long	-127
	.long	-127
	.align 4
LC14:
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.align 4
LC15:
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.align 4
LC16:
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.long	1075838976
	.align 4
LC17:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 4
LC18:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.align 4
LC19:
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.align 4
LC20:
	.long	1068827891
	.long	1068827891
	.long	1068827891
	.long	1068827891
	.align 4
LC21:
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.long	1084227584
	.align 4
LC22:
	.long	854680968
	.long	854680968
	.long	854680968
	.long	854680968
	.align 4
LC23:
	.long	884493110
	.long	884493110
	.long	884493110
	.long	884493110
	.align 4
LC24:
	.long	3060560727
	.long	3060560727
	.long	3060560727
	.long	3060560727
	.align 4
LC25:
	.long	3063110337
	.long	3063110337
	.long	3063110337
	.long	3063110337
	.align 4
LC26:
	.long	962933467
	.long	962933467
	.long	962933467
	.long	962933467
	.align 4
LC27:
	.long	3131331592
	.long	3131331592
	.long	3131331592
	.long	3131331592
	.align 4
LC28:
	.long	3146310895
	.long	3146310895
	.long	3146310895
	.long	3146310895
	.align 4
LC29:
	.long	1048350563
	.long	1048350563
	.long	1048350563
	.long	1048350563
	.align 4
LC30:
	.long	1069559343
	.long	1069559343
	.long	1069559343
	.long	1069559343
	.align 4
LC31:
	.long	3109154971
	.long	3109154971
	.long	3109154971
	.long	3109154971
	.align 4
LC32:
	.long	953398635
	.long	953398635
	.long	953398635
	.long	953398635
	.align 4
LC33:
	.long	984669298
	.long	984669298
	.long	984669298
	.long	984669298
	.align 4
LC34:
	.long	3144728039
	.long	3144728039
	.long	3144728039
	.long	3144728039
	.align 4
LC35:
	.long	1002181243
	.long	1002181243
	.long	1002181243
	.long	1002181243
	.align 4
LC36:
	.long	3153708503
	.long	3153708503
	.long	3153708503
	.long	3153708503
	.align 4
LC37:
	.long	1008379262
	.long	1008379262
	.long	1008379262
	.long	1008379262
	.align 4
LC38:
	.long	1065367259
	.long	1065367259
	.long	1065367259
	.long	1065367259
	.align 4
LC39:
	.long	1077235582
	.long	1077235582
	.long	1077235582
	.long	1077235582
	.align 4
LC40:
	.byte	0
	.byte	1
	.byte	4
	.byte	5
	.byte	8
	.byte	9
	.byte	12
	.byte	13
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 4
LC41:
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
	.byte	4
	.byte	5
	.byte	8
	.byte	9
	.byte	12
	.byte	13
	.align 4
LC42:
	.byte	0
	.byte	2
	.byte	4
	.byte	6
	.byte	8
	.byte	10
	.byte	12
	.byte	14
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 4
LC43:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	2
	.byte	4
	.byte	6
	.byte	8
	.byte	10
	.byte	12
	.byte	14
	.literal4
	.align 2
LC44:
	.long	1065353216
	.align 2
LC45:
	.long	3179186122
	.align 2
LC46:
	.long	1041113284
	.align 2
LC47:
	.long	3191002492
	.align 2
LC48:
	.long	1045333864
	.align 2
LC49:
	.long	3196041016
	.align 2
LC50:
	.long	1051369742
	.align 2
LC51:
	.long	3204448304
	.align 2
LC52:
	.long	1065353222
	.align 2
LC53:
	.long	1060205080
	.literal16
	.align 4
LC54:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.literal4
	.align 2
LC55:
	.long	1077936128
	.align 2
LC56:
	.long	3109154971
	.align 2
LC57:
	.long	953398635
	.align 2
LC58:
	.long	984669298
	.align 2
LC59:
	.long	3144728039
	.align 2
LC60:
	.long	1002181243
	.align 2
LC61:
	.long	3153708503
	.align 2
LC62:
	.long	1008379262
	.align 2
LC63:
	.long	1065367259
	.align 2
LC64:
	.long	1068827891
	.align 2
LC65:
	.long	1077235582
	.align 2
LC66:
	.long	1075838976
	.align 2
LC67:
	.long	854680968
	.align 2
LC68:
	.long	884493110
	.align 2
LC69:
	.long	3060560727
	.align 2
LC70:
	.long	3063110337
	.align 2
LC71:
	.long	962933467
	.align 2
LC72:
	.long	3131331592
	.align 2
LC73:
	.long	3146310895
	.align 2
LC74:
	.long	1048350563
	.align 2
LC75:
	.long	1069559343
	.align 2
LC76:
	.long	3231711232
	.align 2
LC77:
	.long	796917760
	.literal16
	.align 4
LC78:
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.align 4
LC79:
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.align 4
LC80:
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.literal8
	.align 3
LC81:
	.long	0
	.long	1072693248
	.align 3
LC84:
	.long	3944497965
	.long	1058682594
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
	.quad	LFB3420-.
	.set L$set$2,LFE3420-LFB3420
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB3421-.
	.set L$set$4,LFE3421-LFB3421
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB3421
	.long L$set$5
	.byte	0xe
	.byte	0x98,0x3f
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$7,LEFDE5-LASFDE5
	.long L$set$7
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3424-.
	.set L$set$8,LFE3424-LFB3424
	.quad L$set$8
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$9,LEFDE7-LASFDE7
	.long L$set$9
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3425-.
	.set L$set$10,LFE3425-LFB3425
	.quad L$set$10
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$11,LEFDE9-LASFDE9
	.long L$set$11
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB3855-.
	.set L$set$12,LFE3855-LFB3855
	.quad L$set$12
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$13,LEFDE11-LASFDE11
	.long L$set$13
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB3856-.
	.set L$set$14,LFE3856-LFB3856
	.quad L$set$14
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$15,LEFDE13-LASFDE13
	.long L$set$15
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB3858-.
	.set L$set$16,LFE3858-LFB3858
	.quad L$set$16
	.byte	0
	.byte	0x4
	.set L$set$17,LCFI2-LFB3858
	.long L$set$17
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$18,LCFI3-LCFI2
	.long L$set$18
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$19,LCFI4-LCFI3
	.long L$set$19
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$20,LCFI5-LCFI4
	.long L$set$20
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$21,LCFI6-LCFI5
	.long L$set$21
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$22,LCFI7-LCFI6
	.long L$set$22
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$23,LCFI8-LCFI7
	.long L$set$23
	.byte	0xe
	.byte	0x60
	.byte	0x4
	.set L$set$24,LCFI9-LCFI8
	.long L$set$24
	.byte	0xe
	.byte	0x38
	.byte	0x4
	.set L$set$25,LCFI10-LCFI9
	.long L$set$25
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$26,LCFI11-LCFI10
	.long L$set$26
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$27,LCFI12-LCFI11
	.long L$set$27
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$28,LCFI13-LCFI12
	.long L$set$28
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$29,LCFI14-LCFI13
	.long L$set$29
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$30,LCFI15-LCFI14
	.long L$set$30
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$31,LEFDE15-LASFDE15
	.long L$set$31
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB4151-.
	.set L$set$32,LFE4151-LFB4151
	.quad L$set$32
	.byte	0
	.byte	0x4
	.set L$set$33,LCFI16-LFB4151
	.long L$set$33
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$34,LCFI17-LCFI16
	.long L$set$34
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE15:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
