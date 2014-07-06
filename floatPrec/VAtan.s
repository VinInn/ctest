	.text
	.align 4,0x90
	.globl __Z10computeOnev
__Z10computeOnev:
LFB4239:
	pushq	%rbp
LCFI0:
	vxorps	%xmm2, %xmm2, %xmm2
	movq	%rsp, %rbp
LCFI1:
	andq	$-32, %rsp
	addq	$16, %rsp
	vmovaps	_va(%rip), %xmm1
	vrsqrtps	%xmm1, %xmm0
	vcmpneqps	%xmm1, %xmm2, %xmm2
	vandps	%xmm2, %xmm0, %xmm0
	vmulps	%xmm1, %xmm0, %xmm1
	vmulps	%xmm0, %xmm1, %xmm0
	vmulps	LC1(%rip), %xmm1, %xmm1
	vaddps	LC0(%rip), %xmm0, %xmm0
	vmulps	%xmm1, %xmm0, %xmm1
	vmovaps	%xmm1, _vb(%rip)
	leave
LCFI2:
	ret
LFE4239:
	.align 4,0x90
	.globl __Z8computeSv
__Z8computeSv:
LFB4240:
	leaq	_va(%rip), %rcx
	xorl	%eax, %eax
	leaq	_vb(%rip), %rdx
	.align 4,0x90
L4:
	vmovaps	(%rcx,%rax), %xmm1
	vmovaps	%xmm1, %xmm2
	vsqrtss	%xmm2, %xmm2, %xmm2
	vmovss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vsqrtss	%xmm2, %xmm2, %xmm2
	vinsertps	$16, %xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vsqrtss	%xmm2, %xmm2, %xmm2
	vsqrtss	%xmm1, %xmm1, %xmm1
	vinsertps	$32, %xmm2, %xmm0, %xmm0
	vinsertps	$48, %xmm1, %xmm0, %xmm1
	vmovaps	%xmm1, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$16384, %rax
	vxorps	%xmm0, %xmm0, %xmm0
	jne	L4
	rep; ret
LFE4240:
	.align 4,0x90
	.globl __Z8computeVv
__Z8computeVv:
LFB4241:
	xorl	%eax, %eax
	vmovaps	LC2(%rip), %xmm5
	leaq	_va(%rip), %rsi
	vmovaps	LC3(%rip), %xmm3
	vmovaps	LC4(%rip), %xmm10
	leaq	_vb(%rip), %rcx
	vmovaps	LC5(%rip), %xmm9
	vmovaps	LC6(%rip), %xmm8
	vmovaps	LC7(%rip), %xmm7
	vmovaps	LC8(%rip), %xmm6
	jmp	L8
	.align 4,0x90
L16:
	vmovaps	%xmm2, %xmm0
L12:
	vmulps	%xmm0, %xmm0, %xmm1
	vmulps	%xmm9, %xmm1, %xmm2
	vsubps	%xmm8, %xmm2, %xmm2
	vmulps	%xmm1, %xmm2, %xmm2
	vaddps	%xmm7, %xmm2, %xmm2
	vmulps	%xmm1, %xmm2, %xmm2
	vsubps	%xmm6, %xmm2, %xmm2
	vmulps	%xmm1, %xmm2, %xmm1
	vaddps	%xmm3, %xmm1, %xmm1
	vmulps	%xmm0, %xmm1, %xmm0
	vaddps	%xmm4, %xmm0, %xmm0
	vmovaps	%xmm0, (%rcx,%rax)
	addq	$16, %rax
	cmpq	$16384, %rax
	je	L15
L8:
	vmovaps	(%rsi,%rax), %xmm0
	vaddps	%xmm3, %xmm0, %xmm2
	vsubps	%xmm3, %xmm0, %xmm11
	vcmpltps	%xmm0, %xmm5, %xmm4
	vrcpps	%xmm2, %xmm1
	vmovmskps	%xmm4, %edx
	vandps	%xmm10, %xmm4, %xmm4
	testl	%edx, %edx
	vmulps	%xmm2, %xmm1, %xmm2
	vmulps	%xmm2, %xmm1, %xmm2
	vaddps	%xmm1, %xmm1, %xmm1
	vsubps	%xmm2, %xmm1, %xmm2
	vcmpleps	%xmm5, %xmm0, %xmm1
	vmulps	%xmm2, %xmm11, %xmm2
	vblendvps	%xmm1, %xmm0, %xmm2, %xmm2
	jne	L16
	vxorps	%xmm4, %xmm4, %xmm4
	jmp	L12
	.align 4,0x90
L15:
	rep; ret
LFE4241:
	.align 4,0x90
	.globl __Z8computeLv
__Z8computeLv:
LFB4242:
	vmovaps	LC9(%rip), %ymm10
	xorl	%eax, %eax
	vmovaps	LC10(%rip), %ymm3
	leaq	_a(%rip), %rcx
	vmovaps	LC11(%rip), %ymm9
	leaq	_b(%rip), %rdx
	vmovaps	LC12(%rip), %ymm8
	vmovaps	LC13(%rip), %ymm7
	vmovaps	LC14(%rip), %ymm6
	vmovaps	LC15(%rip), %ymm5
	.align 4,0x90
L18:
	vmovaps	(%rcx,%rax), %ymm0
	vaddps	%ymm3, %ymm0, %ymm2
	vsubps	%ymm3, %ymm0, %ymm4
	vrcpps	%ymm2, %ymm1
	vmulps	%ymm2, %ymm1, %ymm2
	vmulps	%ymm2, %ymm1, %ymm2
	vaddps	%ymm1, %ymm1, %ymm1
	vsubps	%ymm2, %ymm1, %ymm2
	vmulps	%ymm2, %ymm4, %ymm1
	vcmpltps	%ymm0, %ymm10, %ymm4
	vblendvps	%ymm4, %ymm1, %ymm0, %ymm1
	vandps	%ymm5, %ymm4, %ymm4
	vmulps	%ymm1, %ymm1, %ymm0
	vmulps	%ymm9, %ymm0, %ymm2
	vaddps	%ymm8, %ymm2, %ymm2
	vmulps	%ymm0, %ymm2, %ymm2
	vaddps	%ymm7, %ymm2, %ymm2
	vmulps	%ymm0, %ymm2, %ymm2
	vaddps	%ymm6, %ymm2, %ymm2
	vmulps	%ymm0, %ymm2, %ymm0
	vaddps	%ymm3, %ymm0, %ymm0
	vmulps	%ymm1, %ymm0, %ymm1
	vaddps	%ymm4, %ymm1, %ymm1
	vmovaps	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$16384, %rax
	jne	L18
	vzeroupper
	ret
LFE4242:
	.align 4,0x90
	.globl __Z8computeAv
__Z8computeAv:
LFB4243:
	vmovss	LC16(%rip), %xmm0
	xorl	%eax, %eax
	vmovss	LC17(%rip), %xmm9
	leaq	_va(%rip), %rcx
	vmovss	LC19(%rip), %xmm8
	leaq	_vb(%rip), %rdx
	vmovss	LC20(%rip), %xmm7
	vmovss	LC21(%rip), %xmm6
	vmovss	LC22(%rip), %xmm5
	vmovss	LC23(%rip), %xmm4
	.align 4,0x90
L21:
	vmovaps	(%rcx,%rax), %xmm2
	vmovaps	%xmm2, %xmm3
	vaddss	%xmm0, %xmm3, %xmm11
	vsubss	%xmm0, %xmm3, %xmm10
	vdivss	%xmm11, %xmm10, %xmm10
	vcmpltss	%xmm3, %xmm8, %xmm11
	vandnps	%xmm3, %xmm11, %xmm3
	vandps	%xmm11, %xmm9, %xmm12
	vandps	%xmm11, %xmm10, %xmm10
	vorps	%xmm10, %xmm3, %xmm10
	vmulss	%xmm10, %xmm10, %xmm3
	vmulss	%xmm7, %xmm3, %xmm11
	vaddss	%xmm6, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm5, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm4, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm3
	vaddss	%xmm0, %xmm3, %xmm3
	vmulss	%xmm10, %xmm3, %xmm10
	vshufps	$85, %xmm2, %xmm2, %xmm3
	vaddss	%xmm0, %xmm3, %xmm11
	vaddss	%xmm12, %xmm10, %xmm10
	vmovss	%xmm10, %xmm1, %xmm1
	vsubss	%xmm0, %xmm3, %xmm10
	vdivss	%xmm11, %xmm10, %xmm10
	vcmpltss	%xmm3, %xmm8, %xmm11
	vandnps	%xmm3, %xmm11, %xmm3
	vandps	%xmm11, %xmm9, %xmm12
	vandps	%xmm11, %xmm10, %xmm10
	vorps	%xmm10, %xmm3, %xmm10
	vmulss	%xmm10, %xmm10, %xmm3
	vmulss	%xmm7, %xmm3, %xmm11
	vaddss	%xmm6, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm5, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm4, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm3
	vaddss	%xmm0, %xmm3, %xmm3
	vmulss	%xmm10, %xmm3, %xmm10
	vunpckhps	%xmm2, %xmm2, %xmm3
	vaddss	%xmm0, %xmm3, %xmm11
	vshufps	$255, %xmm2, %xmm2, %xmm2
	vaddss	%xmm12, %xmm10, %xmm10
	vinsertps	$16, %xmm10, %xmm1, %xmm1
	vsubss	%xmm0, %xmm3, %xmm10
	vdivss	%xmm11, %xmm10, %xmm10
	vcmpltss	%xmm3, %xmm8, %xmm11
	vandnps	%xmm3, %xmm11, %xmm3
	vandps	%xmm11, %xmm9, %xmm12
	vandps	%xmm11, %xmm10, %xmm10
	vorps	%xmm10, %xmm3, %xmm10
	vmulss	%xmm10, %xmm10, %xmm3
	vmulss	%xmm7, %xmm3, %xmm11
	vaddss	%xmm6, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm5, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm4, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm3
	vaddss	%xmm0, %xmm3, %xmm3
	vmulss	%xmm10, %xmm3, %xmm10
	vsubss	%xmm0, %xmm2, %xmm3
	vaddss	%xmm12, %xmm10, %xmm10
	vinsertps	$32, %xmm10, %xmm1, %xmm1
	vaddss	%xmm0, %xmm2, %xmm10
	vdivss	%xmm10, %xmm3, %xmm3
	vcmpltss	%xmm2, %xmm8, %xmm10
	vandnps	%xmm2, %xmm10, %xmm2
	vandps	%xmm10, %xmm9, %xmm11
	vandps	%xmm10, %xmm3, %xmm3
	vorps	%xmm3, %xmm2, %xmm3
	vmulss	%xmm3, %xmm3, %xmm2
	vmulss	%xmm7, %xmm2, %xmm10
	vaddss	%xmm6, %xmm10, %xmm10
	vmulss	%xmm2, %xmm10, %xmm10
	vaddss	%xmm5, %xmm10, %xmm10
	vmulss	%xmm2, %xmm10, %xmm10
	vaddss	%xmm4, %xmm10, %xmm10
	vmulss	%xmm2, %xmm10, %xmm2
	vaddss	%xmm0, %xmm2, %xmm2
	vmulss	%xmm3, %xmm2, %xmm3
	vaddss	%xmm11, %xmm3, %xmm3
	vinsertps	$48, %xmm3, %xmm1, %xmm3
	vmovaps	%xmm3, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$16384, %rax
	vxorps	%xmm1, %xmm1, %xmm1
	jne	L21
	rep; ret
LFE4243:
	.align 4,0x90
	.globl __Z8computeBv
__Z8computeBv:
LFB4244:
	vmovaps	LC9(%rip), %ymm10
	xorl	%eax, %eax
	vmovaps	LC10(%rip), %ymm3
	leaq	_va(%rip), %rcx
	vmovaps	LC11(%rip), %ymm9
	leaq	_vb(%rip), %rdx
	vmovaps	LC12(%rip), %ymm8
	vmovaps	LC13(%rip), %ymm7
	vmovaps	LC14(%rip), %ymm6
	vmovaps	LC15(%rip), %ymm5
	.align 4,0x90
L24:
	vmovaps	(%rcx,%rax), %ymm0
	vaddps	%ymm3, %ymm0, %ymm2
	vsubps	%ymm3, %ymm0, %ymm4
	vrcpps	%ymm2, %ymm1
	vmulps	%ymm2, %ymm1, %ymm2
	vmulps	%ymm2, %ymm1, %ymm2
	vaddps	%ymm1, %ymm1, %ymm1
	vsubps	%ymm2, %ymm1, %ymm2
	vmulps	%ymm2, %ymm4, %ymm1
	vcmpltps	%ymm0, %ymm10, %ymm4
	vblendvps	%ymm4, %ymm1, %ymm0, %ymm1
	vandps	%ymm5, %ymm4, %ymm4
	vmulps	%ymm1, %ymm1, %ymm0
	vmulps	%ymm9, %ymm0, %ymm2
	vaddps	%ymm8, %ymm2, %ymm2
	vmulps	%ymm0, %ymm2, %ymm2
	vaddps	%ymm7, %ymm2, %ymm2
	vmulps	%ymm0, %ymm2, %ymm2
	vaddps	%ymm6, %ymm2, %ymm2
	vmulps	%ymm2, %ymm0, %ymm0
	vaddps	%ymm3, %ymm0, %ymm0
	vmulps	%ymm0, %ymm1, %ymm1
	vaddps	%ymm4, %ymm1, %ymm1
	vmovaps	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$16384, %rax
	jne	L24
	vzeroupper
	ret
LFE4244:
	.align 4,0x90
	.globl __Z8computeKv
__Z8computeKv:
LFB4245:
	vmovss	LC16(%rip), %xmm12
	xorl	%edx, %edx
	vmovss	LC24(%rip), %xmm11
	leaq	_va(%rip), %rdi
	vmovss	LC25(%rip), %xmm10
	leaq	_vb(%rip), %rsi
	vmovss	LC26(%rip), %xmm9
	vmovss	LC27(%rip), %xmm8
	vmovss	LC28(%rip), %xmm7
	vmovss	LC29(%rip), %xmm6
	vmovss	LC30(%rip), %xmm5
	vmovss	LC31(%rip), %xmm4
	vmovss	LC32(%rip), %xmm3
	.align 4,0x90
L27:
	vmovaps	(%rdi,%rdx), %xmm2
	vmovaps	%xmm2, %xmm0
	vmovd	%xmm0, %ecx
	vmovd	%xmm0, %eax
	vmovd	%xmm0, %r8d
	sarl	$22, %ecx
	andl	$8388607, %eax
	sarl	$23, %r8d
	andl	$1, %ecx
	orl	$1065353216, %eax
	movzbl	%r8b, %r8d
	movl	%ecx, %r9d
	sall	$23, %r9d
	subl	%r9d, %eax
	vmovd	%eax, %xmm0
	vsubss	%xmm12, %xmm0, %xmm0
	leal	-127(%rcx,%r8), %eax
	vmulss	%xmm11, %xmm0, %xmm13
	vaddss	%xmm10, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm9, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm8, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm7, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm6, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm5, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm4, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm0
	vcvtsi2ss	%eax, %xmm13, %xmm13
	vextractps	$1, %xmm2, %eax
	movl	%eax, %ecx
	movl	%eax, %r8d
	andl	$8388607, %eax
	sarl	$22, %ecx
	orl	$1065353216, %eax
	sarl	$23, %r8d
	andl	$1, %ecx
	movzbl	%r8b, %r8d
	vmulss	%xmm3, %xmm13, %xmm13
	movl	%ecx, %r9d
	sall	$23, %r9d
	subl	%r9d, %eax
	vaddss	%xmm13, %xmm0, %xmm0
	vmovss	%xmm0, %xmm1, %xmm1
	vmovd	%eax, %xmm0
	vsubss	%xmm12, %xmm0, %xmm0
	leal	-127(%rcx,%r8), %eax
	vmulss	%xmm11, %xmm0, %xmm13
	vaddss	%xmm10, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm9, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm8, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm7, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm6, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm5, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm4, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm0
	vcvtsi2ss	%eax, %xmm13, %xmm13
	vextractps	$2, %xmm2, %eax
	movl	%eax, %ecx
	movl	%eax, %r8d
	andl	$8388607, %eax
	sarl	$22, %ecx
	orl	$1065353216, %eax
	sarl	$23, %r8d
	andl	$1, %ecx
	movzbl	%r8b, %r8d
	vmulss	%xmm3, %xmm13, %xmm13
	movl	%ecx, %r9d
	sall	$23, %r9d
	subl	%r9d, %eax
	vaddss	%xmm13, %xmm0, %xmm0
	vinsertps	$16, %xmm0, %xmm1, %xmm1
	vmovd	%eax, %xmm0
	vsubss	%xmm12, %xmm0, %xmm0
	leal	-127(%rcx,%r8), %eax
	vmulss	%xmm11, %xmm0, %xmm13
	vaddss	%xmm10, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm9, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm8, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm7, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm6, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm5, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm13
	vaddss	%xmm4, %xmm13, %xmm13
	vmulss	%xmm0, %xmm13, %xmm0
	vcvtsi2ss	%eax, %xmm13, %xmm13
	vextractps	$3, %xmm2, %eax
	movl	%eax, %ecx
	movl	%eax, %r8d
	andl	$8388607, %eax
	sarl	$22, %ecx
	orl	$1065353216, %eax
	sarl	$23, %r8d
	andl	$1, %ecx
	movzbl	%r8b, %r8d
	vmulss	%xmm3, %xmm13, %xmm13
	movl	%ecx, %r9d
	sall	$23, %r9d
	subl	%r9d, %eax
	vmovd	%eax, %xmm2
	vaddss	%xmm13, %xmm0, %xmm0
	leal	-127(%rcx,%r8), %eax
	vinsertps	$32, %xmm0, %xmm1, %xmm1
	vsubss	%xmm12, %xmm2, %xmm0
	vmulss	%xmm11, %xmm0, %xmm2
	vaddss	%xmm10, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm9, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm8, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm7, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm6, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm5, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm4, %xmm2, %xmm2
	vmulss	%xmm0, %xmm2, %xmm0
	vcvtsi2ss	%eax, %xmm2, %xmm2
	vmulss	%xmm3, %xmm2, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	vinsertps	$48, %xmm0, %xmm1, %xmm0
	vmovaps	%xmm0, (%rsi,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	vxorps	%xmm1, %xmm1, %xmm1
	jne	L27
	rep; ret
LFE4245:
	.align 4,0x90
	.globl __Z5fillOv
__Z5fillOv:
LFB4250:
	vmovsd	LC33(%rip), %xmm0
	leaq	_va(%rip), %rax
	vxorps	%xmm1, %xmm1, %xmm1
	leaq	16384+_va(%rip), %rdx
	.align 4,0x90
L30:
	vunpcklps	%xmm1, %xmm1, %xmm1
	vcvtps2pd	%xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm1
	addq	$16, %rax
	vmovddup	%xmm1, %xmm1
	vcvtpd2psx	%xmm1, %xmm1
	vmovss	%xmm1, -16(%rax)
	vunpcklps	%xmm1, %xmm1, %xmm1
	vcvtps2pd	%xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm1
	vmovddup	%xmm1, %xmm1
	vcvtpd2psx	%xmm1, %xmm1
	vmovss	%xmm1, -12(%rax)
	vunpcklps	%xmm1, %xmm1, %xmm1
	vcvtps2pd	%xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm1
	vmovddup	%xmm1, %xmm1
	vcvtpd2psx	%xmm1, %xmm1
	vmovss	%xmm1, -8(%rax)
	vunpcklps	%xmm1, %xmm1, %xmm1
	vcvtps2pd	%xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm1
	vmovddup	%xmm1, %xmm1
	vcvtpd2psx	%xmm1, %xmm1
	vmovss	%xmm1, -4(%rax)
	cmpq	%rdx, %rax
	jne	L30
	rep; ret
LFE4250:
	.align 4,0x90
	.globl __Z5fillWv
__Z5fillWv:
LFB4251:
	vmovss	LC34(%rip), %xmm2
	leaq	_va(%rip), %rax
	vxorps	%xmm1, %xmm1, %xmm1
	vmovsd	LC33(%rip), %xmm0
	leaq	16384+_va(%rip), %rdx
	.align 4,0x90
L33:
	vunpcklps	%xmm1, %xmm1, %xmm1
	vunpcklps	%xmm2, %xmm2, %xmm2
	vcvtps2pd	%xmm1, %xmm1
	vcvtps2pd	%xmm2, %xmm2
	vaddsd	%xmm0, %xmm1, %xmm1
	addq	$16, %rax
	vsubsd	%xmm0, %xmm2, %xmm2
	vmovddup	%xmm1, %xmm1
	vcvtpd2psx	%xmm1, %xmm1
	vmovss	%xmm1, -16(%rax)
	vmovddup	%xmm2, %xmm2
	vunpcklps	%xmm1, %xmm1, %xmm1
	vcvtpd2psx	%xmm2, %xmm2
	vcvtps2pd	%xmm1, %xmm1
	vmovss	%xmm2, -12(%rax)
	vaddsd	%xmm0, %xmm1, %xmm1
	vunpcklps	%xmm2, %xmm2, %xmm2
	vcvtps2pd	%xmm2, %xmm2
	vsubsd	%xmm0, %xmm2, %xmm2
	vmovddup	%xmm1, %xmm1
	vcvtpd2psx	%xmm1, %xmm1
	vmovss	%xmm1, -8(%rax)
	vmovddup	%xmm2, %xmm2
	vcvtpd2psx	%xmm2, %xmm2
	vmovss	%xmm2, -4(%rax)
	cmpq	%rdx, %rax
	jne	L33
	rep; ret
LFE4251:
	.align 4,0x90
	.globl __Z3sumv
__Z3sumv:
LFB4252:
	leaq	_vb(%rip), %rax
	vxorps	%xmm5, %xmm5, %xmm5
	leaq	16384+_vb(%rip), %rdx
	.align 4,0x90
L36:
	vmovaps	(%rax), %ymm6
	subq	$-128, %rax
	vmovaps	-96(%rax), %ymm4
	vmovaps	-64(%rax), %ymm8
	vmovaps	-32(%rax), %ymm3
	vshufps	$136, %ymm4, %ymm6, %ymm1
	vshufps	$221, %ymm4, %ymm6, %ymm4
	vperm2f128	$3, %ymm1, %ymm1, %ymm0
	vshufps	$68, %ymm0, %ymm1, %ymm7
	vshufps	$238, %ymm0, %ymm1, %ymm0
	vinsertf128	$1, %xmm0, %ymm7, %ymm7
	vperm2f128	$3, %ymm4, %ymm4, %ymm0
	cmpq	%rdx, %rax
	vshufps	$68, %ymm0, %ymm4, %ymm6
	vshufps	$238, %ymm0, %ymm4, %ymm0
	vshufps	$136, %ymm3, %ymm8, %ymm4
	vinsertf128	$1, %xmm0, %ymm6, %ymm6
	vshufps	$221, %ymm3, %ymm8, %ymm3
	vperm2f128	$3, %ymm4, %ymm4, %ymm0
	vshufps	$68, %ymm0, %ymm4, %ymm2
	vshufps	$238, %ymm0, %ymm4, %ymm0
	vinsertf128	$1, %xmm0, %ymm2, %ymm2
	vperm2f128	$3, %ymm3, %ymm3, %ymm0
	vshufps	$68, %ymm0, %ymm3, %ymm1
	vshufps	$238, %ymm0, %ymm3, %ymm0
	vshufps	$136, %ymm2, %ymm7, %ymm3
	vinsertf128	$1, %xmm0, %ymm1, %ymm1
	vshufps	$221, %ymm2, %ymm7, %ymm2
	vperm2f128	$3, %ymm3, %ymm3, %ymm0
	vshufps	$68, %ymm0, %ymm3, %ymm4
	vshufps	$238, %ymm0, %ymm3, %ymm0
	vinsertf128	$1, %xmm0, %ymm4, %ymm3
	vperm2f128	$3, %ymm2, %ymm2, %ymm0
	vshufps	$68, %ymm0, %ymm2, %ymm4
	vshufps	$238, %ymm0, %ymm2, %ymm0
	vinsertf128	$1, %xmm0, %ymm4, %ymm0
	vaddps	%ymm0, %ymm3, %ymm3
	vshufps	$136, %ymm1, %ymm6, %ymm2
	vperm2f128	$3, %ymm2, %ymm2, %ymm0
	vshufps	$68, %ymm0, %ymm2, %ymm4
	vshufps	$238, %ymm0, %ymm2, %ymm0
	vinsertf128	$1, %xmm0, %ymm4, %ymm0
	vshufps	$221, %ymm1, %ymm6, %ymm1
	vsubps	%ymm0, %ymm3, %ymm2
	vperm2f128	$3, %ymm1, %ymm1, %ymm0
	vshufps	$68, %ymm0, %ymm1, %ymm3
	vshufps	$238, %ymm0, %ymm1, %ymm0
	vinsertf128	$1, %xmm0, %ymm3, %ymm0
	vsubps	%ymm0, %ymm2, %ymm0
	vaddps	%ymm0, %ymm5, %ymm5
	jne	L36
	vhaddps	%ymm5, %ymm5, %ymm5
	vhaddps	%ymm5, %ymm5, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm5
	vaddps	%ymm0, %ymm5, %ymm0
	vzeroupper
	ret
LFE4252:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 4
	.globl __ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineIjLm32ELm624ELm397ELm31ELj2567483615ELm11ELj4294967295ELm7ELj2636928640ELm15ELj4022730752ELm18ELj1812433253EEET_RT1_
	.weak_definition __ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineIjLm32ELm624ELm397ELm31ELj2567483615ELm11ELj4294967295ELm7ELj2636928640ELm15ELj4022730752ELm18ELj1812433253EEET_RT1_
__ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineIjLm32ELm624ELm397ELm31ELj2567483615ELm11ELj4294967295ELm7ELj2636928640ELm15ELj4022730752ELm18ELj1812433253EEET_RT1_:
LFB4505:
	pushq	%rbp
LCFI3:
	movq	%rsp, %rbp
LCFI4:
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	addq	$32, %rsp
LCFI5:
	movq	2496(%rdi), %rdx
	cmpq	$623, %rdx
	ja	L39
	leaq	1(%rdx), %rax
	movl	(%rdi,%rdx,4), %edx
	.align 4
L40:
	movq	%rax, 2496(%rdi)
	movl	%edx, %eax
	shrl	$11, %eax
	xorl	%edx, %eax
	movl	%eax, %edx
	sall	$7, %edx
	andl	$-1658038656, %edx
	xorl	%eax, %edx
	movl	%edx, %eax
	sall	$15, %eax
	andl	$-272236544, %eax
	xorl	%edx, %eax
	movl	%eax, %edx
	shrl	$18, %edx
	xorl	%edx, %eax
	vcvtsi2ssq	%rax, %xmm0, %xmm0
	vmulss	LC39(%rip), %xmm0, %xmm0
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
LCFI6:
	ret
	.align 4
L39:
LCFI7:
	movq	%rdi, %rax
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	je	L66
	movl	4(%rdi), %r8d
	movl	(%rdi), %edx
	movl	%r8d, %ecx
	andl	$-2147483648, %edx
	andl	$2147483647, %ecx
	orl	%edx, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1588(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$1, %rax
	movl	%edx, (%rdi)
	je	L67
	movl	8(%rdi), %esi
	andl	$-2147483648, %r8d
	movl	%esi, %ecx
	andl	$2147483647, %ecx
	orl	%r8d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1592(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$2, %rax
	movl	%edx, 4(%rdi)
	je	L68
	movl	12(%rdi), %r8d
	andl	$-2147483648, %esi
	movl	%r8d, %ecx
	andl	$2147483647, %ecx
	orl	%esi, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1596(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$3, %rax
	movl	%edx, 8(%rdi)
	je	L69
	movl	16(%rdi), %esi
	andl	$-2147483648, %r8d
	movl	%esi, %ecx
	andl	$2147483647, %ecx
	orl	%r8d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1600(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$4, %rax
	movl	%edx, 12(%rdi)
	je	L70
	movl	20(%rdi), %r8d
	andl	$-2147483648, %esi
	movl	%r8d, %ecx
	andl	$2147483647, %ecx
	orl	%esi, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1604(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$5, %rax
	movl	%edx, 16(%rdi)
	je	L71
	movl	24(%rdi), %esi
	andl	$-2147483648, %r8d
	movl	%esi, %ecx
	andl	$2147483647, %ecx
	orl	%r8d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1608(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$7, %rax
	movl	%edx, 20(%rdi)
	jne	L72
	movl	28(%rdi), %ecx
	andl	$-2147483648, %esi
	movl	$220, %r12d
	movl	$7, %ebx
	andl	$2147483647, %ecx
	orl	%esi, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1612(%rdi), %edx
	xorl	%ecx, %edx
	movl	%edx, 24(%rdi)
L56:
	movl	$227, %r10d
	subq	%rax, %r10
	movq	%r10, %r9
	shrq	$3, %r9
	leaq	0(,%r9,8), %r11
L55:
	salq	$2, %rax
	vmovdqa	LC35(%rip), %ymm4
	xorl	%edx, %edx
	vmovdqa	LC36(%rip), %ymm3
	leaq	(%rdi,%rax), %r8
	vpxor	%xmm7, %xmm7, %xmm7
	vmovdqa	LC37(%rip), %ymm2
	leaq	4(%rdi,%rax), %rsi
	vmovdqa	LC38(%rip), %ymm1
	leaq	1588(%rdi,%rax), %rcx
	xorl	%eax, %eax
	.align 4
L57:
	vmovdqu	(%rsi,%rax), %xmm0
	vpand	(%r8,%rax), %ymm3, %ymm5
	addq	$1, %rdx
	vmovdqu	(%rcx,%rax), %xmm6
	vinserti128	$0x1, 16(%rsi,%rax), %ymm0, %ymm0
	vpand	%ymm4, %ymm0, %ymm0
	vpor	%ymm5, %ymm0, %ymm0
	vpand	%ymm2, %ymm0, %ymm5
	vpcmpeqd	%ymm7, %ymm5, %ymm5
	vinserti128	$0x1, 16(%rcx,%rax), %ymm6, %ymm6
	vpsrld	$1, %ymm0, %ymm0
	vpandn	%ymm1, %ymm5, %ymm5
	vpxor	%ymm5, %ymm6, %ymm5
	vpxor	%ymm0, %ymm5, %ymm0
	vmovdqa	%ymm0, (%r8,%rax)
	addq	$32, %rax
	cmpq	%r9, %rdx
	jb	L57
	leaq	(%rbx,%r11), %rax
	movq	%r12, %rcx
	subq	%r11, %rcx
	cmpq	%r11, %r10
	je	L43
	leaq	1(%rax), %r8
	leaq	(%rdi,%rax,4), %r10
	movl	(%rdi,%r8,4), %r9d
	movl	(%r10), %edx
	movl	%r9d, %esi
	andl	$-2147483648, %edx
	andl	$2147483647, %esi
	orl	%edx, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1588(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$1, %rcx
	movl	%edx, (%r10)
	je	L43
	leaq	2(%rax), %r10
	andl	$-2147483648, %r9d
	movl	(%rdi,%r10,4), %r11d
	movl	%r11d, %esi
	andl	$2147483647, %esi
	orl	%r9d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1592(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$2, %rcx
	movl	%edx, (%rdi,%r8,4)
	je	L43
	leaq	3(%rax), %r8
	andl	$-2147483648, %r11d
	movl	(%rdi,%r8,4), %r9d
	movl	%r9d, %esi
	andl	$2147483647, %esi
	orl	%r11d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1596(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$3, %rcx
	movl	%edx, (%rdi,%r10,4)
	je	L43
	leaq	4(%rax), %r10
	andl	$-2147483648, %r9d
	movl	(%rdi,%r10,4), %r11d
	movl	%r11d, %esi
	andl	$2147483647, %esi
	orl	%r9d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1600(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$4, %rcx
	movl	%edx, (%rdi,%r8,4)
	je	L43
	leaq	5(%rax), %r8
	andl	$-2147483648, %r11d
	movl	(%rdi,%r8,4), %r9d
	movl	%r9d, %esi
	andl	$2147483647, %esi
	orl	%r11d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1604(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$5, %rcx
	movl	%edx, (%rdi,%r10,4)
	je	L43
	leaq	6(%rax), %r11
	andl	$-2147483648, %r9d
	movl	(%rdi,%r11,4), %r10d
	movl	%r10d, %esi
	andl	$2147483647, %esi
	orl	%r9d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1608(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$6, %rcx
	movl	%edx, (%rdi,%r8,4)
	je	L43
	movl	28(%rdi,%rax,4), %ecx
	andl	$-2147483648, %r10d
	andl	$2147483647, %ecx
	orl	%r10d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	1612(%rdi,%rax,4), %edx
	xorl	%ecx, %edx
	movl	%edx, (%rdi,%r11,4)
L43:
	leaq	908(%rdi), %rax
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	je	L59
	movl	912(%rdi), %r8d
	movl	908(%rdi), %edx
	movl	%r8d, %ecx
	andl	$-2147483648, %edx
	andl	$2147483647, %ecx
	orl	%edx, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$1, %rax
	movl	%edx, 908(%rdi)
	je	L60
	movl	916(%rdi), %esi
	andl	$-2147483648, %r8d
	movl	%esi, %ecx
	andl	$2147483647, %ecx
	orl	%r8d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	4(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$2, %rax
	movl	%edx, 912(%rdi)
	je	L61
	movl	920(%rdi), %r8d
	andl	$-2147483648, %esi
	movl	%r8d, %ecx
	andl	$2147483647, %ecx
	orl	%esi, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	8(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$3, %rax
	movl	%edx, 916(%rdi)
	je	L62
	movl	924(%rdi), %esi
	andl	$-2147483648, %r8d
	movl	%esi, %ecx
	andl	$2147483647, %ecx
	orl	%r8d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	12(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$4, %rax
	movl	%edx, 920(%rdi)
	je	L63
	movl	928(%rdi), %r8d
	andl	$-2147483648, %esi
	movl	%r8d, %ecx
	andl	$2147483647, %ecx
	orl	%esi, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	16(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$5, %rax
	movl	%edx, 924(%rdi)
	je	L64
	movl	932(%rdi), %esi
	andl	$-2147483648, %r8d
	movl	%esi, %ecx
	andl	$2147483647, %ecx
	orl	%r8d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	20(%rdi), %edx
	xorl	%ecx, %edx
	cmpq	$7, %rax
	movl	%edx, 928(%rdi)
	jne	L65
	movl	936(%rdi), %ecx
	andl	$-2147483648, %esi
	movl	$389, %ebx
	movl	$234, %r12d
	andl	$2147483647, %ecx
	orl	%esi, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	24(%rdi), %edx
	xorl	%ecx, %edx
	movl	%edx, 932(%rdi)
L53:
	movl	$396, %r10d
	subq	%rax, %r10
	movq	%r10, %r9
	shrq	$3, %r9
	leaq	0(,%r9,8), %r11
L52:
	leaq	908(,%rax,4), %rdx
	xorl	%eax, %eax
	vpxor	%xmm7, %xmm7, %xmm7
	leaq	(%rdi,%rdx), %r8
	leaq	4(%rdi,%rdx), %rsi
	leaq	-908(%rdi,%rdx), %rcx
	xorl	%edx, %edx
	.align 4
L54:
	vmovdqu	(%rsi,%rax), %xmm0
	vpand	(%r8,%rax), %ymm3, %ymm5
	addq	$1, %rdx
	vmovdqu	(%rcx,%rax), %xmm6
	vinserti128	$0x1, 16(%rsi,%rax), %ymm0, %ymm0
	vpand	%ymm4, %ymm0, %ymm0
	vpor	%ymm5, %ymm0, %ymm0
	vpand	%ymm2, %ymm0, %ymm5
	vpcmpeqd	%ymm7, %ymm5, %ymm5
	vinserti128	$0x1, 16(%rcx,%rax), %ymm6, %ymm6
	vpsrld	$1, %ymm0, %ymm0
	vpandn	%ymm1, %ymm5, %ymm5
	vpxor	%ymm5, %ymm6, %ymm5
	vpxor	%ymm0, %ymm5, %ymm0
	vmovdqa	%ymm0, (%r8,%rax)
	addq	$32, %rax
	cmpq	%rdx, %r9
	ja	L54
	leaq	(%r12,%r11), %rax
	subq	%r11, %rbx
	cmpq	%r10, %r11
	je	L47
	leaq	1(%rax), %r8
	leaq	(%rdi,%rax,4), %r10
	movl	(%rdi,%r8,4), %r9d
	movl	(%r10), %edx
	movl	%r9d, %esi
	andl	$-2147483648, %edx
	andl	$2147483647, %esi
	orl	%edx, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-908(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$1, %rbx
	movl	%edx, (%r10)
	je	L47
	leaq	2(%rax), %r10
	andl	$-2147483648, %r9d
	movl	(%rdi,%r10,4), %r11d
	movl	%r11d, %esi
	andl	$2147483647, %esi
	orl	%r9d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-904(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$2, %rbx
	movl	%edx, (%rdi,%r8,4)
	je	L47
	leaq	3(%rax), %r8
	andl	$-2147483648, %r11d
	movl	(%rdi,%r8,4), %r9d
	movl	%r9d, %esi
	andl	$2147483647, %esi
	orl	%r11d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-900(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$3, %rbx
	movl	%edx, (%rdi,%r10,4)
	je	L47
	leaq	4(%rax), %r10
	andl	$-2147483648, %r9d
	movl	(%rdi,%r10,4), %r11d
	movl	%r11d, %esi
	andl	$2147483647, %esi
	orl	%r9d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-896(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$4, %rbx
	movl	%edx, (%rdi,%r8,4)
	je	L47
	leaq	5(%rax), %r8
	andl	$-2147483648, %r11d
	movl	(%rdi,%r8,4), %r9d
	movl	%r9d, %esi
	andl	$2147483647, %esi
	orl	%r11d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-892(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$5, %rbx
	movl	%edx, (%rdi,%r10,4)
	je	L47
	leaq	6(%rax), %r11
	andl	$-2147483648, %r9d
	movl	(%rdi,%r11,4), %r10d
	movl	%r10d, %esi
	andl	$2147483647, %esi
	orl	%r9d, %esi
	movl	%esi, %edx
	shrl	%esi
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-888(%rdi,%rax,4), %edx
	xorl	%esi, %edx
	cmpq	$6, %rbx
	movl	%edx, (%rdi,%r8,4)
	je	L47
	movl	28(%rdi,%rax,4), %ecx
	andl	$-2147483648, %r10d
	andl	$2147483647, %ecx
	orl	%r10d, %ecx
	movl	%ecx, %edx
	shrl	%ecx
	andl	$1, %edx
	negl	%edx
	andl	$-1727483681, %edx
	xorl	-884(%rdi,%rax,4), %edx
	xorl	%ecx, %edx
	movl	%edx, (%rdi,%r11,4)
L47:
	movl	(%rdi), %edx
	movl	2492(%rdi), %eax
	movl	%edx, %ecx
	andl	$-2147483648, %eax
	andl	$2147483647, %ecx
	orl	%eax, %ecx
	movl	%ecx, %eax
	andl	$1, %ecx
	shrl	%eax
	negl	%ecx
	xorl	1584(%rdi), %eax
	andl	$-1727483681, %ecx
	xorl	%ecx, %eax
	movl	%eax, 2492(%rdi)
	movl	$1, %eax
	vzeroupper
	jmp	L40
	.align 4
L66:
	movl	$224, %r11d
	movl	$28, %r9d
	xorl	%ebx, %ebx
	movl	$227, %r10d
	movl	$227, %r12d
	jmp	L55
	.align 4
L59:
	movl	$227, %r12d
	movl	$396, %ebx
	movl	$396, %r10d
	movl	$49, %r9d
	movl	$392, %r11d
	jmp	L52
	.align 4
L67:
	movl	$226, %r12d
	movl	$1, %ebx
	jmp	L56
	.align 4
L65:
	movl	$390, %ebx
	movl	$233, %r12d
	jmp	L53
	.align 4
L64:
	movl	$391, %ebx
	movl	$232, %r12d
	jmp	L53
	.align 4
L63:
	movl	$392, %ebx
	movl	$231, %r12d
	jmp	L53
	.align 4
L62:
	movl	$393, %ebx
	movl	$230, %r12d
	jmp	L53
	.align 4
L61:
	movl	$394, %ebx
	movl	$229, %r12d
	jmp	L53
	.align 4
L60:
	movl	$395, %ebx
	movl	$228, %r12d
	jmp	L53
	.align 4
L72:
	movl	$221, %r12d
	movl	$6, %ebx
	jmp	L56
	.align 4
L71:
	movl	$222, %r12d
	movl	$5, %ebx
	jmp	L56
	.align 4
L70:
	movl	$223, %r12d
	movl	$4, %ebx
	jmp	L56
	.align 4
L69:
	movl	$224, %r12d
	movl	$3, %ebx
	jmp	L56
	.align 4
L68:
	movl	$225, %r12d
	movl	$2, %ebx
	jmp	L56
LFE4505:
	.text
	.align 4,0x90
	.globl __Z5fillRv
__Z5fillRv:
LFB4246:
	pushq	%r14
LCFI8:
	leaq	16384+_va(%rip), %r14
	pushq	%r13
LCFI9:
	pushq	%r12
LCFI10:
	leaq	_va(%rip), %r12
	pushq	%rbp
LCFI11:
	pushq	%rbx
LCFI12:
	subq	$16, %rsp
LCFI13:
	movq	%rsp, %r13
	.align 4,0x90
L108:
	leaq	16(%rsp), %rbp
	movq	%r13, %rbx
L109:
	leaq	_eng(%rip), %rdi
	addq	$4, %rbx
	call	__ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineIjLm32ELm624ELm397ELm31ELj2567483615ELm11ELj4294967295ELm7ELj2636928640ELm15ELj4022730752ELm18ELj1812433253EEET_RT1_
	vmovss	_rgen(%rip), %xmm1
	vmovss	4+_rgen(%rip), %xmm2
	vsubss	%xmm1, %xmm2, %xmm2
	vmulss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm1
	vmovss	%xmm1, -4(%rbx)
	cmpq	%rbp, %rbx
	jne	L109
	vmovaps	(%rsp), %xmm0
	addq	$16, %r12
	vxorps	%xmm1, %xmm1, %xmm1
	vmovaps	%xmm1, (%rsp)
	vmovaps	%xmm0, -16(%r12)
	cmpq	%r14, %r12
	jne	L108
	addq	$16, %rsp
LCFI14:
	popq	%rbx
LCFI15:
	popq	%rbp
LCFI16:
	popq	%r12
LCFI17:
	popq	%r13
LCFI18:
	popq	%r14
LCFI19:
	ret
LFE4246:
	.cstring
LC40:
	.ascii " \0"
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB4254:
	pushq	%rbp
LCFI20:
	movq	%rsp, %rbp
LCFI21:
	pushq	%r15
	pushq	%r14
LCFI22:
	leaq	_va(%rip), %r15
	pushq	%r13
	pushq	%r12
LCFI23:
	movq	%r15, %r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$192, %rsp
LCFI24:
	leaq	160(%rsp), %r14
	vmovaps	%xmm7, 144(%rsp)
	leaq	16(%r14), %rbx
L114:
	movq	%r14, %r13
L115:
	leaq	_eng(%rip), %rdi
	addq	$4, %r13
	call	__ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineIjLm32ELm624ELm397ELm31ELj2567483615ELm11ELj4294967295ELm7ELj2636928640ELm15ELj4022730752ELm18ELj1812433253EEET_RT1_
	vmovss	_rgen(%rip), %xmm1
	vmovss	4+_rgen(%rip), %xmm2
	vsubss	%xmm1, %xmm2, %xmm2
	vmulss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, -4(%r13)
	cmpq	%rbx, %r13
	jne	L115
	leaq	16384+_va(%rip), %rax
	addq	$16, %r12
	vxorps	%xmm3, %xmm3, %xmm3
	vmovaps	160(%rsp), %xmm0
	vmovaps	%xmm3, 160(%rsp)
	vmovaps	%xmm0, -16(%r12)
	cmpq	%rax, %r12
	jne	L114
	vmovaps	144(%rsp), %xmm7
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	leaq	_vb(%rip), %rbx
	xorl	%r14d, %r14d
	vmovaps	%xmm7, 112(%rsp)
	call	__Z8computeVv
	vmovss	LC16(%rip), %xmm1
	movq	%rbx, %rsi
	vmovss	LC17(%rip), %xmm2
	vxorps	%xmm12, %xmm12, %xmm12
	movl	$10000, %r10d
	vmovss	LC19(%rip), %xmm3
	vmovd	%xmm12, %r8d
	movl	$0x00000000, 144(%rsp)
	vmovss	LC21(%rip), %xmm4
	leaq	16384+_vb(%rip), %rdi
	movl	$0x00000000, 140(%rsp)
	vmovss	LC22(%rip), %xmm5
	vmovss	LC23(%rip), %xmm6
	vmovaps	112(%rsp), %xmm7
L118:
	leaq	_va(%rip), %rax
	vmovd	%r8d, %xmm0
	.align 4
L119:
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	addq	$16, %rax
	vaddsd	LC33(%rip), %xmm0, %xmm0
	leaq	16384+_va(%rip), %rcx
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -16(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -12(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -8(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpq	%rax, %rcx
	jne	L119
	rdtscp
	salq	$32, %rdx
	movq	%rax, %r9
	movl	%ecx, _taux(%rip)
	orq	%rdx, %r9
	xorl	%edx, %edx
	jmp	L121
	.align 4
L190:
	vmovaps	%xmm9, %xmm0
L125:
	vmulps	%xmm0, %xmm0, %xmm8
	vmulps	LC5(%rip), %xmm8, %xmm9
	vsubps	LC6(%rip), %xmm9, %xmm9
	vmulps	%xmm8, %xmm9, %xmm9
	vaddps	LC7(%rip), %xmm9, %xmm9
	vmulps	%xmm8, %xmm9, %xmm9
	vsubps	LC8(%rip), %xmm9, %xmm9
	vmulps	%xmm8, %xmm9, %xmm8
	vaddps	LC3(%rip), %xmm8, %xmm8
	vmulps	%xmm0, %xmm8, %xmm0
	vaddps	%xmm10, %xmm0, %xmm0
	vmovaps	%xmm0, (%rbx,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	je	L189
L121:
	vmovaps	(%r15,%rdx), %xmm0
	vmovaps	LC2(%rip), %xmm15
	vaddps	LC3(%rip), %xmm0, %xmm9
	vsubps	LC3(%rip), %xmm0, %xmm11
	vcmpltps	%xmm0, %xmm15, %xmm10
	vrcpps	%xmm9, %xmm8
	vmovmskps	%xmm10, %eax
	vandps	LC4(%rip), %xmm10, %xmm10
	testl	%eax, %eax
	vmulps	%xmm9, %xmm8, %xmm9
	vmulps	%xmm9, %xmm8, %xmm9
	vaddps	%xmm8, %xmm8, %xmm8
	vsubps	%xmm9, %xmm8, %xmm9
	vcmpleps	%xmm15, %xmm0, %xmm8
	vmulps	%xmm9, %xmm11, %xmm9
	vblendvps	%xmm8, %xmm0, %xmm9, %xmm9
	jne	L190
	vxorps	%xmm10, %xmm10, %xmm10
	jmp	L125
L189:
	rdtscp
	leaq	_vb(%rip), %rbx
	vxorps	%xmm9, %xmm9, %xmm9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	movq	%rbx, %rax
	subq	%r9, %rdx
	addq	%rdx, %r14
	.align 4
L127:
	vmovaps	32(%rax), %ymm11
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm15
	vmovaps	-128(%rax), %ymm10
	vmovaps	-32(%rax), %ymm13
	cmpq	%rdi, %rax
	vshufps	$136, %ymm11, %ymm10, %ymm8
	vshufps	$221, %ymm11, %ymm10, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm14
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vshufps	$136, %ymm13, %ymm15, %ymm8
	vinsertf128	$1, %xmm0, %ymm14, %ymm14
	vshufps	$221, %ymm13, %ymm15, %ymm13
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm11
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm11
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm10
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm10, %ymm10
	vperm2f128	$3, %ymm13, %ymm13, %ymm0
	vshufps	$68, %ymm0, %ymm13, %ymm8
	vshufps	$238, %ymm0, %ymm13, %ymm0
	vshufps	$136, %ymm10, %ymm14, %ymm13
	vinsertf128	$1, %xmm0, %ymm8, %ymm8
	vshufps	$221, %ymm10, %ymm14, %ymm10
	vperm2f128	$3, %ymm13, %ymm13, %ymm0
	vshufps	$68, %ymm0, %ymm13, %ymm15
	vshufps	$238, %ymm0, %ymm13, %ymm0
	vinsertf128	$1, %xmm0, %ymm15, %ymm15
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm13
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm13, %ymm0
	vaddps	%ymm0, %ymm15, %ymm13
	vshufps	$221, %ymm8, %ymm11, %ymm10
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm14
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm14, %ymm0
	vshufps	$136, %ymm8, %ymm11, %ymm8
	vsubps	%ymm0, %ymm13, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm11
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm0
	vsubps	%ymm0, %ymm10, %ymm0
	vaddps	%ymm0, %ymm9, %ymm9
	jne	L127
	vhaddps	%ymm9, %ymm9, %ymm9
	vhaddps	%ymm9, %ymm9, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm9
	vaddps	%ymm0, %ymm9, %ymm0
	vaddss	%xmm0, %xmm12, %xmm12
	rdtscp
	movq	%rax, %r9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %r9
	xorl	%eax, %eax
	.align 4
L129:
	vmovaps	(%r15,%rax), %xmm0
	vmovaps	%xmm0, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandps	%xmm10, %xmm2, %xmm11
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm8
	vmulss	%xmm8, %xmm8, %xmm10
	vmulss	LC20(%rip), %xmm10, %xmm9
	vaddss	%xmm4, %xmm9, %xmm9
	vmulss	%xmm10, %xmm9, %xmm9
	vaddss	%xmm5, %xmm9, %xmm9
	vmulss	%xmm10, %xmm9, %xmm9
	vaddss	%xmm6, %xmm9, %xmm9
	vmulss	%xmm10, %xmm9, %xmm9
	vaddss	%xmm1, %xmm9, %xmm9
	vmulss	%xmm8, %xmm9, %xmm9
	vshufps	$85, %xmm0, %xmm0, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vaddss	%xmm11, %xmm9, %xmm9
	vmovss	%xmm9, %xmm7, %xmm7
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm10
	vmulss	LC20(%rip), %xmm10, %xmm8
	vaddss	%xmm4, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm5, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm6, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm10
	vunpckhps	%xmm0, %xmm0, %xmm8
	vshufps	$255, %xmm0, %xmm0, %xmm0
	vaddss	%xmm1, %xmm10, %xmm10
	vmulss	%xmm9, %xmm10, %xmm9
	vaddss	%xmm1, %xmm8, %xmm10
	vaddss	%xmm11, %xmm9, %xmm11
	vsubss	%xmm1, %xmm8, %xmm9
	vinsertps	$16, %xmm11, %xmm7, %xmm7
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm10
	vmulss	LC20(%rip), %xmm10, %xmm8
	vaddss	%xmm4, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm5, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm6, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm10
	vsubss	%xmm1, %xmm0, %xmm8
	vaddss	%xmm1, %xmm10, %xmm10
	vmulss	%xmm9, %xmm10, %xmm9
	vaddss	%xmm11, %xmm9, %xmm11
	vaddss	%xmm1, %xmm0, %xmm9
	vinsertps	$32, %xmm11, %xmm7, %xmm7
	vdivss	%xmm9, %xmm8, %xmm8
	vcmpltss	%xmm0, %xmm3, %xmm9
	vandnps	%xmm0, %xmm9, %xmm0
	vandps	%xmm9, %xmm2, %xmm10
	vandps	%xmm9, %xmm8, %xmm8
	vorps	%xmm8, %xmm0, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vmulss	LC20(%rip), %xmm9, %xmm0
	vaddss	%xmm4, %xmm0, %xmm0
	vmulss	%xmm9, %xmm0, %xmm0
	vaddss	%xmm5, %xmm0, %xmm0
	vmulss	%xmm9, %xmm0, %xmm0
	vaddss	%xmm6, %xmm0, %xmm0
	vmulss	%xmm9, %xmm0, %xmm9
	vaddss	%xmm1, %xmm9, %xmm9
	vmulss	%xmm8, %xmm9, %xmm8
	vaddss	%xmm10, %xmm8, %xmm10
	vinsertps	$48, %xmm10, %xmm7, %xmm10
	vmovaps	%xmm10, (%rsi,%rax)
	addq	$16, %rax
	cmpq	$16384, %rax
	vxorps	%xmm7, %xmm7, %xmm7
	jne	L129
	rdtscp
	vxorps	%xmm9, %xmm9, %xmm9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%r9, %rdx
	addq	%rdx, %r13
	.align 4
L131:
	vmovaps	32(%rax), %ymm11
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm15
	vmovaps	-128(%rax), %ymm10
	vmovaps	-32(%rax), %ymm13
	cmpq	%rax, %rdi
	vshufps	$136, %ymm11, %ymm10, %ymm8
	vshufps	$221, %ymm11, %ymm10, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm14
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vshufps	$136, %ymm13, %ymm15, %ymm8
	vinsertf128	$1, %xmm0, %ymm14, %ymm14
	vshufps	$221, %ymm13, %ymm15, %ymm13
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm11
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm11
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm10
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm10, %ymm10
	vperm2f128	$3, %ymm13, %ymm13, %ymm0
	vshufps	$68, %ymm0, %ymm13, %ymm8
	vshufps	$238, %ymm0, %ymm13, %ymm0
	vshufps	$136, %ymm10, %ymm14, %ymm13
	vinsertf128	$1, %xmm0, %ymm8, %ymm8
	vshufps	$221, %ymm10, %ymm14, %ymm10
	vperm2f128	$3, %ymm13, %ymm13, %ymm0
	vshufps	$68, %ymm0, %ymm13, %ymm15
	vshufps	$238, %ymm0, %ymm13, %ymm0
	vinsertf128	$1, %xmm0, %ymm15, %ymm15
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm13
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm13, %ymm0
	vaddps	%ymm0, %ymm15, %ymm13
	vshufps	$136, %ymm8, %ymm11, %ymm10
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm14
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm14, %ymm0
	vshufps	$221, %ymm8, %ymm11, %ymm8
	vsubps	%ymm0, %ymm13, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm11
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm0
	vsubps	%ymm0, %ymm10, %ymm0
	vaddps	%ymm0, %ymm9, %ymm9
	jne	L131
	vhaddps	%ymm9, %ymm9, %ymm9
	vhaddps	%ymm9, %ymm9, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm9
	vaddps	%ymm0, %ymm9, %ymm0
	vaddss	140(%rsp), %xmm0, %xmm0
	vmovss	%xmm0, 140(%rsp)
	rdtscp
	vmovaps	LC9(%rip), %ymm15
	movq	%rax, %r9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rdx, %r9
	xorl	%eax, %eax
	.align 4
L133:
	vmovaps	(%r15,%rax), %ymm0
	vaddps	LC10(%rip), %ymm0, %ymm9
	vsubps	LC10(%rip), %ymm0, %ymm10
	vrcpps	%ymm9, %ymm8
	vmulps	%ymm9, %ymm8, %ymm9
	vmulps	%ymm9, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm8
	vsubps	%ymm9, %ymm8, %ymm9
	vmulps	%ymm9, %ymm10, %ymm8
	vcmpltps	%ymm0, %ymm15, %ymm10
	vblendvps	%ymm10, %ymm8, %ymm0, %ymm8
	vandps	LC15(%rip), %ymm10, %ymm10
	vmulps	%ymm8, %ymm8, %ymm0
	vmulps	LC11(%rip), %ymm0, %ymm9
	vaddps	LC12(%rip), %ymm9, %ymm9
	vmulps	%ymm0, %ymm9, %ymm9
	vaddps	LC13(%rip), %ymm9, %ymm9
	vmulps	%ymm0, %ymm9, %ymm9
	vaddps	LC14(%rip), %ymm9, %ymm9
	vmulps	%ymm9, %ymm0, %ymm0
	vaddps	LC10(%rip), %ymm0, %ymm0
	vmulps	%ymm8, %ymm0, %ymm8
	vaddps	%ymm10, %ymm8, %ymm8
	vmovaps	%ymm8, (%rsi,%rax)
	addq	$32, %rax
	cmpq	$16384, %rax
	jne	L133
	rdtscp
	vxorps	%xmm9, %xmm9, %xmm9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%r9, %rdx
	addq	%rdx, %r12
	.align 4
L135:
	vmovaps	32(%rax), %ymm11
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm15
	vmovaps	-128(%rax), %ymm10
	vmovaps	-32(%rax), %ymm13
	cmpq	%rax, %rdi
	vshufps	$136, %ymm11, %ymm10, %ymm8
	vshufps	$221, %ymm11, %ymm10, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm14
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vshufps	$136, %ymm13, %ymm15, %ymm8
	vinsertf128	$1, %xmm0, %ymm14, %ymm14
	vshufps	$221, %ymm13, %ymm15, %ymm13
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm11
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm11
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm10
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm10, %ymm10
	vperm2f128	$3, %ymm13, %ymm13, %ymm0
	vshufps	$68, %ymm0, %ymm13, %ymm8
	vshufps	$238, %ymm0, %ymm13, %ymm0
	vshufps	$136, %ymm10, %ymm14, %ymm13
	vinsertf128	$1, %xmm0, %ymm8, %ymm8
	vshufps	$221, %ymm10, %ymm14, %ymm10
	vperm2f128	$3, %ymm13, %ymm13, %ymm0
	vshufps	$68, %ymm0, %ymm13, %ymm15
	vshufps	$238, %ymm0, %ymm13, %ymm0
	vinsertf128	$1, %xmm0, %ymm15, %ymm15
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm13
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm13, %ymm0
	vaddps	%ymm0, %ymm15, %ymm13
	vshufps	$136, %ymm8, %ymm11, %ymm10
	vperm2f128	$3, %ymm10, %ymm10, %ymm0
	vshufps	$68, %ymm0, %ymm10, %ymm14
	vshufps	$238, %ymm0, %ymm10, %ymm0
	vinsertf128	$1, %xmm0, %ymm14, %ymm0
	vshufps	$221, %ymm8, %ymm11, %ymm8
	vsubps	%ymm0, %ymm13, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm11
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm0
	vsubps	%ymm0, %ymm10, %ymm0
	vaddps	%ymm0, %ymm9, %ymm9
	jne	L135
	vhaddps	%ymm9, %ymm9, %ymm9
	subl	$1, %r10d
	vhaddps	%ymm9, %ymm9, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm9
	vaddps	%ymm0, %ymm9, %ymm0
	vaddss	144(%rsp), %xmm0, %xmm0
	vmovss	%xmm0, 144(%rsp)
	jne	L118
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vunpcklps	%xmm12, %xmm12, %xmm12
	vcvtps2pd	%xmm12, %xmm0
	vmovss	%xmm6, 104(%rsp)
	vmovss	%xmm5, 108(%rsp)
	vmovss	%xmm4, 128(%rsp)
	vmovss	%xmm3, 132(%rsp)
	vmovss	%xmm2, 136(%rsp)
	vmovss	%xmm1, 112(%rsp)
	vzeroupper
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	%r14, %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	140(%rsp), %xmm0
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	%r13, %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	144(%rsp), %xmm0
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	%r12, %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	vmovss	104(%rsp), %xmm6
	vxorps	%xmm0, %xmm0, %xmm0
	vmovss	108(%rsp), %xmm5
	leaq	_va(%rip), %rax
	vmovss	128(%rsp), %xmm4
	vmovss	132(%rsp), %xmm3
	vmovss	136(%rsp), %xmm2
	vmovss	112(%rsp), %xmm1
L138:
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	addq	$16, %rax
	vaddsd	LC33(%rip), %xmm0, %xmm0
	leaq	16384+_va(%rip), %rdi
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -16(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -12(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -8(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpq	%rax, %rdi
	jne	L138
	vmovss	%xmm6, 128(%rsp)
	vmovss	%xmm5, 132(%rsp)
	vmovss	%xmm4, 136(%rsp)
	vmovss	%xmm3, 112(%rsp)
	vmovss	%xmm2, 140(%rsp)
	vmovss	%xmm1, 144(%rsp)
	call	__Z8computeVv
	leaq	176(%rsp), %rax
	vmovss	128(%rsp), %xmm6
	vmovss	132(%rsp), %xmm5
	movl	$10000, 96(%rsp)
	movq	%rax, %r14
	vmovss	136(%rsp), %xmm4
	movq	%rax, 40(%rsp)
	vmovss	112(%rsp), %xmm3
	movl	$0x00000000, 100(%rsp)
	vmovss	140(%rsp), %xmm2
	movq	$0, 72(%rsp)
	vmovss	144(%rsp), %xmm1
	movl	$0x00000000, 104(%rsp)
	movq	$0, 80(%rsp)
	movl	$0x00000000, 108(%rsp)
	movq	$0, 88(%rsp)
L140:
	leaq	_va(%rip), %r13
	movq	%rbx, 64(%rsp)
	.align 4
L141:
	leaq	192(%rsp), %r12
	movq	%r14, %rbx
L142:
	leaq	_eng(%rip), %rdi
	vmovss	%xmm3, 112(%rsp)
	addq	$4, %rbx
	vmovss	%xmm6, 128(%rsp)
	vmovss	%xmm5, 132(%rsp)
	vmovss	%xmm4, 136(%rsp)
	vmovss	%xmm2, 140(%rsp)
	vmovss	%xmm1, 144(%rsp)
	call	__ZSt18generate_canonicalIfLm24ESt23mersenne_twister_engineIjLm32ELm624ELm397ELm31ELj2567483615ELm11ELj4294967295ELm7ELj2636928640ELm15ELj4022730752ELm18ELj1812433253EEET_RT1_
	vmovss	_rgen(%rip), %xmm7
	vmovss	4+_rgen(%rip), %xmm8
	vmovss	144(%rsp), %xmm1
	vsubss	%xmm7, %xmm8, %xmm8
	vmovss	112(%rsp), %xmm3
	vmovss	140(%rsp), %xmm2
	vmovss	136(%rsp), %xmm4
	vmovss	132(%rsp), %xmm5
	vmulss	%xmm8, %xmm0, %xmm0
	vmovss	128(%rsp), %xmm6
	vaddss	%xmm7, %xmm0, %xmm7
	vmovss	%xmm7, -4(%rbx)
	cmpq	%r12, %rbx
	jne	L142
	leaq	16384+_va(%rip), %rax
	addq	$16, %r13
	vxorps	%xmm7, %xmm7, %xmm7
	vmovaps	176(%rsp), %xmm0
	vmovaps	%xmm7, 176(%rsp)
	vmovaps	%xmm0, -16(%r13)
	cmpq	%r13, %rax
	jne	L141
	movq	64(%rsp), %rbx
	rdtscp
	salq	$32, %rdx
	movq	%rax, %rsi
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rsi
	xorl	%edx, %edx
	jmp	L145
	.align 4
L192:
	vmovaps	%xmm8, %xmm0
L149:
	vmulps	%xmm0, %xmm0, %xmm7
	vmulps	LC5(%rip), %xmm7, %xmm8
	vsubps	LC6(%rip), %xmm8, %xmm8
	vmulps	%xmm7, %xmm8, %xmm8
	vaddps	LC7(%rip), %xmm8, %xmm8
	vmulps	%xmm7, %xmm8, %xmm8
	vsubps	LC8(%rip), %xmm8, %xmm8
	vmulps	%xmm7, %xmm8, %xmm7
	vaddps	LC3(%rip), %xmm7, %xmm7
	vmulps	%xmm0, %xmm7, %xmm0
	vaddps	%xmm9, %xmm0, %xmm0
	vmovaps	%xmm0, (%rbx,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	je	L191
L145:
	vmovaps	(%r15,%rdx), %xmm0
	vmovaps	LC2(%rip), %xmm7
	vaddps	LC3(%rip), %xmm0, %xmm8
	vcmpltps	%xmm0, %xmm7, %xmm9
	vsubps	LC3(%rip), %xmm0, %xmm10
	vrcpps	%xmm8, %xmm7
	vmovmskps	%xmm9, %eax
	vandps	LC4(%rip), %xmm9, %xmm9
	testl	%eax, %eax
	vmulps	%xmm8, %xmm7, %xmm8
	vmulps	%xmm8, %xmm7, %xmm8
	vaddps	%xmm7, %xmm7, %xmm7
	vsubps	%xmm8, %xmm7, %xmm8
	vcmpleps	LC2(%rip), %xmm0, %xmm7
	vmulps	%xmm8, %xmm10, %xmm8
	vblendvps	%xmm7, %xmm0, %xmm8, %xmm8
	jne	L192
	vxorps	%xmm9, %xmm9, %xmm9
	jmp	L149
L191:
	rdtscp
	vxorps	%xmm10, %xmm10, %xmm10
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%rsi, %rdx
	addq	%rdx, 88(%rsp)
	.align 4
L151:
	vmovaps	32(%rax), %ymm11
	leaq	16384+_vb(%rip), %rsi
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm13
	vmovaps	-128(%rax), %ymm8
	vmovaps	-32(%rax), %ymm9
	cmpq	%rax, %rsi
	vshufps	$136, %ymm11, %ymm8, %ymm7
	vshufps	$221, %ymm11, %ymm8, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm12
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vshufps	$136, %ymm9, %ymm13, %ymm7
	vinsertf128	$1, %xmm0, %ymm12, %ymm12
	vshufps	$221, %ymm9, %ymm13, %ymm9
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm11
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm11
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm8
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vinsertf128	$1, %xmm0, %ymm8, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm0
	vshufps	$68, %ymm0, %ymm9, %ymm7
	vshufps	$238, %ymm0, %ymm9, %ymm0
	vshufps	$136, %ymm8, %ymm12, %ymm9
	vinsertf128	$1, %xmm0, %ymm7, %ymm7
	vshufps	$221, %ymm8, %ymm12, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm0
	vshufps	$68, %ymm0, %ymm9, %ymm13
	vshufps	$238, %ymm0, %ymm9, %ymm0
	vinsertf128	$1, %xmm0, %ymm13, %ymm9
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm12
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm12, %ymm0
	vaddps	%ymm0, %ymm9, %ymm9
	vshufps	$221, %ymm7, %ymm11, %ymm8
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm12
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm12, %ymm0
	vshufps	$136, %ymm7, %ymm11, %ymm7
	vsubps	%ymm0, %ymm9, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm9
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vinsertf128	$1, %xmm0, %ymm9, %ymm0
	vsubps	%ymm0, %ymm8, %ymm0
	vaddps	%ymm0, %ymm10, %ymm10
	jne	L151
	vhaddps	%ymm10, %ymm10, %ymm10
	vhaddps	%ymm10, %ymm10, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm10
	vaddps	%ymm0, %ymm10, %ymm0
	vaddss	108(%rsp), %xmm0, %xmm7
	vmovss	%xmm7, 108(%rsp)
	rdtscp
	salq	$32, %rdx
	movq	%rax, %rsi
	vmovaps	48(%rsp), %xmm0
	orq	%rdx, %rsi
	movl	%ecx, _taux(%rip)
	xorl	%edx, %edx
	.align 4
L153:
	vmovaps	(%r15,%rdx), %xmm7
	vmovaps	%xmm7, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm8
	vmulss	LC20(%rip), %xmm8, %xmm10
	vaddss	%xmm4, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm10
	vaddss	%xmm5, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm10
	vaddss	%xmm6, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm8
	vaddss	%xmm1, %xmm8, %xmm8
	vmulss	%xmm9, %xmm8, %xmm9
	vshufps	$85, %xmm7, %xmm7, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vaddss	%xmm11, %xmm9, %xmm9
	vmovss	%xmm9, %xmm0, %xmm0
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm8
	vmulss	LC20(%rip), %xmm8, %xmm10
	vaddss	%xmm4, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm10
	vaddss	%xmm5, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm10
	vaddss	%xmm6, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm8
	vaddss	%xmm1, %xmm8, %xmm8
	vmulss	%xmm9, %xmm8, %xmm9
	vunpckhps	%xmm7, %xmm7, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vshufps	$255, %xmm7, %xmm7, %xmm7
	vaddss	%xmm11, %xmm9, %xmm9
	vinsertps	$16, %xmm9, %xmm0, %xmm0
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm8
	vmulss	LC20(%rip), %xmm8, %xmm10
	vaddss	%xmm4, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm10
	vaddss	%xmm5, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm10
	vaddss	%xmm6, %xmm10, %xmm10
	vmulss	%xmm8, %xmm10, %xmm8
	vaddss	%xmm1, %xmm8, %xmm8
	vmulss	%xmm9, %xmm8, %xmm9
	vsubss	%xmm1, %xmm7, %xmm8
	vaddss	%xmm11, %xmm9, %xmm9
	vinsertps	$32, %xmm9, %xmm0, %xmm0
	vaddss	%xmm1, %xmm7, %xmm9
	vdivss	%xmm9, %xmm8, %xmm8
	vcmpltss	%xmm7, %xmm3, %xmm9
	vandnps	%xmm7, %xmm9, %xmm7
	vandps	%xmm9, %xmm2, %xmm10
	vandps	%xmm9, %xmm8, %xmm8
	vorps	%xmm8, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm7
	vmulss	LC20(%rip), %xmm7, %xmm9
	vaddss	%xmm4, %xmm9, %xmm9
	vmulss	%xmm7, %xmm9, %xmm9
	vaddss	%xmm5, %xmm9, %xmm9
	vmulss	%xmm7, %xmm9, %xmm9
	vaddss	%xmm6, %xmm9, %xmm9
	vmulss	%xmm7, %xmm9, %xmm7
	vaddss	%xmm1, %xmm7, %xmm7
	vmulss	%xmm8, %xmm7, %xmm8
	vaddss	%xmm10, %xmm8, %xmm8
	vinsertps	$48, %xmm8, %xmm0, %xmm8
	vmovaps	%xmm8, (%rbx,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	vxorps	%xmm0, %xmm0, %xmm0
	jne	L153
	vmovaps	%xmm0, 48(%rsp)
	rdtscp
	vxorps	%xmm10, %xmm10, %xmm10
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%rsi, %rdx
	addq	%rdx, 80(%rsp)
	.align 4
L155:
	vmovaps	32(%rax), %ymm11
	leaq	16384+_vb(%rip), %rsi
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm13
	vmovaps	-128(%rax), %ymm8
	vmovaps	-32(%rax), %ymm9
	cmpq	%rax, %rsi
	vshufps	$136, %ymm11, %ymm8, %ymm7
	vshufps	$221, %ymm11, %ymm8, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm12
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vshufps	$136, %ymm9, %ymm13, %ymm7
	vinsertf128	$1, %xmm0, %ymm12, %ymm12
	vshufps	$221, %ymm9, %ymm13, %ymm9
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm11
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm11
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm8
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vinsertf128	$1, %xmm0, %ymm8, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm0
	vshufps	$68, %ymm0, %ymm9, %ymm7
	vshufps	$238, %ymm0, %ymm9, %ymm0
	vshufps	$136, %ymm8, %ymm12, %ymm9
	vinsertf128	$1, %xmm0, %ymm7, %ymm7
	vshufps	$221, %ymm8, %ymm12, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm0
	vshufps	$68, %ymm0, %ymm9, %ymm13
	vshufps	$238, %ymm0, %ymm9, %ymm0
	vinsertf128	$1, %xmm0, %ymm13, %ymm9
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm12
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm12, %ymm0
	vaddps	%ymm0, %ymm9, %ymm9
	vshufps	$221, %ymm7, %ymm11, %ymm8
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm12
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm12, %ymm0
	vshufps	$136, %ymm7, %ymm11, %ymm7
	vsubps	%ymm0, %ymm9, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm9
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vinsertf128	$1, %xmm0, %ymm9, %ymm0
	vsubps	%ymm0, %ymm8, %ymm0
	vaddps	%ymm0, %ymm10, %ymm10
	jne	L155
	vhaddps	%ymm10, %ymm10, %ymm10
	movl	$4096, %edx
	vmovss	%xmm3, 112(%rsp)
	vmovss	%xmm4, 136(%rsp)
	leaq	_va(%rip), %rsi
	vmovss	%xmm6, 128(%rsp)
	leaq	_a(%rip), %rdi
	vmovss	%xmm5, 132(%rsp)
	vhaddps	%ymm10, %ymm10, %ymm0
	vmovss	%xmm2, 140(%rsp)
	vmovss	%xmm1, 144(%rsp)
	vperm2f128	$1, %ymm0, %ymm0, %ymm10
	vaddps	%ymm0, %ymm10, %ymm0
	vaddss	104(%rsp), %xmm0, %xmm4
	vmovss	%xmm4, 104(%rsp)
	vzeroupper
	call	_memcpy
	rdtscp
	vmovaps	LC9(%rip), %ymm15
	salq	$32, %rdx
	movq	%rax, %rsi
	vmovss	112(%rsp), %xmm3
	vmovss	128(%rsp), %xmm6
	orq	%rdx, %rsi
	movl	%ecx, _taux(%rip)
	vmovss	132(%rsp), %xmm5
	xorl	%edx, %edx
	vmovss	136(%rsp), %xmm4
	vmovss	140(%rsp), %xmm2
	vmovss	144(%rsp), %xmm1
	.align 4
L157:
	leaq	_a(%rip), %rax
	vmovaps	(%rax,%rdx), %ymm0
	leaq	_b(%rip), %rax
	vaddps	LC10(%rip), %ymm0, %ymm8
	vsubps	LC10(%rip), %ymm0, %ymm9
	vrcpps	%ymm8, %ymm7
	vmulps	%ymm8, %ymm7, %ymm8
	vmulps	%ymm8, %ymm7, %ymm8
	vaddps	%ymm7, %ymm7, %ymm7
	vsubps	%ymm8, %ymm7, %ymm8
	vmulps	%ymm8, %ymm9, %ymm7
	vcmpltps	%ymm0, %ymm15, %ymm9
	vblendvps	%ymm9, %ymm7, %ymm0, %ymm7
	vandps	LC15(%rip), %ymm9, %ymm9
	vmulps	%ymm7, %ymm7, %ymm0
	vmulps	LC11(%rip), %ymm0, %ymm8
	vaddps	LC12(%rip), %ymm8, %ymm8
	vmulps	%ymm0, %ymm8, %ymm8
	vaddps	LC13(%rip), %ymm8, %ymm8
	vmulps	%ymm0, %ymm8, %ymm8
	vaddps	LC14(%rip), %ymm8, %ymm8
	vmulps	%ymm0, %ymm8, %ymm0
	vaddps	LC10(%rip), %ymm0, %ymm0
	vmulps	%ymm7, %ymm0, %ymm7
	vaddps	%ymm9, %ymm7, %ymm7
	vmovaps	%ymm7, (%rax,%rdx)
	addq	$32, %rdx
	cmpq	$16384, %rdx
	jne	L157
	vmovss	%xmm6, 128(%rsp)
	vmovss	%xmm5, 132(%rsp)
	vmovss	%xmm4, 136(%rsp)
	vmovss	%xmm3, 112(%rsp)
	vmovss	%xmm2, 140(%rsp)
	vmovss	%xmm1, 144(%rsp)
	rdtscp
	leaq	_vb(%rip), %rdi
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	subq	%rsi, %rdx
	addq	%rdx, 72(%rsp)
	leaq	_b(%rip), %rsi
	movl	$4096, %edx
	vzeroupper
	call	_memcpy
	vmovss	112(%rsp), %xmm3
	vmovss	128(%rsp), %xmm6
	leaq	_vb(%rip), %rax
	vxorps	%xmm10, %xmm10, %xmm10
	vmovss	132(%rsp), %xmm5
	vmovss	136(%rsp), %xmm4
	vmovss	140(%rsp), %xmm2
	vmovss	144(%rsp), %xmm1
	.align 4
L159:
	vmovaps	32(%rax), %ymm11
	leaq	16384+_vb(%rip), %rcx
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm13
	vmovaps	-128(%rax), %ymm8
	vmovaps	-32(%rax), %ymm9
	cmpq	%rcx, %rax
	vshufps	$136, %ymm11, %ymm8, %ymm7
	vshufps	$221, %ymm11, %ymm8, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm12
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vshufps	$136, %ymm9, %ymm13, %ymm7
	vinsertf128	$1, %xmm0, %ymm12, %ymm12
	vshufps	$221, %ymm9, %ymm13, %ymm9
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm11
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm11, %ymm11
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm8
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vinsertf128	$1, %xmm0, %ymm8, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm0
	vshufps	$68, %ymm0, %ymm9, %ymm7
	vshufps	$238, %ymm0, %ymm9, %ymm0
	vshufps	$136, %ymm8, %ymm12, %ymm9
	vinsertf128	$1, %xmm0, %ymm7, %ymm7
	vshufps	$221, %ymm8, %ymm12, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm0
	vshufps	$68, %ymm0, %ymm9, %ymm13
	vshufps	$238, %ymm0, %ymm9, %ymm0
	vinsertf128	$1, %xmm0, %ymm13, %ymm9
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm12
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm12, %ymm0
	vaddps	%ymm0, %ymm9, %ymm9
	vshufps	$221, %ymm7, %ymm11, %ymm8
	vperm2f128	$3, %ymm8, %ymm8, %ymm0
	vshufps	$68, %ymm0, %ymm8, %ymm12
	vshufps	$238, %ymm0, %ymm8, %ymm0
	vinsertf128	$1, %xmm0, %ymm12, %ymm0
	vshufps	$136, %ymm7, %ymm11, %ymm7
	vsubps	%ymm0, %ymm9, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm0
	vshufps	$68, %ymm0, %ymm7, %ymm9
	vshufps	$238, %ymm0, %ymm7, %ymm0
	vinsertf128	$1, %xmm0, %ymm9, %ymm0
	vsubps	%ymm0, %ymm8, %ymm0
	vaddps	%ymm0, %ymm10, %ymm10
	jne	L159
	subl	$1, 96(%rsp)
	vhaddps	%ymm10, %ymm10, %ymm10
	vhaddps	%ymm10, %ymm10, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm10
	vaddps	%ymm0, %ymm10, %ymm0
	vaddss	100(%rsp), %xmm0, %xmm7
	vmovss	%xmm7, 100(%rsp)
	je	L193
	vzeroupper
	movq	40(%rsp), %r14
	jmp	L140
L193:
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	108(%rsp), %xmm0
	vmovss	%xmm6, 128(%rsp)
	vmovss	%xmm5, 132(%rsp)
	vcvtps2pd	%xmm0, %xmm0
	vmovss	%xmm4, 136(%rsp)
	vmovss	%xmm3, 112(%rsp)
	vmovss	%xmm2, 140(%rsp)
	vmovss	%xmm1, 144(%rsp)
	vzeroupper
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	88(%rsp), %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	104(%rsp), %xmm0
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	80(%rsp), %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	100(%rsp), %xmm0
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	72(%rsp), %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	vmovss	112(%rsp), %xmm3
	leaq	_va(%rip), %rax
	vmovss	128(%rsp), %xmm6
	vxorps	%xmm0, %xmm0, %xmm0
	vmovss	132(%rsp), %xmm5
	vmovss	136(%rsp), %xmm4
	vmovss	140(%rsp), %xmm2
	vmovss	144(%rsp), %xmm1
L162:
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	addq	$16, %rax
	vaddsd	LC33(%rip), %xmm0, %xmm0
	leaq	16384+_va(%rip), %rdi
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -16(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -12(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -8(%rax)
	vunpcklps	%xmm0, %xmm0, %xmm0
	vcvtps2pd	%xmm0, %xmm0
	vaddsd	LC33(%rip), %xmm0, %xmm0
	vmovddup	%xmm0, %xmm0
	vcvtpd2psx	%xmm0, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpq	%rax, %rdi
	jne	L162
	vmovss	%xmm6, 104(%rsp)
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	vmovss	%xmm5, 108(%rsp)
	xorl	%r14d, %r14d
	vmovss	%xmm4, 128(%rsp)
	vmovss	%xmm3, 132(%rsp)
	vmovss	%xmm2, 136(%rsp)
	vmovss	%xmm1, 112(%rsp)
	call	__Z8computeVv
	vmovaps	16(%rsp), %xmm0
	vxorps	%xmm15, %xmm15, %xmm15
	vmovss	104(%rsp), %xmm6
	movl	$10000, %edi
	vmovd	%xmm15, %r8d
	vmovss	108(%rsp), %xmm5
	vmovss	%xmm15, 144(%rsp)
	vmovss	128(%rsp), %xmm4
	vmovss	%xmm15, 140(%rsp)
	vmovss	132(%rsp), %xmm3
	vmovss	136(%rsp), %xmm2
	vmovss	112(%rsp), %xmm1
L164:
	vmovss	LC34(%rip), %xmm8
	leaq	_va(%rip), %rax
	vmovd	%r8d, %xmm7
	.align 4
L165:
	vunpcklps	%xmm7, %xmm7, %xmm7
	vunpcklps	%xmm8, %xmm8, %xmm8
	vcvtps2pd	%xmm7, %xmm7
	vcvtps2pd	%xmm8, %xmm8
	vaddsd	LC33(%rip), %xmm7, %xmm7
	addq	$16, %rax
	vsubsd	LC33(%rip), %xmm8, %xmm8
	leaq	16384+_va(%rip), %rsi
	vmovddup	%xmm7, %xmm7
	vcvtpd2psx	%xmm7, %xmm7
	vmovss	%xmm7, -16(%rax)
	vmovddup	%xmm8, %xmm8
	vunpcklps	%xmm7, %xmm7, %xmm7
	vcvtpd2psx	%xmm8, %xmm8
	vcvtps2pd	%xmm7, %xmm7
	vmovss	%xmm8, -12(%rax)
	vunpcklps	%xmm8, %xmm8, %xmm8
	vaddsd	LC33(%rip), %xmm7, %xmm7
	vcvtps2pd	%xmm8, %xmm8
	vsubsd	LC33(%rip), %xmm8, %xmm8
	vmovddup	%xmm7, %xmm7
	vcvtpd2psx	%xmm7, %xmm7
	vmovss	%xmm7, -8(%rax)
	vmovddup	%xmm8, %xmm8
	vcvtpd2psx	%xmm8, %xmm8
	vmovss	%xmm8, -4(%rax)
	cmpq	%rax, %rsi
	jne	L165
	rdtscp
	salq	$32, %rdx
	movq	%rax, %rsi
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rsi
	xorl	%edx, %edx
	jmp	L167
	.align 4
L195:
	vmovaps	%xmm9, %xmm7
L171:
	vmulps	%xmm7, %xmm7, %xmm8
	vmulps	LC5(%rip), %xmm8, %xmm9
	vsubps	LC6(%rip), %xmm9, %xmm9
	vmulps	%xmm8, %xmm9, %xmm9
	vaddps	LC7(%rip), %xmm9, %xmm9
	vmulps	%xmm8, %xmm9, %xmm9
	vsubps	LC8(%rip), %xmm9, %xmm9
	vmulps	%xmm8, %xmm9, %xmm8
	vaddps	LC3(%rip), %xmm8, %xmm8
	vmulps	%xmm7, %xmm8, %xmm7
	vaddps	%xmm10, %xmm7, %xmm7
	vmovaps	%xmm7, (%rbx,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	je	L194
L167:
	vmovaps	(%r15,%rdx), %xmm7
	vmovaps	LC2(%rip), %xmm14
	vaddps	LC3(%rip), %xmm7, %xmm9
	vsubps	LC3(%rip), %xmm7, %xmm11
	vcmpltps	%xmm7, %xmm14, %xmm10
	vrcpps	%xmm9, %xmm8
	vmovmskps	%xmm10, %ecx
	vandps	LC4(%rip), %xmm10, %xmm10
	testl	%ecx, %ecx
	vmulps	%xmm9, %xmm8, %xmm9
	vmulps	%xmm9, %xmm8, %xmm9
	vaddps	%xmm8, %xmm8, %xmm8
	vsubps	%xmm9, %xmm8, %xmm9
	vcmpleps	%xmm14, %xmm7, %xmm8
	vmulps	%xmm9, %xmm11, %xmm9
	vblendvps	%xmm8, %xmm7, %xmm9, %xmm9
	jne	L195
	vxorps	%xmm10, %xmm10, %xmm10
	jmp	L171
L194:
	rdtscp
	vxorps	%xmm9, %xmm9, %xmm9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%rsi, %rdx
	addq	%rdx, %r14
	.align 4
L173:
	vmovaps	32(%rax), %ymm11
	leaq	16384+_vb(%rip), %rsi
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm14
	vmovaps	-128(%rax), %ymm10
	vmovaps	-32(%rax), %ymm12
	cmpq	%rax, %rsi
	vshufps	$136, %ymm11, %ymm10, %ymm8
	vshufps	$221, %ymm11, %ymm10, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm13
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vshufps	$136, %ymm12, %ymm14, %ymm8
	vinsertf128	$1, %xmm7, %ymm13, %ymm13
	vshufps	$221, %ymm12, %ymm14, %ymm12
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm11
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm11, %ymm11
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm10
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vinsertf128	$1, %xmm7, %ymm10, %ymm10
	vperm2f128	$3, %ymm12, %ymm12, %ymm7
	vshufps	$68, %ymm7, %ymm12, %ymm8
	vshufps	$238, %ymm7, %ymm12, %ymm7
	vshufps	$136, %ymm10, %ymm13, %ymm12
	vinsertf128	$1, %xmm7, %ymm8, %ymm8
	vshufps	$221, %ymm10, %ymm13, %ymm10
	vperm2f128	$3, %ymm12, %ymm12, %ymm7
	vshufps	$68, %ymm7, %ymm12, %ymm14
	vshufps	$238, %ymm7, %ymm12, %ymm7
	vinsertf128	$1, %xmm7, %ymm14, %ymm14
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm12
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm12, %ymm7
	vaddps	%ymm7, %ymm14, %ymm12
	vshufps	$221, %ymm8, %ymm11, %ymm10
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm13
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm13, %ymm7
	vshufps	$136, %ymm8, %ymm11, %ymm8
	vsubps	%ymm7, %ymm12, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm11
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vinsertf128	$1, %xmm7, %ymm11, %ymm7
	vsubps	%ymm7, %ymm10, %ymm7
	vaddps	%ymm7, %ymm9, %ymm9
	jne	L173
	vhaddps	%ymm9, %ymm9, %ymm9
	vhaddps	%ymm9, %ymm9, %ymm7
	vperm2f128	$1, %ymm7, %ymm7, %ymm9
	vaddps	%ymm7, %ymm9, %ymm7
	vaddss	140(%rsp), %xmm7, %xmm7
	vmovss	%xmm7, 140(%rsp)
	rdtscp
	salq	$32, %rdx
	movq	%rax, %rsi
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rsi
	xorl	%edx, %edx
	.align 4
L175:
	vmovaps	(%r15,%rdx), %xmm7
	vmovaps	%xmm7, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandps	%xmm10, %xmm2, %xmm11
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm8
	vmulss	%xmm8, %xmm8, %xmm10
	vmulss	LC20(%rip), %xmm10, %xmm9
	vaddss	%xmm4, %xmm9, %xmm9
	vmulss	%xmm10, %xmm9, %xmm9
	vaddss	%xmm5, %xmm9, %xmm9
	vmulss	%xmm10, %xmm9, %xmm9
	vaddss	%xmm6, %xmm9, %xmm9
	vmulss	%xmm10, %xmm9, %xmm9
	vaddss	%xmm1, %xmm9, %xmm9
	vmulss	%xmm8, %xmm9, %xmm9
	vshufps	$85, %xmm7, %xmm7, %xmm8
	vaddss	%xmm1, %xmm8, %xmm10
	vaddss	%xmm11, %xmm9, %xmm9
	vmovss	%xmm9, %xmm0, %xmm0
	vsubss	%xmm1, %xmm8, %xmm9
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm10
	vmulss	LC20(%rip), %xmm10, %xmm8
	vaddss	%xmm4, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm5, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm6, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm10
	vunpckhps	%xmm7, %xmm7, %xmm8
	vshufps	$255, %xmm7, %xmm7, %xmm7
	vaddss	%xmm1, %xmm10, %xmm10
	vmulss	%xmm9, %xmm10, %xmm9
	vaddss	%xmm1, %xmm8, %xmm10
	vaddss	%xmm11, %xmm9, %xmm11
	vsubss	%xmm1, %xmm8, %xmm9
	vinsertps	$16, %xmm11, %xmm0, %xmm0
	vdivss	%xmm10, %xmm9, %xmm9
	vcmpltss	%xmm8, %xmm3, %xmm10
	vandnps	%xmm8, %xmm10, %xmm8
	vandps	%xmm10, %xmm2, %xmm11
	vandps	%xmm10, %xmm9, %xmm9
	vorps	%xmm9, %xmm8, %xmm9
	vmulss	%xmm9, %xmm9, %xmm10
	vmulss	LC20(%rip), %xmm10, %xmm8
	vaddss	%xmm4, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm5, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm8
	vaddss	%xmm6, %xmm8, %xmm8
	vmulss	%xmm10, %xmm8, %xmm10
	vsubss	%xmm1, %xmm7, %xmm8
	vaddss	%xmm1, %xmm10, %xmm10
	vmulss	%xmm9, %xmm10, %xmm9
	vaddss	%xmm11, %xmm9, %xmm11
	vaddss	%xmm1, %xmm7, %xmm9
	vinsertps	$32, %xmm11, %xmm0, %xmm0
	vdivss	%xmm9, %xmm8, %xmm8
	vcmpltss	%xmm7, %xmm3, %xmm9
	vandnps	%xmm7, %xmm9, %xmm7
	vandps	%xmm9, %xmm2, %xmm10
	vandps	%xmm9, %xmm8, %xmm8
	vorps	%xmm8, %xmm7, %xmm8
	vmulss	%xmm8, %xmm8, %xmm9
	vmulss	LC20(%rip), %xmm9, %xmm7
	vaddss	%xmm4, %xmm7, %xmm7
	vmulss	%xmm9, %xmm7, %xmm7
	vaddss	%xmm5, %xmm7, %xmm7
	vmulss	%xmm9, %xmm7, %xmm7
	vaddss	%xmm6, %xmm7, %xmm7
	vmulss	%xmm9, %xmm7, %xmm9
	vaddss	%xmm1, %xmm9, %xmm9
	vmulss	%xmm8, %xmm9, %xmm8
	vaddss	%xmm10, %xmm8, %xmm10
	vinsertps	$48, %xmm10, %xmm0, %xmm10
	vmovaps	%xmm10, (%rbx,%rdx)
	addq	$16, %rdx
	cmpq	$16384, %rdx
	vxorps	%xmm0, %xmm0, %xmm0
	jne	L175
	rdtscp
	vxorps	%xmm9, %xmm9, %xmm9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%rsi, %rdx
	addq	%rdx, %r13
	.align 4
L177:
	vmovaps	32(%rax), %ymm11
	leaq	16384+_vb(%rip), %rcx
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm14
	vmovaps	-128(%rax), %ymm10
	vmovaps	-32(%rax), %ymm12
	cmpq	%rax, %rcx
	vshufps	$136, %ymm11, %ymm10, %ymm8
	vshufps	$221, %ymm11, %ymm10, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm13
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vshufps	$136, %ymm12, %ymm14, %ymm8
	vinsertf128	$1, %xmm7, %ymm13, %ymm13
	vshufps	$221, %ymm12, %ymm14, %ymm12
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm11
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm11, %ymm11
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm10
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vinsertf128	$1, %xmm7, %ymm10, %ymm10
	vperm2f128	$3, %ymm12, %ymm12, %ymm7
	vshufps	$68, %ymm7, %ymm12, %ymm8
	vshufps	$238, %ymm7, %ymm12, %ymm7
	vshufps	$136, %ymm10, %ymm13, %ymm12
	vinsertf128	$1, %xmm7, %ymm8, %ymm8
	vshufps	$221, %ymm10, %ymm13, %ymm10
	vperm2f128	$3, %ymm12, %ymm12, %ymm7
	vshufps	$68, %ymm7, %ymm12, %ymm14
	vshufps	$238, %ymm7, %ymm12, %ymm7
	vinsertf128	$1, %xmm7, %ymm14, %ymm14
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm12
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm12, %ymm7
	vaddps	%ymm7, %ymm14, %ymm12
	vshufps	$136, %ymm8, %ymm11, %ymm10
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm13
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm13, %ymm7
	vshufps	$221, %ymm8, %ymm11, %ymm8
	vsubps	%ymm7, %ymm12, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm11
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vinsertf128	$1, %xmm7, %ymm11, %ymm7
	vsubps	%ymm7, %ymm10, %ymm7
	vaddps	%ymm7, %ymm9, %ymm9
	jne	L177
	vhaddps	%ymm9, %ymm9, %ymm9
	vhaddps	%ymm9, %ymm9, %ymm7
	vperm2f128	$1, %ymm7, %ymm7, %ymm9
	vaddps	%ymm7, %ymm9, %ymm7
	vaddss	144(%rsp), %xmm7, %xmm7
	vmovss	%xmm7, 144(%rsp)
	rdtscp
	vmovaps	LC9(%rip), %ymm14
	salq	$32, %rdx
	movq	%rax, %rsi
	movl	%ecx, _taux(%rip)
	orq	%rdx, %rsi
	xorl	%edx, %edx
	.align 4
L179:
	vmovaps	(%r15,%rdx), %ymm7
	vaddps	LC10(%rip), %ymm7, %ymm9
	vsubps	LC10(%rip), %ymm7, %ymm10
	vrcpps	%ymm9, %ymm8
	vmulps	%ymm9, %ymm8, %ymm9
	vmulps	%ymm9, %ymm8, %ymm9
	vaddps	%ymm8, %ymm8, %ymm8
	vsubps	%ymm9, %ymm8, %ymm9
	vmulps	%ymm9, %ymm10, %ymm8
	vcmpltps	%ymm7, %ymm14, %ymm10
	vblendvps	%ymm10, %ymm8, %ymm7, %ymm8
	vandps	LC15(%rip), %ymm10, %ymm10
	vmulps	%ymm8, %ymm8, %ymm7
	vmulps	LC11(%rip), %ymm7, %ymm9
	vaddps	LC12(%rip), %ymm9, %ymm9
	vmulps	%ymm7, %ymm9, %ymm9
	vaddps	LC13(%rip), %ymm9, %ymm9
	vmulps	%ymm7, %ymm9, %ymm9
	vaddps	LC14(%rip), %ymm9, %ymm9
	vmulps	%ymm7, %ymm9, %ymm7
	vaddps	LC10(%rip), %ymm7, %ymm7
	vmulps	%ymm8, %ymm7, %ymm8
	vaddps	%ymm10, %ymm8, %ymm8
	vmovaps	%ymm8, (%rbx,%rdx)
	addq	$32, %rdx
	cmpq	$16384, %rdx
	jne	L179
	rdtscp
	vxorps	%xmm9, %xmm9, %xmm9
	salq	$32, %rdx
	movl	%ecx, _taux(%rip)
	orq	%rax, %rdx
	leaq	_vb(%rip), %rax
	subq	%rsi, %rdx
	addq	%rdx, %r12
	.align 4
L181:
	vmovaps	32(%rax), %ymm11
	leaq	16384+_vb(%rip), %rsi
	subq	$-128, %rax
	vmovaps	-64(%rax), %ymm14
	vmovaps	-128(%rax), %ymm10
	vmovaps	-32(%rax), %ymm12
	cmpq	%rsi, %rax
	vshufps	$136, %ymm11, %ymm10, %ymm8
	vshufps	$221, %ymm11, %ymm10, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm13
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vshufps	$136, %ymm12, %ymm14, %ymm8
	vinsertf128	$1, %xmm7, %ymm13, %ymm13
	vshufps	$221, %ymm12, %ymm14, %ymm12
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm11
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm11, %ymm11
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm10
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vinsertf128	$1, %xmm7, %ymm10, %ymm10
	vperm2f128	$3, %ymm12, %ymm12, %ymm7
	vshufps	$68, %ymm7, %ymm12, %ymm8
	vshufps	$238, %ymm7, %ymm12, %ymm7
	vshufps	$136, %ymm10, %ymm13, %ymm12
	vinsertf128	$1, %xmm7, %ymm8, %ymm8
	vshufps	$221, %ymm10, %ymm13, %ymm10
	vperm2f128	$3, %ymm12, %ymm12, %ymm7
	vshufps	$68, %ymm7, %ymm12, %ymm14
	vshufps	$238, %ymm7, %ymm12, %ymm7
	vinsertf128	$1, %xmm7, %ymm14, %ymm14
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm12
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm12, %ymm7
	vaddps	%ymm7, %ymm14, %ymm12
	vshufps	$221, %ymm8, %ymm11, %ymm10
	vperm2f128	$3, %ymm10, %ymm10, %ymm7
	vshufps	$68, %ymm7, %ymm10, %ymm13
	vshufps	$238, %ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm13, %ymm7
	vshufps	$136, %ymm8, %ymm11, %ymm8
	vsubps	%ymm7, %ymm12, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm7
	vshufps	$68, %ymm7, %ymm8, %ymm11
	vshufps	$238, %ymm7, %ymm8, %ymm7
	vinsertf128	$1, %xmm7, %ymm11, %ymm7
	vsubps	%ymm7, %ymm10, %ymm7
	vaddps	%ymm7, %ymm9, %ymm9
	jne	L181
	vhaddps	%ymm9, %ymm9, %ymm9
	subl	$1, %edi
	vhaddps	%ymm9, %ymm9, %ymm7
	vperm2f128	$1, %ymm7, %ymm7, %ymm9
	vaddps	%ymm7, %ymm9, %ymm7
	vaddss	%xmm7, %xmm15, %xmm15
	jne	L164
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	140(%rsp), %xmm11
	vmovss	%xmm15, 140(%rsp)
	vunpcklps	%xmm11, %xmm11, %xmm11
	vcvtps2pd	%xmm11, %xmm0
	vzeroupper
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	%r14, %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	144(%rsp), %xmm0
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	%r13, %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	vmovss	140(%rsp), %xmm15
	vunpcklps	%xmm15, %xmm15, %xmm15
	vcvtps2pd	%xmm15, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC40(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vcvtsi2sdq	%r12, %xmm0, %xmm0
	movq	%rax, %rdi
	vmulsd	LC41(%rip), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	leaq	-40(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
LCFI25:
	ret
LFE4254:
	.align 4
__GLOBAL__sub_I_VAtan.cpp:
LFB4546:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI26:
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
L197:
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
	jne	L197
	movl	$5489, %edx
	movw	$1, %cx
	movl	$440509467, %edi
	movq	$624, 2496+_eng(%rip)
	leaq	_eng2(%rip), %r8
	movl	$5489, _eng2(%rip)
	.align 4
L199:
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
	jne	L199
	movq	$624, 2496+_eng2(%rip)
	movl	$0x00000000, _rgen(%rip)
	movl	$0x3f800000, 4+_rgen(%rip)
	addq	$8, %rsp
LCFI27:
	ret
LFE4546:
	.globl _taux
	.zerofill __DATA,__pu_bss2,_taux,4,2
	.globl _rgen
	.zerofill __DATA,__pu_bss5,_rgen,8,5
	.globl _eng2
	.zerofill __DATA,__pu_bss5,_eng2,2504,5
	.globl _eng
	.zerofill __DATA,__pu_bss5,_eng,2504,5
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,16384,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,16384,5
	.globl _vc
	.zerofill __DATA,__pu_bss5,_vc,16384,5
	.globl _vb
	.zerofill __DATA,__pu_bss5,_vb,16384,5
	.globl _va
	.zerofill __DATA,__pu_bss5,_va,16384,5
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal16
	.align 4
LC0:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 4
LC1:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.align 4
LC2:
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.align 4
LC3:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC4:
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.align 4
LC5:
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.align 4
LC6:
	.long	1041111941
	.long	1041111941
	.long	1041111941
	.long	1041111941
	.align 4
LC7:
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.align 4
LC8:
	.long	1051372074
	.long	1051372074
	.long	1051372074
	.long	1051372074
	.const
	.align 5
LC9:
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.align 5
LC10:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 5
LC11:
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.align 5
LC12:
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.align 5
LC13:
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.align 5
LC14:
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.align 5
LC15:
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.literal4
	.align 2
LC16:
	.long	1065353216
	.align 2
LC17:
	.long	1061752795
	.align 2
LC19:
	.long	1054086093
	.align 2
LC20:
	.long	1034219729
	.align 2
LC21:
	.long	3188595589
	.align 2
LC22:
	.long	1045205599
	.align 2
LC23:
	.long	3198855722
	.align 2
LC24:
	.long	3179186122
	.align 2
LC25:
	.long	1041113284
	.align 2
LC26:
	.long	3191002492
	.align 2
LC27:
	.long	1045333864
	.align 2
LC28:
	.long	3196041016
	.align 2
LC29:
	.long	1051369742
	.align 2
LC30:
	.long	3204448304
	.align 2
LC31:
	.long	1065353222
	.align 2
LC32:
	.long	1060205080
	.literal8
	.align 3
LC33:
	.long	2576980378
	.long	1059690905
	.literal4
	.align 2
LC34:
	.long	1061997773
	.const
	.align 5
LC35:
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.align 5
LC36:
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.align 5
LC37:
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.align 5
LC38:
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.literal4
	.align 2
LC39:
	.long	796917760
	.literal8
	.align 3
LC41:
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
	.quad	LFB4239-.
	.set L$set$2,LFE4239-LFB4239
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB4239
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
LSFDE3:
	.set L$set$6,LEFDE3-LASFDE3
	.long L$set$6
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB4240-.
	.set L$set$7,LFE4240-LFB4240
	.quad L$set$7
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$8,LEFDE5-LASFDE5
	.long L$set$8
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB4241-.
	.set L$set$9,LFE4241-LFB4241
	.quad L$set$9
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$10,LEFDE7-LASFDE7
	.long L$set$10
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4242-.
	.set L$set$11,LFE4242-LFB4242
	.quad L$set$11
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$12,LEFDE9-LASFDE9
	.long L$set$12
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4243-.
	.set L$set$13,LFE4243-LFB4243
	.quad L$set$13
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$14,LEFDE11-LASFDE11
	.long L$set$14
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB4244-.
	.set L$set$15,LFE4244-LFB4244
	.quad L$set$15
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$16,LEFDE13-LASFDE13
	.long L$set$16
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB4245-.
	.set L$set$17,LFE4245-LFB4245
	.quad L$set$17
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$18,LEFDE15-LASFDE15
	.long L$set$18
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB4250-.
	.set L$set$19,LFE4250-LFB4250
	.quad L$set$19
	.byte	0
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$20,LEFDE17-LASFDE17
	.long L$set$20
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB4251-.
	.set L$set$21,LFE4251-LFB4251
	.quad L$set$21
	.byte	0
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$22,LEFDE19-LASFDE19
	.long L$set$22
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB4252-.
	.set L$set$23,LFE4252-LFB4252
	.quad L$set$23
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$24,LEFDE21-LASFDE21
	.long L$set$24
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB4505-.
	.set L$set$25,LFE4505-LFB4505
	.quad L$set$25
	.byte	0
	.byte	0x4
	.set L$set$26,LCFI3-LFB4505
	.long L$set$26
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$27,LCFI4-LCFI3
	.long L$set$27
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$28,LCFI5-LCFI4
	.long L$set$28
	.byte	0x8c
	.byte	0x3
	.byte	0x83
	.byte	0x4
	.byte	0x4
	.set L$set$29,LCFI6-LCFI5
	.long L$set$29
	.byte	0xa
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$30,LCFI7-LCFI6
	.long L$set$30
	.byte	0xb
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$31,LEFDE23-LASFDE23
	.long L$set$31
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB4246-.
	.set L$set$32,LFE4246-LFB4246
	.quad L$set$32
	.byte	0
	.byte	0x4
	.set L$set$33,LCFI8-LFB4246
	.long L$set$33
	.byte	0xe
	.byte	0x10
	.byte	0x8e
	.byte	0x2
	.byte	0x4
	.set L$set$34,LCFI9-LCFI8
	.long L$set$34
	.byte	0xe
	.byte	0x18
	.byte	0x8d
	.byte	0x3
	.byte	0x4
	.set L$set$35,LCFI10-LCFI9
	.long L$set$35
	.byte	0xe
	.byte	0x20
	.byte	0x8c
	.byte	0x4
	.byte	0x4
	.set L$set$36,LCFI11-LCFI10
	.long L$set$36
	.byte	0xe
	.byte	0x28
	.byte	0x86
	.byte	0x5
	.byte	0x4
	.set L$set$37,LCFI12-LCFI11
	.long L$set$37
	.byte	0xe
	.byte	0x30
	.byte	0x83
	.byte	0x6
	.byte	0x4
	.set L$set$38,LCFI13-LCFI12
	.long L$set$38
	.byte	0xe
	.byte	0x40
	.byte	0x4
	.set L$set$39,LCFI14-LCFI13
	.long L$set$39
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$40,LCFI15-LCFI14
	.long L$set$40
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$41,LCFI16-LCFI15
	.long L$set$41
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$42,LCFI17-LCFI16
	.long L$set$42
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$43,LCFI18-LCFI17
	.long L$set$43
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$44,LCFI19-LCFI18
	.long L$set$44
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE23:
LSFDE25:
	.set L$set$45,LEFDE25-LASFDE25
	.long L$set$45
LASFDE25:
	.long	LASFDE25-EH_frame1
	.quad	LFB4254-.
	.set L$set$46,LFE4254-LFB4254
	.quad L$set$46
	.byte	0
	.byte	0x4
	.set L$set$47,LCFI20-LFB4254
	.long L$set$47
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$48,LCFI21-LCFI20
	.long L$set$48
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$49,LCFI22-LCFI21
	.long L$set$49
	.byte	0x8f
	.byte	0x3
	.byte	0x8e
	.byte	0x4
	.byte	0x4
	.set L$set$50,LCFI23-LCFI22
	.long L$set$50
	.byte	0x8d
	.byte	0x5
	.byte	0x8c
	.byte	0x6
	.byte	0x4
	.set L$set$51,LCFI24-LCFI23
	.long L$set$51
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$52,LCFI25-LCFI24
	.long L$set$52
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE25:
LSFDE27:
	.set L$set$53,LEFDE27-LASFDE27
	.long L$set$53
LASFDE27:
	.long	LASFDE27-EH_frame1
	.quad	LFB4546-.
	.set L$set$54,LFE4546-LFB4546
	.quad L$set$54
	.byte	0
	.byte	0x4
	.set L$set$55,LCFI26-LFB4546
	.long L$set$55
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$56,LCFI27-LCFI26
	.long L$set$56
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE27:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_VAtan.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
