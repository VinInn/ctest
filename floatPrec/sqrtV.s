	.text
	.align 4,0x90
	.globl __Z10computeOnev
__Z10computeOnev:
LFB224:
	pushq	%rbp
LCFI0:
	vxorps	%xmm2, %xmm2, %xmm2
	movq	%rsp, %rbp
LCFI1:
	andq	$-32, %rsp
	addq	$16, %rsp
	vmovaps	_va(%rip), %ymm1
	vcmpneqps	%ymm1, %ymm2, %ymm2
	vrsqrtps	%ymm1, %ymm0
	vandps	%ymm2, %ymm0, %ymm0
	vmulps	%ymm1, %ymm0, %ymm1
	vmulps	%ymm0, %ymm1, %ymm0
	vaddps	LC0(%rip), %ymm0, %ymm0
	vmulps	LC1(%rip), %ymm1, %ymm1
	vmulps	%ymm1, %ymm0, %ymm1
	vmovaps	%ymm1, _vb(%rip)
	vzeroupper
	leave
LCFI2:
	ret
LFE224:
	.align 4,0x90
	.globl __Z8computeSv
__Z8computeSv:
LFB225:
	leaq	_va(%rip), %rcx
	xorl	%eax, %eax
	leaq	_vb(%rip), %rdx
	.align 4,0x90
L5:
	vmovaps	(%rcx,%rax), %ymm1
	vmovaps	%xmm1, %xmm2
	vextractf128	$0x1, %ymm1, %xmm1
	vmovaps	%xmm2, %xmm3
	vsqrtss	%xmm3, %xmm4, %xmm4
	vmovss	%xmm4, %xmm0, %xmm3
	vshufps	$85, %xmm2, %xmm2, %xmm4
	vsqrtss	%xmm4, %xmm4, %xmm4
	vinsertf128	$0x0, %xmm3, %ymm0, %ymm0
	vmovaps	%xmm0, %xmm3
	vinsertps	$16, %xmm4, %xmm3, %xmm3
	vunpckhps	%xmm2, %xmm2, %xmm4
	vsqrtss	%xmm4, %xmm4, %xmm4
	vshufps	$255, %xmm2, %xmm2, %xmm2
	vsqrtss	%xmm2, %xmm2, %xmm2
	vinsertf128	$0x0, %xmm3, %ymm0, %ymm0
	vmovaps	%xmm0, %xmm3
	vinsertps	$32, %xmm4, %xmm3, %xmm3
	vinsertf128	$0x0, %xmm3, %ymm0, %ymm0
	vmovaps	%xmm0, %xmm3
	vinsertps	$48, %xmm2, %xmm3, %xmm3
	vmovaps	%xmm1, %xmm2
	vinsertf128	$0x0, %xmm3, %ymm0, %ymm0
	vsqrtss	%xmm2, %xmm3, %xmm3
	vextractf128	$0x1, %ymm0, %xmm2
	vmovss	%xmm3, %xmm2, %xmm2
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vsqrtss	%xmm3, %xmm3, %xmm3
	vinsertf128	$0x1, %xmm2, %ymm0, %ymm0
	vextractf128	$0x1, %ymm0, %xmm2
	vinsertps	$16, %xmm3, %xmm2, %xmm2
	vunpckhps	%xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vsqrtss	%xmm3, %xmm3, %xmm3
	vinsertf128	$0x1, %xmm2, %ymm0, %ymm0
	vextractf128	$0x1, %ymm0, %xmm2
	vsqrtss	%xmm1, %xmm1, %xmm1
	vinsertps	$32, %xmm3, %xmm2, %xmm2
	vinsertf128	$0x1, %xmm2, %ymm0, %ymm0
	vextractf128	$0x1, %ymm0, %xmm2
	vinsertps	$48, %xmm1, %xmm2, %xmm2
	vinsertf128	$0x1, %xmm2, %ymm0, %ymm2
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%ymm2, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$32768, %rax
	jne	L5
	vzeroupper
	ret
LFE225:
	.align 4,0x90
	.globl __Z8computeLv
__Z8computeLv:
LFB226:
	pushq	%rbp
LCFI3:
	leaq	_va(%rip), %rax
	vxorps	%xmm15, %xmm15, %xmm15
	leaq	_vb(%rip), %rdx
	movq	%rsp, %rbp
LCFI4:
	andq	$-32, %rsp
	leaq	32768+_va(%rip), %rcx
	addq	$16, %rsp
	vmovaps	LC0(%rip), %ymm1
	vmovaps	LC1(%rip), %ymm0
	.align 4,0x90
L9:
	vmovaps	(%rax), %ymm8
	addq	$256, %rax
	addq	$256, %rdx
	vmovaps	-192(%rax), %ymm10
	vmovaps	-128(%rax), %ymm6
	vmovaps	-64(%rax), %ymm12
	vmovaps	-224(%rax), %ymm4
	vmovaps	-160(%rax), %ymm5
	vmovaps	-96(%rax), %ymm3
	vshufps	$136, %ymm4, %ymm8, %ymm7
	vshufps	$221, %ymm4, %ymm8, %ymm4
	vmovaps	-32(%rax), %ymm9
	vperm2f128	$3, %ymm7, %ymm7, %ymm2
	vshufps	$68, %ymm2, %ymm7, %ymm11
	vshufps	$238, %ymm2, %ymm7, %ymm2
	vinsertf128	$1, %xmm2, %ymm11, %ymm11
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm7
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vshufps	$136, %ymm5, %ymm10, %ymm4
	vinsertf128	$1, %xmm2, %ymm7, %ymm7
	vshufps	$221, %ymm5, %ymm10, %ymm5
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm8
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vinsertf128	$1, %xmm2, %ymm8, %ymm8
	vperm2f128	$3, %ymm5, %ymm5, %ymm2
	vshufps	$68, %ymm2, %ymm5, %ymm4
	vshufps	$238, %ymm2, %ymm5, %ymm2
	vshufps	$136, %ymm3, %ymm6, %ymm5
	vinsertf128	$1, %xmm2, %ymm4, %ymm4
	vshufps	$221, %ymm3, %ymm6, %ymm3
	vperm2f128	$3, %ymm5, %ymm5, %ymm2
	vshufps	$68, %ymm2, %ymm5, %ymm10
	vshufps	$238, %ymm2, %ymm5, %ymm2
	vinsertf128	$1, %xmm2, %ymm10, %ymm10
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm5
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vshufps	$136, %ymm9, %ymm12, %ymm3
	vinsertf128	$1, %xmm2, %ymm5, %ymm5
	vshufps	$221, %ymm9, %ymm12, %ymm9
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm6
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm6, %ymm6
	vperm2f128	$3, %ymm9, %ymm9, %ymm2
	vshufps	$68, %ymm2, %ymm9, %ymm3
	vshufps	$238, %ymm2, %ymm9, %ymm2
	vshufps	$136, %ymm8, %ymm11, %ymm9
	vinsertf128	$1, %xmm2, %ymm3, %ymm3
	vshufps	$221, %ymm8, %ymm11, %ymm8
	vperm2f128	$3, %ymm9, %ymm9, %ymm2
	vshufps	$68, %ymm2, %ymm9, %ymm13
	vshufps	$238, %ymm2, %ymm9, %ymm2
	vinsertf128	$1, %xmm2, %ymm13, %ymm13
	vperm2f128	$3, %ymm8, %ymm8, %ymm2
	vshufps	$68, %ymm2, %ymm8, %ymm9
	vshufps	$238, %ymm2, %ymm8, %ymm2
	vshufps	$136, %ymm6, %ymm10, %ymm8
	vinsertf128	$1, %xmm2, %ymm9, %ymm9
	vshufps	$221, %ymm6, %ymm10, %ymm6
	vperm2f128	$3, %ymm8, %ymm8, %ymm2
	vshufps	$68, %ymm2, %ymm8, %ymm12
	vshufps	$238, %ymm2, %ymm8, %ymm2
	vinsertf128	$1, %xmm2, %ymm12, %ymm12
	vperm2f128	$3, %ymm6, %ymm6, %ymm2
	vshufps	$68, %ymm2, %ymm6, %ymm8
	vshufps	$238, %ymm2, %ymm6, %ymm2
	vshufps	$136, %ymm4, %ymm7, %ymm6
	vinsertf128	$1, %xmm2, %ymm8, %ymm8
	vshufps	$221, %ymm4, %ymm7, %ymm4
	vperm2f128	$3, %ymm6, %ymm6, %ymm2
	vshufps	$68, %ymm2, %ymm6, %ymm11
	vshufps	$238, %ymm2, %ymm6, %ymm2
	vinsertf128	$1, %xmm2, %ymm11, %ymm11
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm7
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vshufps	$136, %ymm3, %ymm5, %ymm4
	vinsertf128	$1, %xmm2, %ymm7, %ymm7
	vshufps	$221, %ymm3, %ymm5, %ymm3
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm10
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vinsertf128	$1, %xmm2, %ymm10, %ymm10
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm4
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vshufps	$136, %ymm12, %ymm13, %ymm3
	vinsertf128	$1, %xmm2, %ymm4, %ymm6
	vmovaps	%ymm6, %ymm14
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm4
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm4, %ymm3
	vcmpneqps	%ymm3, %ymm15, %ymm4
	vrsqrtps	%ymm3, %ymm2
	vmovaps	%ymm14, -48(%rsp)
	vshufps	$221, %ymm12, %ymm13, %ymm12
	vandps	%ymm4, %ymm2, %ymm2
	vmulps	%ymm3, %ymm2, %ymm3
	vmulps	%ymm2, %ymm3, %ymm2
	vaddps	%ymm1, %ymm2, %ymm6
	vmulps	%ymm0, %ymm3, %ymm3
	vmulps	%ymm3, %ymm6, %ymm6
	vshufps	$136, %ymm10, %ymm11, %ymm3
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm4
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm4, %ymm3
	vcmpneqps	%ymm3, %ymm15, %ymm4
	vrsqrtps	%ymm3, %ymm2
	vshufps	$221, %ymm10, %ymm11, %ymm10
	vandps	%ymm4, %ymm2, %ymm2
	vmulps	%ymm3, %ymm2, %ymm3
	vmulps	%ymm2, %ymm3, %ymm2
	vaddps	%ymm1, %ymm2, %ymm5
	vmulps	%ymm0, %ymm3, %ymm3
	vmulps	%ymm3, %ymm5, %ymm5
	vshufps	$136, %ymm8, %ymm9, %ymm3
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm4
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm4, %ymm3
	vcmpneqps	%ymm3, %ymm15, %ymm4
	vrsqrtps	%ymm3, %ymm2
	vshufps	$221, %ymm8, %ymm9, %ymm8
	vandps	%ymm4, %ymm2, %ymm2
	vmulps	%ymm3, %ymm2, %ymm3
	vmulps	%ymm2, %ymm3, %ymm2
	vaddps	%ymm1, %ymm2, %ymm4
	vmulps	%ymm0, %ymm3, %ymm3
	vmulps	%ymm3, %ymm4, %ymm4
	vshufps	$136, %ymm14, %ymm7, %ymm3
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm14
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm14, %ymm3
	vcmpneqps	%ymm3, %ymm15, %ymm14
	vrsqrtps	%ymm3, %ymm2
	vshufps	$221, -48(%rsp), %ymm7, %ymm7
	vandps	%ymm14, %ymm2, %ymm2
	vmulps	%ymm3, %ymm2, %ymm3
	vmulps	%ymm2, %ymm3, %ymm2
	vaddps	%ymm1, %ymm2, %ymm2
	vmulps	%ymm0, %ymm3, %ymm3
	vmulps	%ymm3, %ymm2, %ymm3
	vperm2f128	$3, %ymm12, %ymm12, %ymm2
	vshufps	$68, %ymm2, %ymm12, %ymm13
	vshufps	$238, %ymm2, %ymm12, %ymm2
	vinsertf128	$1, %xmm2, %ymm13, %ymm12
	vcmpneqps	%ymm12, %ymm15, %ymm13
	vrsqrtps	%ymm12, %ymm2
	vandps	%ymm13, %ymm2, %ymm2
	vmulps	%ymm12, %ymm2, %ymm12
	vmulps	%ymm2, %ymm12, %ymm2
	vaddps	%ymm1, %ymm2, %ymm2
	vmulps	%ymm0, %ymm12, %ymm12
	vmulps	%ymm12, %ymm2, %ymm12
	vperm2f128	$3, %ymm10, %ymm10, %ymm2
	vshufps	$68, %ymm2, %ymm10, %ymm11
	vshufps	$238, %ymm2, %ymm10, %ymm2
	vinsertf128	$1, %xmm2, %ymm11, %ymm10
	vcmpneqps	%ymm10, %ymm15, %ymm11
	vrsqrtps	%ymm10, %ymm2
	vandps	%ymm11, %ymm2, %ymm2
	vmulps	%ymm10, %ymm2, %ymm10
	vmulps	%ymm2, %ymm10, %ymm2
	vaddps	%ymm1, %ymm2, %ymm2
	vmulps	%ymm0, %ymm10, %ymm10
	vmulps	%ymm10, %ymm2, %ymm10
	vperm2f128	$3, %ymm8, %ymm8, %ymm2
	vshufps	$68, %ymm2, %ymm8, %ymm9
	vshufps	$238, %ymm2, %ymm8, %ymm2
	vinsertf128	$1, %xmm2, %ymm9, %ymm8
	vcmpneqps	%ymm8, %ymm15, %ymm9
	vrsqrtps	%ymm8, %ymm2
	vunpcklps	%ymm10, %ymm5, %ymm11
	vunpckhps	%ymm10, %ymm5, %ymm5
	vandps	%ymm9, %ymm2, %ymm2
	vmulps	%ymm8, %ymm2, %ymm8
	vmulps	%ymm2, %ymm8, %ymm2
	vaddps	%ymm1, %ymm2, %ymm2
	vmulps	%ymm0, %ymm8, %ymm8
	vmulps	%ymm8, %ymm2, %ymm8
	vperm2f128	$3, %ymm7, %ymm7, %ymm2
	vshufps	$68, %ymm2, %ymm7, %ymm9
	vshufps	$238, %ymm2, %ymm7, %ymm2
	vinsertf128	$1, %xmm2, %ymm9, %ymm7
	vcmpneqps	%ymm7, %ymm15, %ymm9
	vrsqrtps	%ymm7, %ymm2
	vandps	%ymm9, %ymm2, %ymm2
	vmulps	%ymm7, %ymm2, %ymm7
	vmulps	%ymm2, %ymm7, %ymm2
	vaddps	%ymm1, %ymm2, %ymm2
	vinsertf128	$1, %xmm5, %ymm11, %ymm9
	vmulps	%ymm0, %ymm7, %ymm7
	vperm2f128	$49, %ymm5, %ymm11, %ymm11
	vunpcklps	%ymm8, %ymm4, %ymm5
	vunpckhps	%ymm8, %ymm4, %ymm4
	vinsertf128	$1, %xmm4, %ymm5, %ymm10
	vperm2f128	$49, %ymm4, %ymm5, %ymm5
	vmulps	%ymm7, %ymm2, %ymm7
	vunpcklps	%ymm12, %ymm6, %ymm2
	vunpckhps	%ymm12, %ymm6, %ymm6
	vinsertf128	$1, %xmm6, %ymm2, %ymm13
	vperm2f128	$49, %ymm6, %ymm2, %ymm6
	vunpcklps	%ymm10, %ymm13, %ymm8
	vunpcklps	%ymm7, %ymm3, %ymm2
	vunpcklps	%ymm5, %ymm6, %ymm4
	vunpckhps	%ymm7, %ymm3, %ymm3
	vinsertf128	$1, %xmm3, %ymm2, %ymm7
	vperm2f128	$49, %ymm3, %ymm2, %ymm2
	vunpckhps	%ymm10, %ymm13, %ymm3
	vinsertf128	$1, %xmm3, %ymm8, %ymm10
	vperm2f128	$49, %ymm3, %ymm8, %ymm8
	vunpckhps	%ymm5, %ymm6, %ymm3
	vunpcklps	%ymm7, %ymm9, %ymm5
	vinsertf128	$1, %xmm3, %ymm4, %ymm6
	vperm2f128	$49, %ymm3, %ymm4, %ymm4
	vunpckhps	%ymm7, %ymm9, %ymm3
	vunpcklps	%ymm2, %ymm11, %ymm9
	vinsertf128	$1, %xmm3, %ymm5, %ymm7
	vunpckhps	%ymm2, %ymm11, %ymm2
	vperm2f128	$49, %ymm3, %ymm5, %ymm5
	vinsertf128	$1, %xmm2, %ymm9, %ymm3
	vperm2f128	$49, %ymm2, %ymm9, %ymm2
	vunpcklps	%ymm7, %ymm10, %ymm9
	vunpckhps	%ymm7, %ymm10, %ymm7
	vinsertf128	$1, %xmm7, %ymm9, %ymm10
	vperm2f128	$49, %ymm7, %ymm9, %ymm7
	vmovaps	%ymm10, -256(%rdx)
	vmovaps	%ymm7, -224(%rdx)
	vunpcklps	%ymm5, %ymm8, %ymm7
	vunpckhps	%ymm5, %ymm8, %ymm5
	vinsertf128	$1, %xmm5, %ymm7, %ymm8
	vperm2f128	$49, %ymm5, %ymm7, %ymm5
	vmovaps	%ymm8, -192(%rdx)
	vmovaps	%ymm5, -160(%rdx)
	vunpcklps	%ymm3, %ymm6, %ymm5
	vunpckhps	%ymm3, %ymm6, %ymm3
	vinsertf128	$1, %xmm3, %ymm5, %ymm6
	vperm2f128	$49, %ymm3, %ymm5, %ymm3
	vmovaps	%ymm6, -128(%rdx)
	vmovaps	%ymm3, -96(%rdx)
	vunpcklps	%ymm2, %ymm4, %ymm3
	vunpckhps	%ymm2, %ymm4, %ymm2
	vinsertf128	$1, %xmm2, %ymm3, %ymm4
	vperm2f128	$49, %ymm2, %ymm3, %ymm2
	vmovaps	%ymm4, -64(%rdx)
	vmovaps	%ymm2, -32(%rdx)
	cmpq	%rcx, %rax
	jne	L9
	vzeroupper
	leave
LCFI5:
	ret
LFE226:
	.align 4,0x90
	.globl __Z8computeAv
__Z8computeAv:
LFB228:
	pushq	%rbp
LCFI6:
	leaq	_va(%rip), %rcx
	xorl	%eax, %eax
	leaq	_vb(%rip), %rdx
	movq	%rsp, %rbp
LCFI7:
	andq	$-32, %rsp
	addq	$16, %rsp
	vmovaps	LC2(%rip), %ymm3
	vmovaps	LC3(%rip), %ymm10
	vmovaps	LC4(%rip), %ymm8
	vmovaps	LC5(%rip), %ymm7
	vmovaps	LC6(%rip), %ymm6
	vmovaps	LC7(%rip), %ymm5
	vmovaps	LC8(%rip), %ymm9
	.align 4,0x90
L13:
	vmovaps	(%rcx,%rax), %ymm0
	vaddps	%ymm3, %ymm0, %ymm2
	vsubps	%ymm3, %ymm0, %ymm4
	vrcpps	%ymm2, %ymm1
	vmulps	%ymm2, %ymm1, %ymm2
	vmulps	%ymm2, %ymm1, %ymm2
	vaddps	%ymm1, %ymm1, %ymm1
	vsubps	%ymm2, %ymm1, %ymm2
	vmulps	%ymm0, %ymm0, %ymm1
	vmulps	%ymm2, %ymm4, %ymm4
	vmulps	%ymm4, %ymm4, %ymm2
	vmulps	%ymm8, %ymm2, %ymm11
	vaddps	%ymm7, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm11
	vaddps	%ymm6, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm11
	vaddps	%ymm5, %ymm11, %ymm11
	vmulps	%ymm2, %ymm11, %ymm2
	vaddps	%ymm3, %ymm2, %ymm2
	vmulps	%ymm4, %ymm2, %ymm4
	vmulps	%ymm8, %ymm1, %ymm2
	vaddps	%ymm7, %ymm2, %ymm2
	vmulps	%ymm1, %ymm2, %ymm2
	vaddps	%ymm6, %ymm2, %ymm2
	vmulps	%ymm1, %ymm2, %ymm2
	vaddps	%ymm5, %ymm2, %ymm2
	vmulps	%ymm1, %ymm2, %ymm1
	vaddps	%ymm3, %ymm1, %ymm1
	vmulps	%ymm0, %ymm1, %ymm1
	vcmpltps	%ymm0, %ymm10, %ymm0
	vaddps	%ymm9, %ymm1, %ymm1
	vblendvps	%ymm0, %ymm4, %ymm1, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$32768, %rax
	jne	L13
	vzeroupper
	leave
LCFI8:
	ret
LFE228:
	.align 4,0x90
	.globl __Z9computeALv
__Z9computeALv:
LFB229:
	vmovss	LC16(%rip), %xmm0
	pushq	%rbx
LCFI9:
	leaq	_va(%rip), %rax
	vmovaps	LC9(%rip), %xmm2
	leaq	_vb(%rip), %rcx
	vmovss	LC17(%rip), %xmm8
	leaq	32768+_va(%rip), %r10
	vmovaps	%xmm0, %xmm1
	vmovss	LC18(%rip), %xmm7
	vmovss	LC19(%rip), %xmm10
	vmovss	LC20(%rip), %xmm9
	.align 4,0x90
L16:
	movq	%rax, %rdx
	andl	$15, %edx
	shrq	$2, %rdx
	negq	%rdx
	movq	%rdx, %rsi
	andl	$3, %esi
	testl	%esi, %esi
	je	L23
	vmovss	(%rax), %xmm3
	cmpl	$1, %esi
	vmulss	%xmm3, %xmm3, %xmm5
	vaddss	%xmm0, %xmm3, %xmm4
	vsubss	%xmm0, %xmm3, %xmm6
	vdivss	%xmm4, %xmm6, %xmm6
	vmulss	%xmm8, %xmm5, %xmm4
	vaddss	%xmm7, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm4
	vaddss	%xmm10, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm4
	vmulss	%xmm6, %xmm6, %xmm11
	vaddss	%xmm9, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm5
	vmulss	%xmm8, %xmm11, %xmm4
	vaddss	%xmm0, %xmm5, %xmm5
	vaddss	%xmm7, %xmm4, %xmm4
	vmulss	%xmm3, %xmm5, %xmm5
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	LC21(%rip), %xmm5, %xmm5
	vaddss	%xmm10, %xmm4, %xmm4
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	%xmm9, %xmm4, %xmm4
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	%xmm0, %xmm4, %xmm4
	vmulss	%xmm6, %xmm4, %xmm4
	vmovss	LC22(%rip), %xmm6
	vcmpltss	%xmm3, %xmm6, %xmm3
	vandps	%xmm3, %xmm4, %xmm4
	vandnps	%xmm5, %xmm3, %xmm3
	vorps	%xmm4, %xmm3, %xmm3
	vmovss	%xmm3, (%rcx)
	je	L24
	vmovss	4(%rax), %xmm3
	cmpl	$3, %esi
	vmulss	%xmm3, %xmm3, %xmm5
	vaddss	%xmm0, %xmm3, %xmm4
	vsubss	%xmm0, %xmm3, %xmm6
	vdivss	%xmm4, %xmm6, %xmm6
	vmulss	%xmm8, %xmm5, %xmm4
	vaddss	%xmm7, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm4
	vaddss	%xmm10, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm4
	vmulss	%xmm6, %xmm6, %xmm11
	vaddss	%xmm9, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm5
	vmulss	%xmm8, %xmm11, %xmm4
	vaddss	%xmm0, %xmm5, %xmm5
	vaddss	%xmm7, %xmm4, %xmm4
	vmulss	%xmm3, %xmm5, %xmm5
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	LC21(%rip), %xmm5, %xmm5
	vaddss	%xmm10, %xmm4, %xmm4
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	%xmm9, %xmm4, %xmm4
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	%xmm0, %xmm4, %xmm4
	vmulss	%xmm6, %xmm4, %xmm4
	vmovss	LC22(%rip), %xmm6
	vcmpltss	%xmm3, %xmm6, %xmm3
	vandps	%xmm3, %xmm4, %xmm4
	vandnps	%xmm5, %xmm3, %xmm3
	vorps	%xmm4, %xmm3, %xmm3
	vmovss	%xmm3, 4(%rcx)
	jne	L25
	vmovss	8(%rax), %xmm3
	movl	$5, %ebx
	movl	$3, %r11d
	vmulss	%xmm3, %xmm3, %xmm5
	vaddss	%xmm0, %xmm3, %xmm4
	vsubss	%xmm0, %xmm3, %xmm6
	vdivss	%xmm4, %xmm6, %xmm6
	vmulss	%xmm8, %xmm5, %xmm4
	vaddss	%xmm7, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm4
	vaddss	%xmm10, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm4
	vmulss	%xmm6, %xmm6, %xmm11
	vaddss	%xmm9, %xmm4, %xmm4
	vmulss	%xmm5, %xmm4, %xmm5
	vmulss	%xmm8, %xmm11, %xmm4
	vaddss	%xmm0, %xmm5, %xmm5
	vaddss	%xmm7, %xmm4, %xmm4
	vmulss	%xmm3, %xmm5, %xmm5
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	LC21(%rip), %xmm5, %xmm5
	vaddss	%xmm10, %xmm4, %xmm4
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	%xmm9, %xmm4, %xmm4
	vmulss	%xmm11, %xmm4, %xmm4
	vaddss	%xmm0, %xmm4, %xmm4
	vmulss	%xmm6, %xmm4, %xmm4
	vmovss	LC22(%rip), %xmm6
	vcmpltss	%xmm3, %xmm6, %xmm3
	vandps	%xmm3, %xmm4, %xmm4
	vandnps	%xmm5, %xmm3, %xmm3
	vorps	%xmm4, %xmm3, %xmm3
	vmovss	%xmm3, 8(%rcx)
L19:
	andl	$3, %edx
	movl	$8, %edi
	salq	$2, %rdx
	subl	%esi, %edi
	leaq	(%rax,%rdx), %r9
	movl	%edi, %esi
	addq	%rcx, %rdx
	vmovaps	(%r9), %xmm3
	shrl	$2, %esi
	leal	0(,%rsi,4), %r8d
	cmpl	$2, %esi
	vaddps	%xmm2, %xmm3, %xmm5
	vsubps	%xmm2, %xmm3, %xmm6
	vrcpps	%xmm5, %xmm4
	vmulps	%xmm5, %xmm4, %xmm5
	vmulps	%xmm5, %xmm4, %xmm5
	vaddps	%xmm4, %xmm4, %xmm4
	vsubps	%xmm5, %xmm4, %xmm5
	vmulps	%xmm3, %xmm3, %xmm4
	vmulps	%xmm5, %xmm6, %xmm6
	vmulps	%xmm6, %xmm6, %xmm5
	vmulps	LC11(%rip), %xmm5, %xmm11
	vaddps	LC12(%rip), %xmm11, %xmm11
	vmulps	%xmm5, %xmm11, %xmm11
	vaddps	LC13(%rip), %xmm11, %xmm11
	vmulps	%xmm5, %xmm11, %xmm11
	vaddps	LC14(%rip), %xmm11, %xmm11
	vmulps	%xmm5, %xmm11, %xmm5
	vaddps	%xmm2, %xmm5, %xmm5
	vmulps	%xmm6, %xmm5, %xmm6
	vmulps	LC11(%rip), %xmm4, %xmm5
	vaddps	LC12(%rip), %xmm5, %xmm5
	vmulps	%xmm4, %xmm5, %xmm5
	vaddps	LC13(%rip), %xmm5, %xmm5
	vmulps	%xmm4, %xmm5, %xmm5
	vaddps	LC14(%rip), %xmm5, %xmm5
	vmulps	%xmm4, %xmm5, %xmm4
	vmovaps	LC10(%rip), %xmm5
	vaddps	%xmm2, %xmm4, %xmm4
	vmulps	%xmm3, %xmm4, %xmm4
	vcmpltps	%xmm3, %xmm5, %xmm3
	vaddps	LC15(%rip), %xmm4, %xmm4
	vblendvps	%xmm3, %xmm6, %xmm4, %xmm3
	vmovups	%xmm3, (%rdx)
	jne	L22
	vmovaps	16(%r9), %xmm3
	vaddps	%xmm2, %xmm3, %xmm5
	vsubps	%xmm2, %xmm3, %xmm6
	vrcpps	%xmm5, %xmm4
	vmulps	%xmm5, %xmm4, %xmm5
	vmulps	%xmm5, %xmm4, %xmm5
	vaddps	%xmm4, %xmm4, %xmm4
	vsubps	%xmm5, %xmm4, %xmm5
	vmulps	%xmm3, %xmm3, %xmm4
	vmulps	%xmm5, %xmm6, %xmm6
	vmulps	%xmm6, %xmm6, %xmm5
	vmulps	LC11(%rip), %xmm5, %xmm11
	vaddps	LC12(%rip), %xmm11, %xmm11
	vmulps	%xmm5, %xmm11, %xmm11
	vaddps	LC13(%rip), %xmm11, %xmm11
	vmulps	%xmm5, %xmm11, %xmm11
	vaddps	LC14(%rip), %xmm11, %xmm11
	vmulps	%xmm5, %xmm11, %xmm5
	vaddps	%xmm2, %xmm5, %xmm5
	vmulps	%xmm6, %xmm5, %xmm6
	vmulps	LC11(%rip), %xmm4, %xmm5
	vaddps	LC12(%rip), %xmm5, %xmm5
	vmulps	%xmm4, %xmm5, %xmm5
	vaddps	LC13(%rip), %xmm5, %xmm5
	vmulps	%xmm4, %xmm5, %xmm5
	vaddps	LC14(%rip), %xmm5, %xmm5
	vmulps	%xmm4, %xmm5, %xmm4
	vmovaps	LC10(%rip), %xmm5
	vaddps	%xmm2, %xmm4, %xmm4
	vmulps	%xmm3, %xmm4, %xmm4
	vcmpltps	%xmm3, %xmm5, %xmm3
	vaddps	LC15(%rip), %xmm4, %xmm4
	vblendvps	%xmm3, %xmm6, %xmm4, %xmm3
	vmovups	%xmm3, 16(%rdx)
L22:
	leal	(%r11,%r8), %edx
	subl	%r8d, %ebx
	cmpl	%r8d, %edi
	je	L17
	movslq	%edx, %rsi
	vmovss	LC19(%rip), %xmm4
	cmpl	$1, %ebx
	vmovss	(%rax,%rsi,4), %xmm3
	vmovss	LC20(%rip), %xmm5
	vmulss	%xmm3, %xmm3, %xmm11
	vaddss	%xmm1, %xmm3, %xmm6
	vmovss	LC21(%rip), %xmm13
	vsubss	%xmm1, %xmm3, %xmm12
	vdivss	%xmm6, %xmm12, %xmm12
	vmulss	LC17(%rip), %xmm11, %xmm6
	vaddss	LC18(%rip), %xmm6, %xmm6
	vmulss	%xmm11, %xmm6, %xmm6
	vaddss	%xmm4, %xmm6, %xmm6
	vmulss	%xmm11, %xmm6, %xmm6
	vmulss	%xmm12, %xmm12, %xmm14
	vaddss	%xmm5, %xmm6, %xmm6
	vmulss	%xmm11, %xmm6, %xmm11
	vmulss	LC17(%rip), %xmm14, %xmm6
	vaddss	%xmm1, %xmm11, %xmm11
	vaddss	LC18(%rip), %xmm6, %xmm6
	vmulss	%xmm3, %xmm11, %xmm11
	vmulss	%xmm14, %xmm6, %xmm6
	vaddss	%xmm13, %xmm11, %xmm11
	vaddss	%xmm4, %xmm6, %xmm6
	vmulss	%xmm14, %xmm6, %xmm6
	vaddss	%xmm5, %xmm6, %xmm6
	vmulss	%xmm14, %xmm6, %xmm6
	vaddss	%xmm1, %xmm6, %xmm6
	vmulss	%xmm12, %xmm6, %xmm6
	vmovss	LC22(%rip), %xmm12
	vcmpltss	%xmm3, %xmm12, %xmm3
	vandps	%xmm3, %xmm6, %xmm6
	vandnps	%xmm11, %xmm3, %xmm3
	vorps	%xmm6, %xmm3, %xmm3
	vmovss	%xmm3, (%rcx,%rsi,4)
	leal	1(%rdx), %esi
	je	L17
	movslq	%esi, %rsi
	addl	$2, %edx
	cmpl	$2, %ebx
	vmovss	(%rax,%rsi,4), %xmm6
	vmulss	%xmm6, %xmm6, %xmm15
	vaddss	%xmm1, %xmm6, %xmm11
	vsubss	%xmm1, %xmm6, %xmm14
	vdivss	%xmm11, %xmm14, %xmm14
	vmulss	LC17(%rip), %xmm15, %xmm11
	vaddss	LC18(%rip), %xmm11, %xmm11
	vmulss	%xmm15, %xmm11, %xmm11
	vaddss	%xmm4, %xmm11, %xmm11
	vmulss	%xmm15, %xmm11, %xmm11
	vmulss	%xmm14, %xmm14, %xmm3
	vaddss	%xmm5, %xmm11, %xmm11
	vmulss	%xmm15, %xmm11, %xmm15
	vmulss	LC17(%rip), %xmm3, %xmm11
	vaddss	%xmm1, %xmm15, %xmm15
	vaddss	LC18(%rip), %xmm11, %xmm11
	vmulss	%xmm6, %xmm15, %xmm15
	vcmpltss	%xmm6, %xmm12, %xmm6
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm13, %xmm15, %xmm15
	vaddss	%xmm4, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm5, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vaddss	%xmm1, %xmm11, %xmm11
	vmulss	%xmm14, %xmm11, %xmm11
	vandps	%xmm6, %xmm11, %xmm11
	vandnps	%xmm15, %xmm6, %xmm6
	vorps	%xmm11, %xmm6, %xmm6
	vmovss	%xmm6, (%rcx,%rsi,4)
	je	L17
	movslq	%edx, %rdx
	vmovss	(%rax,%rdx,4), %xmm3
	vmulss	%xmm3, %xmm3, %xmm11
	vaddss	%xmm1, %xmm3, %xmm6
	vsubss	%xmm1, %xmm3, %xmm14
	vmulss	LC17(%rip), %xmm11, %xmm15
	vdivss	%xmm6, %xmm14, %xmm14
	vaddss	LC18(%rip), %xmm15, %xmm15
	vmulss	%xmm11, %xmm15, %xmm15
	vaddss	%xmm4, %xmm15, %xmm15
	vmulss	%xmm11, %xmm15, %xmm15
	vaddss	%xmm5, %xmm15, %xmm15
	vmulss	%xmm14, %xmm14, %xmm6
	vmulss	%xmm11, %xmm15, %xmm11
	vaddss	%xmm1, %xmm11, %xmm11
	vmulss	%xmm3, %xmm11, %xmm11
	vcmpltss	%xmm3, %xmm12, %xmm3
	vaddss	%xmm13, %xmm11, %xmm13
	vmulss	LC17(%rip), %xmm6, %xmm11
	vaddss	LC18(%rip), %xmm11, %xmm11
	vmulss	%xmm6, %xmm11, %xmm11
	vaddss	%xmm4, %xmm11, %xmm4
	vmulss	%xmm6, %xmm4, %xmm4
	vaddss	%xmm5, %xmm4, %xmm4
	vmulss	%xmm6, %xmm4, %xmm4
	vaddss	%xmm1, %xmm4, %xmm4
	vmulss	%xmm14, %xmm4, %xmm4
	vandps	%xmm3, %xmm4, %xmm4
	vandnps	%xmm13, %xmm3, %xmm3
	vorps	%xmm4, %xmm3, %xmm3
	vmovss	%xmm3, (%rcx,%rdx,4)
L17:
	addq	$32, %rax
	addq	$32, %rcx
	cmpq	%r10, %rax
	jne	L16
	popq	%rbx
LCFI10:
	ret
	.align 4,0x90
L23:
LCFI11:
	movl	$8, %ebx
	xorl	%r11d, %r11d
	jmp	L19
	.align 4,0x90
L25:
	movl	$6, %ebx
	movl	$2, %r11d
	jmp	L19
	.align 4,0x90
L24:
	movl	$7, %ebx
	movl	$1, %r11d
	jmp	L19
LFE229:
	.globl _vc
	.zerofill __DATA,__pu_bss5,_vc,32768,5
	.globl _vb
	.zerofill __DATA,__pu_bss5,_vb,32768,5
	.globl _va
	.zerofill __DATA,__pu_bss5,_va,32768,5
	.const
	.align 5
LC0:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 5
LC1:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.align 5
LC2:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 5
LC3:
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.align 5
LC4:
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.align 5
LC5:
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.align 5
LC6:
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.align 5
LC7:
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.align 5
LC8:
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.literal16
	.align 4
LC9:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC10:
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.align 4
LC11:
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.align 4
LC12:
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.align 4
LC13:
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.align 4
LC14:
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.align 4
LC15:
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
	.long	1034219729
	.align 2
LC18:
	.long	3188595589
	.align 2
LC19:
	.long	1045205599
	.align 2
LC20:
	.long	3198855722
	.align 2
LC21:
	.long	1061752795
	.align 2
LC22:
	.long	1054086093
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
	.quad	LFB224-.
	.set L$set$2,LFE224-LFB224
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB224
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
	.quad	LFB225-.
	.set L$set$7,LFE225-LFB225
	.quad L$set$7
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$8,LEFDE5-LASFDE5
	.long L$set$8
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB226-.
	.set L$set$9,LFE226-LFB226
	.quad L$set$9
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI3-LFB226
	.long L$set$10
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$12,LCFI5-LCFI4
	.long L$set$12
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$13,LEFDE7-LASFDE7
	.long L$set$13
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB228-.
	.set L$set$14,LFE228-LFB228
	.quad L$set$14
	.byte	0
	.byte	0x4
	.set L$set$15,LCFI6-LFB228
	.long L$set$15
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$16,LCFI7-LCFI6
	.long L$set$16
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$17,LCFI8-LCFI7
	.long L$set$17
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$18,LEFDE9-LASFDE9
	.long L$set$18
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB229-.
	.set L$set$19,LFE229-LFB229
	.quad L$set$19
	.byte	0
	.byte	0x4
	.set L$set$20,LCFI9-LFB229
	.long L$set$20
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$21,LCFI10-LCFI9
	.long L$set$21
	.byte	0xa
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$22,LCFI11-LCFI10
	.long L$set$22
	.byte	0xb
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
