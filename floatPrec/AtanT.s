	.text
	.align 4,0x90
	.globl __Z8computeVv
__Z8computeVv:
LFB1:
	pushq	%rbp
LCFI0:
	leaq	_va(%rip), %rdi
	xorl	%eax, %eax
	leaq	_vb(%rip), %rsi
	movq	%rsp, %rbp
LCFI1:
	andq	$-32, %rsp
	subq	$472, %rsp
	movaps	LC0(%rip), %xmm2
	movss	LC1(%rip), %xmm0
	jmp	L27
	.align 4,0x90
L39:
	movss	440(%rsp), %xmm15
	comiss	%xmm14, %xmm0
	jb	L29
L40:
	movss	444(%rsp), %xmm14
	comiss	%xmm13, %xmm0
	jb	L30
L41:
	movss	448(%rsp), %xmm13
	comiss	%xmm12, %xmm0
	jb	L31
L42:
	movss	452(%rsp), %xmm12
L12:
	movl	$0x3f490fdb, %r10d
	comiss	-120(%rsp), %xmm0
	jb	L15
	movl	456(%rsp), %r10d
L15:
	comiss	%xmm11, %xmm0
	jb	L33
	movss	460(%rsp), %xmm11
L18:
	comiss	%xmm10, %xmm0
	jb	L34
	movss	464(%rsp), %xmm10
L21:
	comiss	%xmm9, %xmm0
	jb	L35
	movss	468(%rsp), %xmm9
L24:
	movd	%ecx, %xmm3
	movd	%r8d, %xmm1
	unpcklps	%xmm6, %xmm7
	unpcklps	%xmm3, %xmm1
	movd	%edx, %xmm3
	movaps	%xmm5, %xmm6
	unpcklps	%xmm8, %xmm3
	movaps	%xmm3, %xmm8
	movlhps	%xmm1, %xmm8
	movaps	%xmm8, %xmm3
	unpcklps	%xmm4, %xmm6
	movaps	%xmm6, %xmm4
	movlhps	%xmm7, %xmm4
	unpcklps	%xmm12, %xmm13
	unpcklps	%xmm14, %xmm15
	mulps	%xmm8, %xmm3
	movaps	%xmm4, %xmm1
	movaps	LC3(%rip), %xmm6
	mulps	%xmm4, %xmm1
	movaps	LC3(%rip), %xmm5
	movaps	%xmm15, %xmm7
	movlhps	%xmm13, %xmm7
	unpcklps	%xmm9, %xmm10
	mulps	%xmm3, %xmm6
	mulps	%xmm1, %xmm5
	subps	LC4(%rip), %xmm6
	subps	LC4(%rip), %xmm5
	mulps	%xmm3, %xmm6
	mulps	%xmm1, %xmm5
	addps	LC5(%rip), %xmm6
	addps	LC5(%rip), %xmm5
	mulps	%xmm3, %xmm6
	mulps	%xmm1, %xmm5
	subps	LC6(%rip), %xmm6
	subps	LC6(%rip), %xmm5
	mulps	%xmm6, %xmm3
	mulps	%xmm5, %xmm1
	addps	%xmm2, %xmm3
	addps	%xmm2, %xmm1
	mulps	%xmm3, %xmm8
	mulps	%xmm1, %xmm4
	movd	%r10d, %xmm1
	unpcklps	%xmm11, %xmm1
	addps	%xmm7, %xmm8
	movaps	%xmm1, %xmm7
	movlhps	%xmm10, %xmm7
	addps	%xmm7, %xmm4
	movaps	%xmm8, -120(%rsp)
	movaps	%xmm8, -104(%rsp)
	movq	-120(%rsp), %r9
	movaps	%xmm4, -88(%rsp)
	movq	-96(%rsp), %r8
	movaps	%xmm4, -120(%rsp)
	movq	-80(%rsp), %rdx
	movq	-120(%rsp), %rcx
	movq	%r9, (%rsi,%rax)
	movq	%r8, 8(%rsi,%rax)
	movq	%rdx, 24(%rsi,%rax)
	movq	%rcx, 16(%rsi,%rax)
	addq	$32, %rax
	cmpq	$32768, %rax
	movq	%r9, 440(%rsp)
	movq	%r8, 448(%rsp)
	movq	%rcx, 456(%rsp)
	movq	%rdx, 464(%rsp)
	je	L38
L27:
	movq	(%rdi,%rax), %rdx
	movaps	%xmm0, %xmm8
	movq	%rdx, -72(%rsp)
	movq	8(%rdi,%rax), %rdx
	movss	-72(%rsp), %xmm15
	movss	-68(%rsp), %xmm14
	comiss	%xmm15, %xmm0
	movq	%rdx, -64(%rsp)
	cmpltss	%xmm14, %xmm8
	movaps	-72(%rsp), %xmm1
	movq	16(%rdi,%rax), %rdx
	movaps	%xmm1, %xmm6
	addps	%xmm2, %xmm1
	movss	-64(%rsp), %xmm13
	subps	%xmm2, %xmm6
	movss	-60(%rsp), %xmm12
	rcpps	%xmm1, %xmm3
	movq	%rdx, -56(%rsp)
	movq	24(%rdi,%rax), %rdx
	movss	-52(%rsp), %xmm11
	mulps	%xmm3, %xmm1
	movq	%rdx, -48(%rsp)
	movaps	-56(%rsp), %xmm4
	movss	-48(%rsp), %xmm10
	mulps	%xmm3, %xmm1
	movaps	%xmm4, %xmm5
	movss	-44(%rsp), %xmm9
	addps	%xmm2, %xmm4
	addps	%xmm3, %xmm3
	subps	%xmm2, %xmm5
	subps	%xmm1, %xmm3
	rcpps	%xmm4, %xmm1
	mulps	%xmm6, %xmm3
	mulps	%xmm1, %xmm4
	mulps	%xmm1, %xmm4
	addps	%xmm1, %xmm1
	subps	%xmm4, %xmm1
	movaps	%xmm3, %xmm4
	mulps	%xmm5, %xmm1
	movaps	%xmm0, %xmm5
	cmpltss	%xmm15, %xmm5
	andps	%xmm5, %xmm4
	andnps	%xmm15, %xmm5
	movaps	%xmm5, %xmm7
	movaps	%xmm0, %xmm5
	orps	%xmm4, %xmm7
	movd	%xmm7, %edx
	movl	%edx, -104(%rsp)
	movaps	%xmm3, %xmm7
	cmpltss	%xmm13, %xmm5
	shufps	$85, %xmm3, %xmm7
	movaps	%xmm7, %xmm4
	movaps	%xmm3, %xmm7
	andps	%xmm8, %xmm4
	unpckhps	%xmm3, %xmm7
	andnps	%xmm14, %xmm8
	orps	%xmm4, %xmm8
	movaps	%xmm7, %xmm4
	shufps	$255, %xmm3, %xmm3
	andps	%xmm5, %xmm4
	andnps	%xmm13, %xmm5
	movaps	%xmm5, %xmm7
	orps	%xmm4, %xmm7
	movaps	%xmm0, %xmm4
	movd	%xmm7, %r8d
	cmpltss	%xmm12, %xmm4
	movss	-56(%rsp), %xmm7
	movaps	%xmm0, %xmm5
	movl	%r8d, -96(%rsp)
	movss	%xmm8, -100(%rsp)
	cmpltss	%xmm7, %xmm5
	movss	%xmm7, -120(%rsp)
	andps	%xmm4, %xmm3
	andnps	%xmm12, %xmm4
	movaps	%xmm4, %xmm6
	movaps	%xmm0, %xmm4
	orps	%xmm3, %xmm6
	movaps	%xmm1, %xmm3
	cmpltss	%xmm11, %xmm4
	movd	%xmm6, %ecx
	movaps	%xmm1, %xmm6
	andps	%xmm5, %xmm3
	shufps	$85, %xmm1, %xmm6
	andnps	%xmm7, %xmm5
	orps	%xmm3, %xmm5
	movaps	%xmm6, %xmm3
	movaps	%xmm1, %xmm6
	andps	%xmm4, %xmm3
	unpckhps	%xmm1, %xmm6
	andnps	%xmm11, %xmm4
	movaps	%xmm0, %xmm7
	orps	%xmm3, %xmm4
	movaps	%xmm6, %xmm3
	shufps	$255, %xmm1, %xmm1
	movl	%ecx, -92(%rsp)
	movaps	%xmm0, %xmm6
	cmpltss	%xmm10, %xmm7
	movss	%xmm5, -88(%rsp)
	cmpltss	%xmm9, %xmm6
	movss	%xmm4, -84(%rsp)
	andps	%xmm7, %xmm3
	andnps	%xmm10, %xmm7
	orps	%xmm3, %xmm7
	andps	%xmm6, %xmm1
	movss	%xmm7, -80(%rsp)
	andnps	%xmm9, %xmm6
	orps	%xmm1, %xmm6
	movss	%xmm6, -76(%rsp)
	jae	L39
	movss	LC2(%rip), %xmm15
	comiss	%xmm14, %xmm0
	jae	L40
L29:
	movss	LC2(%rip), %xmm14
	comiss	%xmm13, %xmm0
	jae	L41
L30:
	movss	LC2(%rip), %xmm13
	comiss	%xmm12, %xmm0
	jae	L42
L31:
	movss	LC2(%rip), %xmm12
	jmp	L12
	.align 4,0x90
L35:
	movss	LC2(%rip), %xmm9
	jmp	L24
	.align 4,0x90
L34:
	movss	LC2(%rip), %xmm10
	jmp	L21
	.align 4,0x90
L33:
	movss	LC2(%rip), %xmm11
	jmp	L18
	.align 4,0x90
L38:
	leave
LCFI2:
	ret
LFE1:
	.align 4,0x90
	.globl __Z8computeLv
__Z8computeLv:
LFB2:
	movaps	LC0(%rip), %xmm4
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	movaps	LC7(%rip), %xmm10
	leaq	_b(%rip), %rdx
	movaps	LC3(%rip), %xmm9
	movaps	LC8(%rip), %xmm8
	movaps	LC5(%rip), %xmm7
	movaps	LC9(%rip), %xmm6
	movaps	LC10(%rip), %xmm5
	.align 4,0x90
L45:
	movaps	(%rcx,%rax), %xmm1
	movaps	%xmm1, %xmm3
	movaps	%xmm1, %xmm2
	addps	%xmm4, %xmm3
	subps	%xmm4, %xmm2
	rcpps	%xmm3, %xmm0
	mulps	%xmm0, %xmm3
	mulps	%xmm0, %xmm3
	addps	%xmm0, %xmm0
	subps	%xmm3, %xmm0
	mulps	%xmm0, %xmm2
	movaps	%xmm10, %xmm0
	cmpltps	%xmm1, %xmm0
	blendvps	%xmm0, %xmm2, %xmm1
	movaps	%xmm1, %xmm2
	mulps	%xmm1, %xmm2
	movaps	%xmm2, %xmm0
	mulps	%xmm9, %xmm0
	addps	%xmm8, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm7, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm6, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm4, %xmm0
	mulps	%xmm1, %xmm0
	addps	%xmm5, %xmm0
	movaps	%xmm0, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$32768, %rax
	jne	L45
	rep; ret
LFE2:
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,32768,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,32768,5
	.globl _vb
	.zerofill __DATA,__pu_bss5,_vb,32768,5
	.globl _va
	.zerofill __DATA,__pu_bss5,_va,32768,5
	.literal16
	.align 4
LC0:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.literal4
	.align 2
LC1:
	.long	1054086093
	.align 2
LC2:
	.long	1061752795
	.literal16
	.align 4
LC3:
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.long	1034219729
	.align 4
LC4:
	.long	1041111941
	.long	1041111941
	.long	1041111941
	.long	1041111941
	.align 4
LC5:
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.long	1045205599
	.align 4
LC6:
	.long	1051372074
	.long	1051372074
	.long	1051372074
	.long	1051372074
	.align 4
LC7:
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.long	1054086093
	.align 4
LC8:
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.long	3188595589
	.align 4
LC9:
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.long	3198855722
	.align 4
LC10:
	.long	1061752795
	.long	1061752795
	.long	1061752795
	.long	1061752795
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
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB1
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
	.quad	LFB2-.
	.set L$set$7,LFE2-LFB2
	.quad L$set$7
	.byte	0
	.align 3
LEFDE3:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
