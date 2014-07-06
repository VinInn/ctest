	.text
	.align 4,0x90
	.globl __Z3foov
__Z3foov:
LFB13:
	movdqa	LC0(%rip), %xmm7
	xorl	%eax, %eax
	movdqa	LC1(%rip), %xmm6
	leaq	_x(%rip), %rsi
	movdqa	LC2(%rip), %xmm5
	leaq	_e(%rip), %rcx
	movdqa	LC3(%rip), %xmm4
	leaq	_m(%rip), %rdx
	movdqa	LC4(%rip), %xmm3
	.align 4,0x90
L2:
	movdqa	(%rsi,%rax), %xmm2
	movdqa	%xmm2, %xmm0
	movdqa	%xmm2, %xmm1
	pand	%xmm4, %xmm2
	psrad	$23, %xmm0
	psrad	$22, %xmm1
	pand	%xmm6, %xmm0
	paddd	%xmm5, %xmm0
	pand	%xmm7, %xmm1
	paddd	%xmm1, %xmm0
	pslld	$23, %xmm1
	movdqa	%xmm0, (%rcx,%rax)
	movdqa	%xmm2, %xmm0
	por	%xmm3, %xmm0
	psubd	%xmm1, %xmm0
	movaps	%xmm0, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4816, %rax
	jne	L2
	rep; ret
LFE13:
	.align 4,0x90
	.globl __Z4sumiv
__Z4sumiv:
LFB14:
	movaps	LC5(%rip), %xmm3
	leaq	_x(%rip), %rax
	pxor	%xmm4, %xmm4
	movdqa	LC0(%rip), %xmm9
	leaq	4816+_x(%rip), %rdx
	movdqa	LC1(%rip), %xmm8
	movdqa	LC2(%rip), %xmm7
	movdqa	LC3(%rip), %xmm6
	movdqa	LC4(%rip), %xmm5
	.align 4,0x90
L6:
	movdqa	(%rax), %xmm2
	addq	$16, %rax
	cmpq	%rdx, %rax
	movdqa	%xmm2, %xmm0
	movdqa	%xmm2, %xmm1
	pand	%xmm6, %xmm2
	psrad	$23, %xmm0
	psrad	$22, %xmm1
	pand	%xmm8, %xmm0
	paddd	%xmm7, %xmm0
	pand	%xmm9, %xmm1
	paddd	%xmm1, %xmm0
	pslld	$23, %xmm1
	paddd	%xmm0, %xmm4
	movdqa	%xmm2, %xmm0
	por	%xmm5, %xmm0
	psubd	%xmm1, %xmm0
	mulps	%xmm0, %xmm3
	jne	L6
	movaps	%xmm3, %xmm6
	movaps	%xmm3, %xmm7
	movaps	%xmm3, %xmm5
	unpckhps	%xmm3, %xmm6
	shufps	$255, %xmm3, %xmm7
	movaps	%xmm6, %xmm0
	movaps	%xmm7, %xmm1
	shufps	$85, %xmm3, %xmm5
	movaps	%xmm5, %xmm2
	mulss	%xmm1, %xmm0
	movdqa	%xmm4, %xmm5
	psrldq	$8, %xmm5
	paddd	%xmm5, %xmm4
	movdqa	%xmm4, %xmm6
	mulss	%xmm2, %xmm0
	psrldq	$4, %xmm6
	paddd	%xmm6, %xmm4
	mulss	%xmm3, %xmm0
	movd	%xmm0, %eax
	movd	%xmm0, %edx
	sarl	$23, %edx
	sarl	$22, %eax
	andl	$1, %eax
	movzbl	%dl, %edx
	leal	-127(%rdx,%rax), %edx
	movd	%xmm4, %eax
	addl	%edx, %eax
	ret
LFE14:
	.align 4,0x90
	.globl __Z4sumav
__Z4sumav:
LFB15:
	movss	LC6(%rip), %xmm0
	xorl	%edi, %edi
	xorl	%esi, %esi
	leaq	_x(%rip), %rcx
	leaq	4816+_x(%rip), %r9
	.align 4,0x90
L9:
	movl	(%rcx), %eax
	addq	$4, %rcx
	movl	%eax, %edx
	movl	%eax, %r8d
	andl	$8388607, %eax
	sarl	$22, %edx
	sarl	$23, %r8d
	orl	$1065353216, %eax
	andl	$1, %edx
	movzbl	%r8b, %r8d
	leal	-127(%rdx,%r8), %r8d
	sall	$23, %edx
	subl	%edx, %eax
	addl	%r8d, %esi
	movd	%eax, %xmm2
	mulss	%xmm2, %xmm0
	movd	%xmm0, %eax
	movd	%xmm0, %edx
	sarl	$23, %edx
	sarl	$22, %eax
	andl	$1, %eax
	movzbl	%dl, %edx
	leal	-127(%rdx,%rax), %eax
	addl	%eax, %edi
	cmpq	%r9, %rcx
	jne	L9
	addl	%edi, %esi
	cvtsi2ss	%esi, %xmm1
	addss	%xmm0, %xmm1
	cvttss2si	%xmm1, %eax
	ret
LFE15:
	.globl _m
	.zerofill __DATA,__pu_bss5,_m,4096,5
	.globl _e
	.zerofill __DATA,__pu_bss5,_e,4096,5
	.globl _x
	.zerofill __DATA,__pu_bss5,_x,4096,5
	.literal16
	.align 4
LC0:
	.long	1
	.long	1
	.long	1
	.long	1
	.align 4
LC1:
	.long	255
	.long	255
	.long	255
	.long	255
	.align 4
LC2:
	.long	-127
	.long	-127
	.long	-127
	.long	-127
	.align 4
LC3:
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.align 4
LC4:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC5:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.literal4
	.align 2
LC6:
	.long	1065353216
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
	.quad	LFB13-.
	.set L$set$2,LFE13-LFB13
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB14-.
	.set L$set$4,LFE14-LFB14
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB15-.
	.set L$set$6,LFE15-LFB15
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
