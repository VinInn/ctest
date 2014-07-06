	.text
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB3789:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	vmovaps	LC0(%rip), %ymm3
	xorl	%eax, %eax
	pushq	-8(%r10)
	pushq	%rbp
	leaq	_a(%rip), %rcx
	vmovaps	LC1(%rip), %ymm2
LCFI1:
	movq	%rsp, %rbp
	vmovdqa	LC2(%rip), %ymm1
	pushq	%r10
LCFI2:
	leaq	_b(%rip), %rdx
	.align 4,0x90
L2:
	vmovaps	%ymm3, %ymm0
	vfmadd132ps	(%rcx,%rax), %ymm2, %ymm0
	vroundps	$1, %ymm0, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpaddd	%ymm1, %ymm0, %ymm0
	vpslld	$23, %ymm0, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L2
	vzeroupper
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
LFE3789:
	.align 4,0x90
	.globl __Z3fooPKfPfi
__Z3fooPKfPfi:
LFB3790:
	leaq	8(%rsp), %r10
LCFI5:
	andq	$-32, %rsp
	testl	%edx, %edx
	pushq	-8(%r10)
	pushq	%rbp
LCFI6:
	movq	%rsp, %rbp
	pushq	%r10
LCFI7:
	jle	L17
	movl	%edx, %ecx
	shrl	$3, %ecx
	movl	%ecx, %eax
	sall	$3, %eax
	je	L14
	cmpl	$7, %edx
	jbe	L14
	vmovaps	LC0(%rip), %ymm2
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	vmovaps	LC1(%rip), %ymm1
	vmovdqa	LC2(%rip), %ymm0
L13:
	vmovaps	%ymm2, %ymm3
	vfmadd132ps	(%rdi,%r8), %ymm1, %ymm3
	vroundps	$1, %ymm3, %ymm3
	vcvttps2dq	%ymm3, %ymm3
	vpaddd	%ymm0, %ymm3, %ymm3
	addl	$1, %r9d
	vpslld	$23, %ymm3, %ymm3
	vmovaps	%ymm3, (%rsi,%r8)
	addq	$32, %r8
	cmpl	%r9d, %ecx
	ja	L13
	cmpl	%edx, %eax
	je	L19
	vzeroupper
L8:
	vmovss	LC3(%rip), %xmm1
	movslq	%eax, %r8
	vmovss	LC4(%rip), %xmm0
	vmovaps	%xmm1, %xmm2
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	addl	$127, %ecx
	sall	$23, %ecx
	movl	%ecx, (%rsi,%r8,4)
	leal	1(%rax), %r8d
	cmpl	%r8d, %edx
	jle	L17
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm2
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	addl	$127, %ecx
	sall	$23, %ecx
	movl	%ecx, (%rsi,%r8,4)
	leal	2(%rax), %r8d
	cmpl	%r8d, %edx
	jle	L17
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm2
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	addl	$127, %ecx
	sall	$23, %ecx
	movl	%ecx, (%rsi,%r8,4)
	leal	3(%rax), %r8d
	cmpl	%r8d, %edx
	jle	L17
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm2
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	addl	$127, %ecx
	sall	$23, %ecx
	movl	%ecx, (%rsi,%r8,4)
	leal	4(%rax), %r8d
	cmpl	%r8d, %edx
	jle	L17
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm2
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	addl	$127, %ecx
	sall	$23, %ecx
	movl	%ecx, (%rsi,%r8,4)
	leal	5(%rax), %r8d
	cmpl	%r8d, %edx
	jle	L17
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm2
	addl	$6, %eax
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%eax, %edx
	movl	%ecx, (%rsi,%r8,4)
	jle	L17
	cltq
	vfmadd132ss	(%rdi,%rax,4), %xmm0, %xmm1
	vroundss	$1, %xmm1, %xmm1, %xmm1
	vcvttss2si	%xmm1, %edx
	addl	$127, %edx
	sall	$23, %edx
	movl	%edx, (%rsi,%rax,4)
L17:
	popq	%r10
LCFI8:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI9:
	ret
	.align 4,0x90
L19:
LCFI10:
	vzeroupper
	popq	%r10
LCFI11:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI12:
	ret
	.align 4,0x90
L14:
LCFI13:
	xorl	%eax, %eax
	jmp	L8
LFE3790:
	.align 4,0x90
	.globl __Z3sumPKfPfi
__Z3sumPKfPfi:
LFB3791:
	leaq	8(%rsp), %r10
LCFI14:
	andq	$-32, %rsp
	testl	%edx, %edx
	pushq	-8(%r10)
	pushq	%rbp
LCFI15:
	movq	%rsp, %rbp
	pushq	%r10
LCFI16:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	movq	$0, -24(%rbp)
	jle	L21
	movl	%edx, %ecx
	shrl	$3, %ecx
	movl	%ecx, %eax
	sall	$3, %eax
	je	L28
	cmpl	$7, %edx
	jbe	L28
	vmovaps	-48(%rbp), %ymm0
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	vmovaps	LC0(%rip), %ymm3
	vmovaps	LC1(%rip), %ymm2
	vmovdqa	LC2(%rip), %ymm1
L27:
	vaddps	(%rsi,%r8), %ymm0, %ymm0
	vmovaps	%ymm3, %ymm4
	addl	$1, %r9d
	vfmadd132ps	(%rdi,%r8), %ymm2, %ymm4
	vroundps	$1, %ymm4, %ymm4
	vcvttps2dq	%ymm4, %ymm4
	vpaddd	%ymm1, %ymm4, %ymm4
	addq	$32, %r8
	vpslld	$23, %ymm4, %ymm4
	cmpl	%r9d, %ecx
	vaddps	%ymm4, %ymm0, %ymm0
	ja	L27
	cmpl	%edx, %eax
	vmovaps	%ymm0, -48(%rbp)
	je	L21
L22:
	vmovss	LC3(%rip), %xmm1
	movslq	%eax, %r8
	vmovss	LC4(%rip), %xmm0
	vmovaps	%xmm1, %xmm3
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm3
	vroundss	$1, %xmm3, %xmm3, %xmm3
	vcvttss2si	%xmm3, %ecx
	vmovss	-48(%rbp), %xmm2
	vaddss	(%rsi,%r8,4), %xmm2, %xmm2
	leal	1(%rax), %r8d
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%r8d, %edx
	vmovd	%ecx, %xmm6
	vaddss	%xmm6, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
	jle	L21
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm3
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm3
	vroundss	$1, %xmm3, %xmm3, %xmm3
	vcvttss2si	%xmm3, %ecx
	vaddss	(%rsi,%r8,4), %xmm2, %xmm2
	leal	2(%rax), %r8d
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%r8d, %edx
	vmovd	%ecx, %xmm7
	vaddss	%xmm7, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
	jle	L21
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm3
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm3
	vroundss	$1, %xmm3, %xmm3, %xmm3
	vcvttss2si	%xmm3, %ecx
	vaddss	(%rsi,%r8,4), %xmm2, %xmm2
	leal	3(%rax), %r8d
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%r8d, %edx
	vmovd	%ecx, %xmm5
	vaddss	%xmm5, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
	jle	L21
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm3
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm3
	vroundss	$1, %xmm3, %xmm3, %xmm3
	vcvttss2si	%xmm3, %ecx
	vaddss	(%rsi,%r8,4), %xmm2, %xmm2
	leal	4(%rax), %r8d
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%r8d, %edx
	vmovd	%ecx, %xmm6
	vaddss	%xmm6, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
	jle	L21
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm3
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm3
	vroundss	$1, %xmm3, %xmm3, %xmm3
	vcvttss2si	%xmm3, %ecx
	vaddss	(%rsi,%r8,4), %xmm2, %xmm2
	leal	5(%rax), %r8d
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%r8d, %edx
	vmovd	%ecx, %xmm7
	vaddss	%xmm7, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
	jle	L21
	movslq	%r8d, %r8
	vmovaps	%xmm1, %xmm3
	addl	$6, %eax
	vfmadd132ss	(%rdi,%r8,4), %xmm0, %xmm3
	vroundss	$1, %xmm3, %xmm3, %xmm3
	vcvttss2si	%xmm3, %ecx
	vaddss	(%rsi,%r8,4), %xmm2, %xmm2
	addl	$127, %ecx
	sall	$23, %ecx
	cmpl	%eax, %edx
	vmovd	%ecx, %xmm5
	vaddss	%xmm5, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
	jle	L21
	cltq
	vfmadd132ss	(%rdi,%rax,4), %xmm0, %xmm1
	vaddss	(%rsi,%rax,4), %xmm2, %xmm2
	vroundss	$1, %xmm1, %xmm1, %xmm1
	vcvttss2si	%xmm1, %eax
	addl	$127, %eax
	sall	$23, %eax
	vmovd	%eax, %xmm5
	vaddss	%xmm5, %xmm2, %xmm2
	vmovss	%xmm2, -48(%rbp)
L21:
	vmovaps	-48(%rbp), %ymm0
	vhaddps	%ymm0, %ymm0, %ymm0
	vhaddps	%ymm0, %ymm0, %ymm1
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vaddps	%ymm1, %ymm0, %ymm0
	vzeroupper
	popq	%r10
LCFI17:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI18:
	ret
	.align 4,0x90
L28:
LCFI19:
	xorl	%eax, %eax
	jmp	L22
LFE3791:
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,4096,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,4096,5
	.const
	.align 5
LC0:
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.align 5
LC1:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.align 5
LC2:
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.literal4
	.align 2
LC3:
	.long	1069066811
	.align 2
LC4:
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
	.quad	LFB3789-.
	.set L$set$2,LFE3789-LFB3789
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB3789
	.long L$set$3
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$8,LEFDE3-LASFDE3
	.long L$set$8
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB3790-.
	.set L$set$9,LFE3790-LFB3790
	.quad L$set$9
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI5-LFB3790
	.long L$set$10
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI6-LCFI5
	.long L$set$11
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$12,LCFI7-LCFI6
	.long L$set$12
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$13,LCFI8-LCFI7
	.long L$set$13
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$14,LCFI9-LCFI8
	.long L$set$14
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$15,LCFI10-LCFI9
	.long L$set$15
	.byte	0xb
	.byte	0x4
	.set L$set$16,LCFI11-LCFI10
	.long L$set$16
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$17,LCFI12-LCFI11
	.long L$set$17
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$18,LCFI13-LCFI12
	.long L$set$18
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$19,LEFDE5-LASFDE5
	.long L$set$19
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3791-.
	.set L$set$20,LFE3791-LFB3791
	.quad L$set$20
	.byte	0
	.byte	0x4
	.set L$set$21,LCFI14-LFB3791
	.long L$set$21
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$22,LCFI15-LCFI14
	.long L$set$22
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$23,LCFI16-LCFI15
	.long L$set$23
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$24,LCFI17-LCFI16
	.long L$set$24
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$25,LCFI18-LCFI17
	.long L$set$25
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$26,LCFI19-LCFI18
	.long L$set$26
	.byte	0xb
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
