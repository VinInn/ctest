	.text
	.align 4,0x90
	.globl __Z4initjj
__Z4initjj:
LFB704:
	pushq	%rbp
LCFI0:
	movl	%edi, %eax
	movl	%esi, %ebp
	pushq	%rbx
LCFI1:
	movl	%edi, %ebx
	leaq	(%rax,%rax,2), %rdi
	subq	$8, %rsp
LCFI2:
	salq	$3, %rdi
	call	_malloc
	testl	%ebx, %ebx
	je	L7
	subl	$1, %ebx
	movl	%ebp, %esi
	movq	%rax, %rdx
	leaq	(%rbx,%rbx,2), %rcx
	leaq	24(%rax,%rcx,8), %rcx
	.align 4,0x90
L3:
	movl	$-3535, %edi
	movl	$0x42c80000, (%rdx)
	addq	$24, %rdx
	movl	$0x42c80000, -20(%rdx)
	movl	$0x6aa56fa6, -16(%rdx)
	movl	$0x461c4000, -12(%rdx)
	movw	%si, -8(%rdx)
	movw	%si, -6(%rdx)
	movw	%di, -4(%rdx)
	movb	$0, -2(%rdx)
	cmpq	%rcx, %rdx
	jne	L3
L7:
	addq	$8, %rsp
LCFI3:
	popq	%rbx
LCFI4:
	popq	%rbp
LCFI5:
	ret
LFE704:
	.align 4,0x90
	.globl __Z5initVjj
__Z5initVjj:
LFB706:
	pushq	%r13
LCFI6:
	pushq	%r12
LCFI7:
	movq	%rdi, %r12
	pushq	%rbp
LCFI8:
	pushq	%rbx
LCFI9:
	movl	%esi, %ebx
	subq	$8, %rsp
LCFI10:
	testq	%rbx, %rbx
	movq	$0, (%rdi)
	movq	$0, 8(%rdi)
	movq	$0, 16(%rdi)
	jne	L23
	movq	$0, 16(%rdi)
	xorl	%eax, %eax
L13:
	movq	%rax, 8(%r12)
	addq	$8, %rsp
LCFI11:
	movq	%r12, %rax
	popq	%rbx
LCFI12:
	popq	%rbp
LCFI13:
	popq	%r12
LCFI14:
	popq	%r13
LCFI15:
	ret
	.align 4,0x90
L23:
LCFI16:
	leaq	(%rbx,%rbx,2), %r13
	movl	%edx, %ebp
	salq	$3, %r13
	movq	%r13, %rdi
	call	__Znwm
	movq	%rax, %rcx
	movq	%rax, (%r12)
	movq	%rax, 8(%r12)
	leaq	(%rax,%r13), %rax
	movq	%rax, 16(%r12)
	.align 4,0x90
L14:
	testq	%rcx, %rcx
	je	L12
	movl	$-3535, %r8d
	movl	$0x42c80000, (%rcx)
	movl	$0x42c80000, 4(%rcx)
	movl	$0x6aa56fa6, 8(%rcx)
	movl	$0x461c4000, 12(%rcx)
	movw	%bp, 16(%rcx)
	movw	%bp, 18(%rcx)
	movw	%r8w, 20(%rcx)
	movb	$0, 22(%rcx)
L12:
	addq	$24, %rcx
	subq	$1, %rbx
	jne	L14
	jmp	L13
LFE706:
	.align 4,0x90
	.globl __Z4copyR9OTiledJetRKS_
__Z4copyR9OTiledJetRKS_:
LFB707:
	movq	(%rsi), %rax
	movq	%rax, (%rdi)
	movq	8(%rsi), %rax
	movq	%rax, 8(%rdi)
	movl	16(%rsi), %eax
	movl	%eax, 16(%rdi)
	movzwl	20(%rsi), %eax
	movw	%ax, 20(%rdi)
	movzbl	22(%rsi), %eax
	movb	%al, 22(%rdi)
	ret
LFE707:
	.align 4,0x90
	.globl __Z4copyPhPKhi
__Z4copyPhPKhi:
LFB708:
	testl	%edx, %edx
	je	L27
	movl	%edx, %edx
	jmp	_memcpy
	.align 4,0x90
L27:
	rep; ret
LFE708:
	.align 4,0x90
	.globl __Z5minitjj
__Z5minitjj:
LFB709:
	pushq	%r14
LCFI17:
	movl	%edi, %edi
	pushq	%r13
LCFI18:
	leaq	(%rdi,%rdi,2), %r13
	pushq	%r12
LCFI19:
	leaq	0(,%r13,8), %r12
	pushq	%rbp
LCFI20:
	movq	%r12, %rdi
	pushq	%rbx
LCFI21:
	movl	%esi, %ebx
	call	_malloc
	movl	$-3535, %r9d
	movq	%rax, %rbp
	movq	%r13, %rax
	movabsq	$-6148914691236517205, %r13
	leaq	24(%rbp), %rcx
	salq	$4, %rax
	movw	%bx, 16(%rbp)
	addq	%r12, %rax
	movw	%bx, 18(%rbp)
	movl	$1, %ebx
	leaq	0(%rbp,%rax,8), %r14
	movl	$0x42c80000, 0(%rbp)
	imulq	%r13, %rax
	movl	$0x42c80000, 4(%rbp)
	movl	$0x6aa56fa6, 8(%rbp)
	movl	$0x461c4000, 12(%rbp)
	movw	%r9w, 20(%rbp)
	movb	$0, 22(%rbp)
	sarq	%rax
	leaq	(%rax,%rax,2), %rax
	leaq	0(%rbp,%rax,8), %r12
	cmpq	%rcx, %r12
	jbe	L29
	.align 4,0x90
L30:
	movq	%rbx, %rdx
	movq	%rcx, %rdi
	movq	%rbp, %rsi
	call	_memcpy
	movq	%rax, %rcx
	leaq	(%rbx,%rbx,2), %rax
	leaq	(%rcx,%rax,8), %rcx
	movq	%rcx, %rbx
	subq	%rbp, %rbx
	sarq	$3, %rbx
	imulq	%r13, %rbx
	cmpq	%rcx, %r12
	ja	L30
L29:
	movq	%r14, %rdx
	movq	%rbp, %rsi
	movq	%rcx, %rdi
	movabsq	$-6148914691236517205, %rax
	subq	%rcx, %rdx
	sarq	$3, %rdx
	imulq	%rax, %rdx
	call	_memcpy
	popq	%rbx
LCFI22:
	movq	%rbp, %rax
	popq	%rbp
LCFI23:
	popq	%r12
LCFI24:
	popq	%r13
LCFI25:
	popq	%r14
LCFI26:
	ret
LFE709:
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
	.quad	LFB704-.
	.set L$set$2,LFE704-LFB704
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB704
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xe
	.byte	0x18
	.byte	0x83
	.byte	0x3
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$8,LCFI5-LCFI4
	.long L$set$8
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$9,LEFDE3-LASFDE3
	.long L$set$9
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB706-.
	.set L$set$10,LFE706-LFB706
	.quad L$set$10
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI6-LFB706
	.long L$set$11
	.byte	0xe
	.byte	0x10
	.byte	0x8d
	.byte	0x2
	.byte	0x4
	.set L$set$12,LCFI7-LCFI6
	.long L$set$12
	.byte	0xe
	.byte	0x18
	.byte	0x8c
	.byte	0x3
	.byte	0x4
	.set L$set$13,LCFI8-LCFI7
	.long L$set$13
	.byte	0xe
	.byte	0x20
	.byte	0x86
	.byte	0x4
	.byte	0x4
	.set L$set$14,LCFI9-LCFI8
	.long L$set$14
	.byte	0xe
	.byte	0x28
	.byte	0x83
	.byte	0x5
	.byte	0x4
	.set L$set$15,LCFI10-LCFI9
	.long L$set$15
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$16,LCFI11-LCFI10
	.long L$set$16
	.byte	0xa
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$17,LCFI12-LCFI11
	.long L$set$17
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$18,LCFI13-LCFI12
	.long L$set$18
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$19,LCFI14-LCFI13
	.long L$set$19
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$20,LCFI15-LCFI14
	.long L$set$20
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$21,LCFI16-LCFI15
	.long L$set$21
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$22,LEFDE5-LASFDE5
	.long L$set$22
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB707-.
	.set L$set$23,LFE707-LFB707
	.quad L$set$23
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$24,LEFDE7-LASFDE7
	.long L$set$24
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB708-.
	.set L$set$25,LFE708-LFB708
	.quad L$set$25
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$26,LEFDE9-LASFDE9
	.long L$set$26
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB709-.
	.set L$set$27,LFE709-LFB709
	.quad L$set$27
	.byte	0
	.byte	0x4
	.set L$set$28,LCFI17-LFB709
	.long L$set$28
	.byte	0xe
	.byte	0x10
	.byte	0x8e
	.byte	0x2
	.byte	0x4
	.set L$set$29,LCFI18-LCFI17
	.long L$set$29
	.byte	0xe
	.byte	0x18
	.byte	0x8d
	.byte	0x3
	.byte	0x4
	.set L$set$30,LCFI19-LCFI18
	.long L$set$30
	.byte	0xe
	.byte	0x20
	.byte	0x8c
	.byte	0x4
	.byte	0x4
	.set L$set$31,LCFI20-LCFI19
	.long L$set$31
	.byte	0xe
	.byte	0x28
	.byte	0x86
	.byte	0x5
	.byte	0x4
	.set L$set$32,LCFI21-LCFI20
	.long L$set$32
	.byte	0xe
	.byte	0x30
	.byte	0x83
	.byte	0x6
	.byte	0x4
	.set L$set$33,LCFI22-LCFI21
	.long L$set$33
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$34,LCFI23-LCFI22
	.long L$set$34
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$35,LCFI24-LCFI23
	.long L$set$35
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$36,LCFI25-LCFI24
	.long L$set$36
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$37,LCFI26-LCFI25
	.long L$set$37
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
