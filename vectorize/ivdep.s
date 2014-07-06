	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB0:
	movl	_N(%rip), %ecx
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
LCFI1:
	movq	%rsp, %rbp
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
LCFI2:
	pushq	%rbx
LCFI3:
	testl	%ecx, %ecx
	je	L25
	movq	_b(%rip), %rdx
	movq	_a(%rip), %rsi
	movq	_c(%rip), %rdi
	movq	%rdx, %rax
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	cmpl	%ecx, %eax
	cmova	%ecx, %eax
	cmpl	$8, %ecx
	ja	L29
	movl	%ecx, %eax
L3:
	vmovss	(%rdi), %xmm0
	vmulss	(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi)
	cmpl	$1, %eax
	je	L15
	vmovss	4(%rdi), %xmm0
	vmulss	4(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 4(%rsi)
	cmpl	$2, %eax
	je	L16
	vmovss	8(%rdi), %xmm0
	vmulss	8(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 8(%rsi)
	cmpl	$3, %eax
	je	L17
	vmovss	12(%rdi), %xmm0
	vmulss	12(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 12(%rsi)
	cmpl	$4, %eax
	je	L18
	vmovss	16(%rdi), %xmm0
	vmulss	16(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 16(%rsi)
	cmpl	$5, %eax
	je	L19
	vmovss	20(%rdi), %xmm0
	vmulss	20(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 20(%rsi)
	cmpl	$6, %eax
	je	L20
	vmovss	24(%rdi), %xmm0
	vmulss	24(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 24(%rsi)
	cmpl	$8, %eax
	jne	L21
	vmovss	28(%rdi), %xmm0
	movl	$8, %r8d
	vmulss	28(%rdx), %xmm0, %xmm0
	vmovss	%xmm0, 28(%rsi)
L5:
	cmpl	%eax, %ecx
	je	L25
L4:
	leal	-1(%rcx), %r12d
	movl	%ecx, %r11d
	movl	%eax, %r10d
	subl	%eax, %r11d
	subl	%eax, %r12d
	leal	-8(%r11), %r9d
	shrl	$3, %r9d
	addl	$1, %r9d
	leal	0(,%r9,8), %ebx
	cmpl	$6, %r12d
	jbe	L7
	leaq	0(,%r10,4), %rax
	xorl	%r12d, %r12d
	xorl	%r10d, %r10d
	leaq	(%rdx,%rax), %r14
	leaq	(%rdi,%rax), %r13
	addq	%rsi, %rax
L8:
	addl	$1, %r12d
	vmovups	0(%r13,%r10), %ymm0
	vmulps	(%r14,%r10), %ymm0, %ymm0
	vmovups	%ymm0, (%rax,%r10)
	addq	$32, %r10
	cmpl	%r12d, %r9d
	ja	L8
	addl	%ebx, %r8d
	cmpl	%ebx, %r11d
	je	L24
	vzeroupper
L7:
	movl	%r8d, %eax
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	leal	1(%r8), %eax
	cmpl	%ecx, %eax
	jae	L25
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	leal	2(%r8), %eax
	cmpl	%eax, %ecx
	jbe	L25
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	leal	3(%r8), %eax
	cmpl	%eax, %ecx
	jbe	L25
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	leal	4(%r8), %eax
	cmpl	%eax, %ecx
	jbe	L25
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	leal	5(%r8), %eax
	cmpl	%eax, %ecx
	jbe	L25
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
	leal	6(%r8), %eax
	cmpl	%eax, %ecx
	jbe	L25
	vmovss	(%rdi,%rax,4), %xmm0
	vmulss	(%rdx,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%rax,4)
L25:
	popq	%rbx
	popq	%r10
LCFI4:
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI5:
	ret
	.align 4,0x90
L29:
LCFI6:
	testl	%eax, %eax
	jne	L3
	xorl	%r8d, %r8d
	jmp	L4
	.align 4,0x90
L24:
	vzeroupper
	popq	%rbx
	popq	%r10
LCFI7:
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI8:
	ret
	.align 4,0x90
L18:
LCFI9:
	movl	$4, %r8d
	jmp	L5
	.align 4,0x90
L19:
	movl	$5, %r8d
	jmp	L5
	.align 4,0x90
L20:
	movl	$6, %r8d
	jmp	L5
	.align 4,0x90
L21:
	movl	$7, %r8d
	jmp	L5
	.align 4,0x90
L16:
	movl	$2, %r8d
	jmp	L5
	.align 4,0x90
L17:
	movl	$3, %r8d
	jmp	L5
	.align 4,0x90
L15:
	movl	$1, %r8d
	jmp	L5
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE0:
	.text
LHOTE0:
	.globl _c
	.zerofill __DATA,__pu_bss3,_c,8,3
	.globl _b
	.zerofill __DATA,__pu_bss3,_b,8,3
	.globl _a
	.zerofill __DATA,__pu_bss3,_a,8,3
	.globl _N
	.zerofill __DATA,__pu_bss2,_N,4,2
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
	.byte	0x4
	.set L$set$3,LCFI0-LFB0
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
	.byte	0x60
	.byte	0x6
	.byte	0x10
	.byte	0xe
	.byte	0x2
	.byte	0x76
	.byte	0x78
	.byte	0x10
	.byte	0xd
	.byte	0x2
	.byte	0x76
	.byte	0x70
	.byte	0x10
	.byte	0xc
	.byte	0x2
	.byte	0x76
	.byte	0x68
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0x10
	.byte	0x3
	.byte	0x2
	.byte	0x76
	.byte	0x58
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$8,LCFI5-LCFI4
	.long L$set$8
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$9,LCFI6-LCFI5
	.long L$set$9
	.byte	0xb
	.byte	0x4
	.set L$set$10,LCFI7-LCFI6
	.long L$set$10
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI8-LCFI7
	.long L$set$11
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$12,LCFI9-LCFI8
	.long L$set$12
	.byte	0xb
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
