	.text
	.align 4,0x90
	.globl __Z3asqv
__Z3asqv:
LFB222:
	vmovaps	LC0(%rip), %ymm5
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	vmovaps	LC1(%rip), %ymm4
	leaq	_b(%rip), %rdx
	vxorps	%xmm3, %xmm3, %xmm3
	.align 4,0x90
L2:
	vmovaps	(%rcx,%rax), %ymm1
	vcmpneqps	%ymm1, %ymm3, %ymm2
	vrsqrtps	%ymm1, %ymm0
	vandps	%ymm2, %ymm0, %ymm0
	vmulps	%ymm1, %ymm0, %ymm1
	vmulps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm5, %ymm0, %ymm0
	vmulps	%ymm4, %ymm1, %ymm1
	vmulps	%ymm1, %ymm0, %ymm1
	vmovaps	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L2
	vzeroupper
	ret
LFE222:
	.align 4,0x90
	.globl __Z4sumqv
__Z4sumqv:
LFB223:
	vmovaps	LC0(%rip), %ymm6
	vxorps	%xmm2, %xmm2, %xmm2
	vmovaps	%ymm2, %ymm4
	vmovaps	LC1(%rip), %ymm5
	leaq	_a(%rip), %rax
	leaq	4096+_a(%rip), %rdx
	.align 4,0x90
L6:
	vmovaps	(%rax), %ymm1
	addq	$32, %rax
	cmpq	%rdx, %rax
	vcmpneqps	%ymm1, %ymm4, %ymm3
	vrsqrtps	%ymm1, %ymm0
	vandps	%ymm3, %ymm0, %ymm0
	vmulps	%ymm1, %ymm0, %ymm1
	vmulps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm6, %ymm0, %ymm0
	vmulps	%ymm5, %ymm1, %ymm1
	vmulps	%ymm1, %ymm0, %ymm1
	vaddps	%ymm1, %ymm2, %ymm2
	jne	L6
	vhaddps	%ymm2, %ymm2, %ymm2
	vhaddps	%ymm2, %ymm2, %ymm0
	vperm2f128	$1, %ymm0, %ymm0, %ymm2
	vaddps	%ymm0, %ymm2, %ymm0
	vzeroupper
	ret
LFE223:
	.align 4,0x90
	.globl __Z3sumv
__Z3sumv:
LFB224:
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	leaq	_b(%rip), %rdx
	.align 4,0x90
L9:
	vmovaps	(%rcx,%rax), %ymm2
	vfmadd231ps	(%rdx,%rax), %ymm2, %ymm0
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L9
	vhaddps	%ymm0, %ymm0, %ymm0
	vhaddps	%ymm0, %ymm0, %ymm1
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vaddps	%ymm1, %ymm0, %ymm0
	vzeroupper
	ret
LFE224:
	.align 4,0x90
	.globl __Z4sumOv
__Z4sumOv:
LFB225:
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	leaq	_b(%rip), %rdx
	.align 4,0x90
L12:
	vmovaps	(%rcx,%rax), %ymm2
	vfmadd231ps	(%rdx,%rax), %ymm2, %ymm0
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L12
	vhaddps	%ymm0, %ymm0, %ymm0
	vhaddps	%ymm0, %ymm0, %ymm1
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vaddps	%ymm1, %ymm0, %ymm0
	vzeroupper
	ret
LFE225:
	.align 4,0x90
	.globl __Z5sumO1v
__Z5sumO1v:
LFB226:
	leaq	_a(%rip), %rcx
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	leaq	_b(%rip), %rdx
	.align 4,0x90
L15:
	vmovaps	(%rcx,%rax), %ymm2
	vfmadd231ps	(%rdx,%rax), %ymm2, %ymm0
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L15
	vhaddps	%ymm0, %ymm0, %ymm0
	vhaddps	%ymm0, %ymm0, %ymm1
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vaddps	%ymm1, %ymm0, %ymm0
	vzeroupper
	ret
LFE226:
	.align 4,0x90
	.globl __Z3minff
__Z3minff:
LFB227:
	vminss	%xmm1, %xmm0, %xmm0
	ret
LFE227:
	.align 4,0x90
	.globl __Z6distsqff
__Z6distsqff:
LFB228:
	vsubss	%xmm1, %xmm0, %xmm1
	vmulss	%xmm1, %xmm1, %xmm0
	ret
LFE228:
	.align 4,0x90
	.globl __Z7examplev
__Z7examplev:
LFB229:
	leaq	_a(%rip), %rdi
	xorl	%eax, %eax
	leaq	_b(%rip), %rsi
	leaq	_d(%rip), %rcx
	leaq	_c(%rip), %rdx
	.align 4,0x90
L20:
	vmovaps	(%rdi,%rax), %ymm0
	vsubps	(%rsi,%rax), %ymm0, %ymm0
	vmulps	%ymm0, %ymm0, %ymm0
	vminps	(%rdx,%rax), %ymm0, %ymm0
	vmovaps	%ymm0, (%rcx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L20
	vzeroupper
	ret
LFE229:
	.align 4,0x90
	.globl __Z8multSelfRii
__Z8multSelfRii:
LFB230:
	imull	(%rdi), %esi
	movl	%esi, (%rdi)
	ret
LFE230:
	.align 4,0x90
	.globl __Z4multjj
__Z4multjj:
LFB231:
	movl	%edi, %eax
	movl	%esi, %esi
	imulq	%rsi, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	ret
LFE231:
	.align 4,0x90
	.globl __Z8multSelfRjj
__Z8multSelfRjj:
LFB232:
	movl	(%rdi), %eax
	movl	%esi, %esi
	imulq	%rsi, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	movl	%eax, (%rdi)
	ret
LFE232:
	.align 4,0x90
	.globl __Z3fooPKii
__Z3fooPKii:
LFB233:
	testl	%esi, %esi
	jle	L39
	movq	%rdi, %rax
	movl	%esi, %edx
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	cmpl	%esi, %eax
	cmova	%esi, %eax
	cmpl	$17, %esi
	jg	L64
L38:
	cmpl	$1, %edx
	movl	(%rdi), %eax
	je	L41
	imull	4(%rdi), %eax
	cmpl	$2, %edx
	je	L42
	imull	8(%rdi), %eax
	cmpl	$3, %edx
	je	L43
	imull	12(%rdi), %eax
	cmpl	$4, %edx
	je	L44
	imull	16(%rdi), %eax
	cmpl	$5, %edx
	je	L45
	imull	20(%rdi), %eax
	cmpl	$6, %edx
	je	L46
	imull	24(%rdi), %eax
	cmpl	$7, %edx
	je	L47
	imull	28(%rdi), %eax
	cmpl	$8, %edx
	je	L48
	imull	32(%rdi), %eax
	cmpl	$9, %edx
	je	L49
	imull	36(%rdi), %eax
	cmpl	$10, %edx
	je	L50
	imull	40(%rdi), %eax
	cmpl	$11, %edx
	je	L51
	imull	44(%rdi), %eax
	cmpl	$12, %edx
	je	L52
	imull	48(%rdi), %eax
	cmpl	$13, %edx
	je	L53
	imull	52(%rdi), %eax
	cmpl	$14, %edx
	je	L54
	imull	56(%rdi), %eax
	cmpl	$15, %edx
	je	L55
	imull	60(%rdi), %eax
	cmpl	$17, %edx
	jne	L56
	imull	64(%rdi), %eax
	movl	$17, %ecx
L29:
	cmpl	%edx, %esi
	je	L65
L28:
	movl	%esi, %r11d
	movl	%edx, %r8d
	subl	%edx, %r11d
	movl	%r11d, %r10d
	shrl	$3, %r10d
	leal	0(,%r10,8), %r9d
	testl	%r9d, %r9d
	je	L31
	vmovdqa	LC2(%rip), %ymm0
	leaq	(%rdi,%r8,4), %r8
	xorl	%edx, %edx
L37:
	addl	$1, %edx
	vpmulld	(%r8), %ymm0, %ymm0
	addq	$32, %r8
	cmpl	%edx, %r10d
	ja	L37
	vmovdqa	%xmm0, %xmm1
	vextracti128	$0x1, %ymm0, %xmm0
	addl	%r9d, %ecx
	vmovd	%xmm1, %r8d
	vpextrd	$1, %xmm1, %edx
	imull	%r8d, %edx
	vpextrd	$2, %xmm1, %r8d
	imull	%edx, %r8d
	vpextrd	$3, %xmm1, %edx
	imull	%r8d, %edx
	vmovd	%xmm0, %r8d
	imull	%edx, %r8d
	vpextrd	$1, %xmm0, %edx
	imull	%r8d, %edx
	vpextrd	$2, %xmm0, %r8d
	imull	%edx, %r8d
	vpextrd	$3, %xmm0, %edx
	imull	%r8d, %edx
	imull	%edx, %eax
	cmpl	%r9d, %r11d
	je	L62
	vzeroupper
L31:
	movslq	%ecx, %rdx
	imull	(%rdi,%rdx,4), %eax
	leal	1(%rcx), %edx
	cmpl	%edx, %esi
	jle	L63
	movslq	%edx, %rdx
	imull	(%rdi,%rdx,4), %eax
	leal	2(%rcx), %edx
	cmpl	%edx, %esi
	jle	L63
	movslq	%edx, %rdx
	imull	(%rdi,%rdx,4), %eax
	leal	3(%rcx), %edx
	cmpl	%edx, %esi
	jle	L63
	movslq	%edx, %rdx
	imull	(%rdi,%rdx,4), %eax
	leal	4(%rcx), %edx
	cmpl	%edx, %esi
	jle	L63
	movslq	%edx, %rdx
	imull	(%rdi,%rdx,4), %eax
	leal	5(%rcx), %edx
	cmpl	%edx, %esi
	jle	L63
	movslq	%edx, %rdx
	addl	$6, %ecx
	imull	(%rdi,%rdx,4), %eax
	cmpl	%ecx, %esi
	jle	L66
	movslq	%ecx, %rcx
	imull	(%rdi,%rcx,4), %eax
	ret
	.align 4,0x90
L62:
	vzeroupper
L63:
	rep; ret
	.align 4,0x90
L65:
	rep; ret
	.align 4,0x90
L64:
	testl	%eax, %eax
	movl	%eax, %edx
	jne	L38
	movl	$1, %eax
	xorl	%ecx, %ecx
	jmp	L28
	.align 4,0x90
L66:
	rep; ret
	.align 4,0x90
L39:
	movl	$1, %eax
	ret
	.align 4,0x90
L46:
	movl	$6, %ecx
	jmp	L29
	.align 4,0x90
L47:
	movl	$7, %ecx
	jmp	L29
	.align 4,0x90
L56:
	movl	$16, %ecx
	jmp	L29
	.align 4,0x90
L41:
	movl	$1, %ecx
	jmp	L29
	.align 4,0x90
L44:
	movl	$4, %ecx
	jmp	L29
	.align 4,0x90
L45:
	movl	$5, %ecx
	jmp	L29
	.align 4,0x90
L54:
	movl	$14, %ecx
	jmp	L29
	.align 4,0x90
L55:
	movl	$15, %ecx
	jmp	L29
	.align 4,0x90
L48:
	movl	$8, %ecx
	jmp	L29
	.align 4,0x90
L49:
	movl	$9, %ecx
	jmp	L29
	.align 4,0x90
L50:
	movl	$10, %ecx
	jmp	L29
	.align 4,0x90
L51:
	movl	$11, %ecx
	jmp	L29
	.align 4,0x90
L52:
	movl	$12, %ecx
	jmp	L29
	.align 4,0x90
L53:
	movl	$13, %ecx
	jmp	L29
	.align 4,0x90
L42:
	movl	$2, %ecx
	jmp	L29
	.align 4,0x90
L43:
	movl	$3, %ecx
	jmp	L29
LFE233:
	.align 4,0x90
	.globl __Z3barPKjS0_Pji
__Z3barPKjS0_Pji:
LFB234:
	testl	%ecx, %ecx
	jle	L102
	leaq	32(%rdx), %rax
	leaq	32(%rdi), %r9
	cmpq	%rax, %rdi
	setae	%r8b
	cmpq	%r9, %rdx
	setae	%r9b
	orl	%r9d, %r8d
	cmpq	%rax, %rsi
	leaq	32(%rsi), %r9
	setae	%al
	cmpq	%r9, %rdx
	setae	%r9b
	orl	%r9d, %eax
	testb	%al, %r8b
	je	L69
	cmpl	$10, %ecx
	jbe	L69
	pushq	%rbp
LCFI0:
	movq	%rdi, %rax
	movq	%rsp, %rbp
LCFI1:
	pushq	%r14
	andl	$31, %eax
	pushq	%r13
	shrq	$2, %rax
	pushq	%r12
	negq	%rax
	pushq	%rbx
	andl	$7, %eax
	andq	$-32, %rsp
	addq	$16, %rsp
	cmpl	%eax, %ecx
	cmovbe	%ecx, %eax
	xorl	%r8d, %r8d
LCFI2:
	testl	%eax, %eax
	je	L70
	movl	(%rsi), %r8d
	movl	(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	cmpl	$1, %eax
	movl	%r8d, (%rdx)
	je	L81
	movl	4(%rsi), %r8d
	movl	4(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	cmpl	$2, %eax
	movl	%r8d, 4(%rdx)
	je	L82
	movl	8(%rsi), %r8d
	movl	8(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	cmpl	$3, %eax
	movl	%r8d, 8(%rdx)
	je	L83
	movl	12(%rsi), %r8d
	movl	12(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	cmpl	$4, %eax
	movl	%r8d, 12(%rdx)
	je	L84
	movl	16(%rsi), %r8d
	movl	16(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	cmpl	$5, %eax
	movl	%r8d, 16(%rdx)
	je	L85
	movl	20(%rsi), %r8d
	movl	20(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	cmpl	$7, %eax
	movl	%r8d, 20(%rdx)
	jne	L86
	movl	24(%rsi), %r8d
	movl	24(%rdi), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	movl	%r8d, 24(%rdx)
	movl	$7, %r8d
L70:
	movl	%ecx, %ebx
	movl	%eax, %r9d
	subl	%eax, %ebx
	movl	%ebx, %r11d
	shrl	$3, %r11d
	leal	0(,%r11,8), %r10d
	testl	%r10d, %r10d
	je	L72
	leaq	0(,%r9,4), %rax
	vmovdqa	LC3(%rip), %ymm4
	xorl	%r9d, %r9d
	leaq	(%rsi,%rax), %r14
	leaq	(%rdi,%rax), %r13
	leaq	(%rdx,%rax), %r12
	xorl	%eax, %eax
L77:
	vmovdqu	(%r14,%rax), %ymm3
	vpermq	$216, 0(%r13,%rax), %ymm1
	vpshufd	$80, %ymm1, %ymm0
	vpshufd	$250, %ymm1, %ymm1
	addl	$1, %r9d
	vpermq	$216, %ymm3, %ymm3
	vpshufd	$80, %ymm3, %ymm2
	vpmuludq	%ymm0, %ymm2, %ymm2
	vpshufd	$250, %ymm3, %ymm0
	vpmuludq	%ymm1, %ymm0, %ymm0
	vpaddq	%ymm4, %ymm2, %ymm2
	vpaddq	%ymm4, %ymm0, %ymm0
	vpsrlq	$23, %ymm2, %ymm2
	vpsrlq	$23, %ymm0, %ymm0
	vperm2i128	$32, %ymm0, %ymm2, %ymm1
	vperm2i128	$49, %ymm0, %ymm2, %ymm0
	vpshufd	$216, %ymm1, %ymm1
	vpshufd	$216, %ymm0, %ymm0
	vpunpcklqdq	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, (%r12,%rax)
	addq	$32, %rax
	cmpl	%r11d, %r9d
	jb	L77
	addl	%r10d, %r8d
	cmpl	%r10d, %ebx
	je	L98
	vzeroupper
L72:
	movslq	%r8d, %rax
	movl	(%rsi,%rax,4), %r9d
	movl	(%rdi,%rax,4), %r10d
	imulq	%r10, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	movl	%r9d, (%rdx,%rax,4)
	leal	1(%r8), %eax
	cmpl	%eax, %ecx
	jle	L99
	cltq
	movl	(%rsi,%rax,4), %r9d
	movl	(%rdi,%rax,4), %r10d
	imulq	%r10, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	movl	%r9d, (%rdx,%rax,4)
	leal	2(%r8), %eax
	cmpl	%eax, %ecx
	jle	L99
	cltq
	movl	(%rsi,%rax,4), %r9d
	movl	(%rdi,%rax,4), %r10d
	imulq	%r10, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	movl	%r9d, (%rdx,%rax,4)
	leal	3(%r8), %eax
	cmpl	%eax, %ecx
	jle	L99
	cltq
	movl	(%rsi,%rax,4), %r9d
	movl	(%rdi,%rax,4), %r10d
	imulq	%r10, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	movl	%r9d, (%rdx,%rax,4)
	leal	4(%r8), %eax
	cmpl	%eax, %ecx
	jle	L99
	cltq
	movl	(%rsi,%rax,4), %r9d
	movl	(%rdi,%rax,4), %r10d
	imulq	%r10, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	movl	%r9d, (%rdx,%rax,4)
	leal	5(%r8), %eax
	cmpl	%eax, %ecx
	jle	L99
	cltq
	addl	$6, %r8d
	movl	(%rsi,%rax,4), %r9d
	movl	(%rdi,%rax,4), %r10d
	imulq	%r10, %r9
	addq	$4194304, %r9
	shrq	$23, %r9
	cmpl	%r8d, %ecx
	movl	%r9d, (%rdx,%rax,4)
	jle	L99
	movslq	%r8d, %r8
	movl	(%rsi,%r8,4), %eax
	movl	(%rdi,%r8,4), %ecx
	imulq	%rcx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	movl	%eax, (%rdx,%r8,4)
L99:
	leaq	-32(%rbp), %rsp
	popq	%rbx
LCFI3:
	popq	%r12
LCFI4:
	popq	%r13
LCFI5:
	popq	%r14
LCFI6:
	popq	%rbp
LCFI7:
L102:
	rep; ret
	.align 4,0x90
L69:
	xorl	%eax, %eax
	.align 4,0x90
L79:
	movl	(%rsi,%rax,4), %r8d
	movl	(%rdi,%rax,4), %r9d
	imulq	%r9, %r8
	addq	$4194304, %r8
	shrq	$23, %r8
	movl	%r8d, (%rdx,%rax,4)
	addq	$1, %rax
	cmpl	%eax, %ecx
	jg	L79
	rep; ret
	.align 4,0x90
L98:
LCFI8:
	vzeroupper
	jmp	L99
	.align 4,0x90
L86:
	movl	$6, %r8d
	jmp	L70
	.align 4,0x90
L85:
	movl	$5, %r8d
	jmp	L70
	.align 4,0x90
L84:
	movl	$4, %r8d
	jmp	L70
	.align 4,0x90
L83:
	movl	$3, %r8d
	jmp	L70
	.align 4,0x90
L82:
	movl	$2, %r8d
	jmp	L70
	.align 4,0x90
L81:
	movl	$1, %r8d
	jmp	L70
LFE234:
	.align 4,0x90
	.globl __Z6barRedPKji
__Z6barRedPKji:
LFB235:
	testl	%esi, %esi
	jle	L106
	xorl	%edx, %edx
	movl	$1, %eax
	.align 4,0x90
L105:
	movl	(%rdi,%rdx,4), %ecx
	movl	%eax, %eax
	addq	$1, %rdx
	imulq	%rax, %rcx
	addq	$4194304, %rcx
	shrq	$23, %rcx
	cmpl	%edx, %esi
	movq	%rcx, %rax
	jg	L105
	rep; ret
L106:
	movl	$1, %eax
	ret
LFE235:
	.globl _d
	.zerofill __DATA,__pu_bss5,_d,4096,5
	.globl _c
	.zerofill __DATA,__pu_bss5,_c,4096,5
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,4096,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,4096,5
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
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.align 5
LC3:
	.quad	4194304
	.quad	4194304
	.quad	4194304
	.quad	4194304
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
	.quad	LFB222-.
	.set L$set$2,LFE222-LFB222
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB223-.
	.set L$set$4,LFE223-LFB223
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB224-.
	.set L$set$6,LFE224-LFB224
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB225-.
	.set L$set$8,LFE225-LFB225
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB226-.
	.set L$set$10,LFE226-LFB226
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB227-.
	.set L$set$12,LFE227-LFB227
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB228-.
	.set L$set$14,LFE228-LFB228
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB229-.
	.set L$set$16,LFE229-LFB229
	.quad L$set$16
	.byte	0
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$17,LEFDE17-LASFDE17
	.long L$set$17
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB230-.
	.set L$set$18,LFE230-LFB230
	.quad L$set$18
	.byte	0
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$19,LEFDE19-LASFDE19
	.long L$set$19
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB231-.
	.set L$set$20,LFE231-LFB231
	.quad L$set$20
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$21,LEFDE21-LASFDE21
	.long L$set$21
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB232-.
	.set L$set$22,LFE232-LFB232
	.quad L$set$22
	.byte	0
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$23,LEFDE23-LASFDE23
	.long L$set$23
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB233-.
	.set L$set$24,LFE233-LFB233
	.quad L$set$24
	.byte	0
	.align 3
LEFDE23:
LSFDE25:
	.set L$set$25,LEFDE25-LASFDE25
	.long L$set$25
LASFDE25:
	.long	LASFDE25-EH_frame1
	.quad	LFB234-.
	.set L$set$26,LFE234-LFB234
	.quad L$set$26
	.byte	0
	.byte	0x4
	.set L$set$27,LCFI0-LFB234
	.long L$set$27
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$28,LCFI1-LCFI0
	.long L$set$28
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$29,LCFI2-LCFI1
	.long L$set$29
	.byte	0x8e
	.byte	0x3
	.byte	0x8d
	.byte	0x4
	.byte	0x8c
	.byte	0x5
	.byte	0x83
	.byte	0x6
	.byte	0x4
	.set L$set$30,LCFI3-LCFI2
	.long L$set$30
	.byte	0xc3
	.byte	0x4
	.set L$set$31,LCFI4-LCFI3
	.long L$set$31
	.byte	0xcc
	.byte	0x4
	.set L$set$32,LCFI5-LCFI4
	.long L$set$32
	.byte	0xcd
	.byte	0x4
	.set L$set$33,LCFI6-LCFI5
	.long L$set$33
	.byte	0xce
	.byte	0x4
	.set L$set$34,LCFI7-LCFI6
	.long L$set$34
	.byte	0xc6
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$35,LCFI8-LCFI7
	.long L$set$35
	.byte	0xc
	.byte	0x6
	.byte	0x10
	.byte	0x83
	.byte	0x6
	.byte	0x86
	.byte	0x2
	.byte	0x8c
	.byte	0x5
	.byte	0x8d
	.byte	0x4
	.byte	0x8e
	.byte	0x3
	.align 3
LEFDE25:
LSFDE27:
	.set L$set$36,LEFDE27-LASFDE27
	.long L$set$36
LASFDE27:
	.long	LASFDE27-EH_frame1
	.quad	LFB235-.
	.set L$set$37,LFE235-LFB235
	.quad L$set$37
	.byte	0
	.align 3
LEFDE27:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
