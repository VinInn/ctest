	.text
	.align 4,0x90
	.globl __Z8knuthPoifRN3vdt15MersenneTwisterE
__Z8knuthPoifRN3vdt15MersenneTwisterE:
LFB3748:
	pushq	%r15
LCFI0:
	vmovss	LC3(%rip), %xmm2
	pushq	%r14
LCFI1:
	vxorps	%xmm0, %xmm2, %xmm2
	vcomiss	LC4(%rip), %xmm2
	pushq	%r13
LCFI2:
	pushq	%r12
LCFI3:
	pushq	%rbp
LCFI4:
	pushq	%rbx
LCFI5:
	ja	L28
	vmovss	LC6(%rip), %xmm4
	vmovaps	%xmm2, %xmm1
	vmovss	LC10(%rip), %xmm5
	vfmadd132ss	LC5(%rip), %xmm4, %xmm1
	vcvttss2si	%xmm1, %eax
	vmovd	%xmm1, %edx
	shrl	$31, %edx
	vmovss	LC8(%rip), %xmm1
	vmovss	LC2(%rip), %xmm8
	subl	%edx, %eax
	vcvtsi2ss	%eax, %xmm3, %xmm3
	vfmsub231ss	LC7(%rip), %xmm3, %xmm0
	vcvttss2si	%xmm3, %eax
	vfnmadd132ss	%xmm3, %xmm0, %xmm1
	vmulss	%xmm1, %xmm1, %xmm9
	vmovaps	%xmm1, %xmm0
	vfmadd132ss	LC9(%rip), %xmm5, %xmm0
	vfmadd132ss	%xmm1, %xmm5, %xmm0
	vfmadd213ss	LC11(%rip), %xmm1, %xmm0
	vfmadd213ss	LC12(%rip), %xmm1, %xmm0
	vfmadd213ss	LC13(%rip), %xmm1, %xmm0
	vfmadd132ss	%xmm1, %xmm4, %xmm0
	addl	$127, %eax
	vaddss	%xmm8, %xmm1, %xmm1
	sall	$23, %eax
	vmovd	%eax, %xmm4
	vfmadd132ss	%xmm0, %xmm1, %xmm9
	vmulss	%xmm4, %xmm9, %xmm9
L2:
	movq	%rdi, %rax
	movl	$396, -32(%rsp)
	xorl	%r15d, %r15d
	andl	$15, %eax
	movl	2496(%rdi), %r11d
	movl	$227, -24(%rsp)
	shrq	$2, %rax
	vmovdqa	LC15(%rip), %xmm5
	vcmpnltss	LC14(%rip), %xmm2, %xmm2
	negq	%rax
	vmovdqa	LC16(%rip), %xmm4
	vpxor	%xmm1, %xmm1, %xmm1
	andl	$3, %eax
	vmovdqa	LC17(%rip), %xmm3
	movl	%eax, %ecx
	movl	%eax, -40(%rsp)
	subl	$1, %eax
	subl	%ecx, -32(%rsp)
	movl	%eax, -28(%rsp)
	movl	%ecx, %eax
	subl	%ecx, -24(%rsp)
	vandps	%xmm2, %xmm9, %xmm9
	vmovdqa	LC18(%rip), %xmm2
	leaq	1588(,%rax,4), %rdx
	salq	$2, %rax
	movl	-32(%rsp), %ecx
	leaq	(%rdi,%rdx), %r8
	leaq	-1584(%rdi,%rdx), %rsi
	leaq	(%rdi,%rax), %r14
	leaq	-1588(%rdi,%rdx), %rdx
	shrl	$2, %ecx
	movl	%ecx, -16(%rsp)
	sall	$2, %ecx
	movl	%ecx, -12(%rsp)
	leaq	912(%rdi,%rax), %rcx
	leaq	908(%rdi,%rax), %rax
	jmp	L19
	.align 4,0x90
L54:
	movl	%r11d, %r9d
	addl	$1, %r11d
L5:
	movl	%r11d, 2496(%rdi)
	movl	(%rdi,%r9,4), %r10d
	movl	%r10d, %r9d
	shrl	$11, %r9d
	xorl	%r10d, %r9d
	movl	%r9d, %r10d
	sall	$7, %r10d
	andl	$-1658038656, %r10d
	xorl	%r9d, %r10d
	movl	%r10d, %r9d
	sall	$15, %r9d
	andl	$-272236544, %r9d
	xorl	%r10d, %r9d
	movl	%r9d, %r10d
	shrl	$18, %r10d
	xorl	%r10d, %r9d
	vcvtsi2ssq	%r9, %xmm0, %xmm0
	leal	1(%r15), %r9d
	vmulss	%xmm0, %xmm8, %xmm8
	vcomiss	%xmm9, %xmm8
	jbe	L49
	movl	%r9d, %r15d
L19:
	cmpl	$624, %r11d
	jne	L54
	movl	-40(%rsp), %ebx
	testl	%ebx, %ebx
	je	L34
	cmpl	$3, %ebx
	jne	L35
	movl	4(%rdi), %r10d
	movl	(%rdi), %r11d
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	andl	$-2147483648, %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	andl	$-2147483648, %r11d
	negl	%r9d
	orl	%ebx, %r11d
	andl	$-1727483681, %r9d
	shrq	%r11
	xorl	1588(%rdi), %r9d
	xorl	%r11d, %r9d
	movl	8(%rdi), %r11d
	movl	%r9d, (%rdi)
	movl	%r11d, %r9d
	movl	%r11d, %ebx
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r10d
	andl	$-1727483681, %r9d
	shrq	%r10
	xorl	1592(%rdi), %r9d
	xorl	%r10d, %r9d
	cmpl	$3, -28(%rsp)
	movl	%r9d, 4(%rdi)
	jbe	L36
	movl	12(%rdi), %r10d
	andl	$-2147483648, %r11d
	movl	$221, %r13d
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	andl	$-2147483648, %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r11d
	andl	$-1727483681, %r9d
	shrq	%r11
	xorl	1596(%rdi), %r9d
	xorl	%r11d, %r9d
	movl	16(%rdi), %r11d
	movl	%r9d, 8(%rdi)
	movl	%r11d, %r9d
	movl	%r11d, %ebx
	andl	$-2147483648, %r11d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r10d
	andl	$-1727483681, %r9d
	shrq	%r10
	xorl	1600(%rdi), %r9d
	xorl	%r10d, %r9d
	movl	20(%rdi), %r10d
	movl	%r9d, 12(%rdi)
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	andl	$-2147483648, %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r11d
	andl	$-1727483681, %r9d
	shrq	%r11
	xorl	1604(%rdi), %r9d
	xorl	%r11d, %r9d
	movl	24(%rdi), %r11d
	movl	%r9d, 16(%rdi)
	movl	%r11d, %r9d
	andl	$2147483647, %r11d
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	1608(%rdi), %r9d
	orl	%r11d, %r10d
	movl	$6, %r11d
	shrq	%r10
	xorl	%r10d, %r9d
	movl	%r9d, 20(%rdi)
L25:
	leal	1(%r11), %r9d
	movl	%r11d, %r10d
	leaq	(%rdi,%r10,4), %r12
	movl	%r9d, %ebp
	movl	(%rdi,%rbp,4), %r10d
	leal	397(%r11), %ebx
	andl	$1, %r10d
	negl	%r10d
	andl	$-1727483681, %r10d
	xorl	(%rdi,%rbx,4), %r10d
	movl	(%rdi,%rbp,4), %ebx
	andl	$2147483647, %ebx
	movl	%ebx, -36(%rsp)
	movl	(%r12), %ebx
	andl	$-2147483648, %ebx
	orl	-36(%rsp), %ebx
	shrq	%rbx
	xorl	%ebx, %r10d
	cmpl	%r9d, -40(%rsp)
	movl	%r10d, (%r12)
	leal	-1(%r13), %ebx
	jbe	L24
	leal	2(%r11), %r9d
	addl	$398, %r11d
	leaq	(%rdi,%rbp,4), %rbx
	movl	%r9d, %r12d
	movl	(%rdi,%r12,4), %r10d
	movl	(%rdi,%r12,4), %ebp
	andl	$1, %r10d
	negl	%r10d
	andl	$2147483647, %ebp
	andl	$-1727483681, %r10d
	xorl	(%rdi,%r11,4), %r10d
	movl	(%rbx), %r11d
	andl	$-2147483648, %r11d
	orl	%ebp, %r11d
	shrq	%r11
	xorl	%r11d, %r10d
	cmpl	%r9d, -40(%rsp)
	movl	%r10d, (%rbx)
	leal	-2(%r13), %ebx
	jbe	L24
	movl	12(%rdi), %r10d
	leaq	(%rdi,%r12,4), %r11
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	movl	(%r11), %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	1596(%rdi), %r9d
	andl	$-2147483648, %r10d
	orl	%ebx, %r10d
	leal	-3(%r13), %ebx
	shrq	%r10
	xorl	%r10d, %r9d
	movl	%r9d, (%r11)
	movl	$3, %r9d
L24:
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	.align 4,0x90
L27:
	vmovdqu	(%rsi,%r10), %xmm0
	addl	$1, %r11d
	vmovdqu	(%r8,%r10), %xmm7
	vpand	%xmm5, %xmm0, %xmm6
	vpcmpeqd	%xmm1, %xmm6, %xmm6
	vpand	%xmm3, %xmm0, %xmm0
	vpandn	%xmm4, %xmm6, %xmm6
	vpxor	%xmm6, %xmm7, %xmm6
	vpand	(%rdx,%r10), %xmm2, %xmm7
	vpor	%xmm7, %xmm0, %xmm0
	vpsrld	$1, %xmm0, %xmm0
	vpxor	%xmm0, %xmm6, %xmm0
	vmovdqa	%xmm0, (%rdx,%r10)
	addq	$16, %r10
	cmpl	$55, %r11d
	jbe	L27
	cmpl	$224, -24(%rsp)
	leal	-224(%rbx), %r11d
	movl	%r11d, -36(%rsp)
	leal	224(%r9), %r10d
	je	L8
	leal	225(%r9), %ebp
	movl	(%rdi,%rbp,4), %r13d
	leaq	(%rdi,%r10,4), %r12
	leal	621(%r9), %r11d
	movl	%r13d, %r10d
	andl	$2147483647, %r13d
	andl	$1, %r10d
	negl	%r10d
	andl	$-1727483681, %r10d
	xorl	(%rdi,%r11,4), %r10d
	movl	(%r12), %r11d
	andl	$-2147483648, %r11d
	orl	%r13d, %r11d
	shrq	%r11
	xorl	%r11d, %r10d
	cmpl	$225, %ebx
	movl	%r10d, (%r12)
	je	L8
	leal	226(%r9), %r12d
	movl	(%rdi,%r12,4), %r13d
	leaq	(%rdi,%rbp,4), %rbx
	leal	622(%r9), %r11d
	movl	%r13d, %r10d
	movl	%r13d, %ebp
	andl	$1, %r10d
	andl	$2147483647, %ebp
	negl	%r10d
	andl	$-1727483681, %r10d
	xorl	(%rdi,%r11,4), %r10d
	movl	(%rbx), %r11d
	andl	$-2147483648, %r11d
	orl	%ebp, %r11d
	shrq	%r11
	xorl	%r11d, %r10d
	cmpl	$2, -36(%rsp)
	movl	%r10d, (%rbx)
	je	L8
	leal	227(%r9), %r10d
	addl	$623, %r9d
	movl	(%rdi,%r10,4), %r11d
	leaq	(%rdi,%r12,4), %rbx
	movl	%r11d, %r10d
	andl	$2147483647, %r11d
	andl	$1, %r10d
	negl	%r10d
	andl	$-1727483681, %r10d
	xorl	(%rdi,%r9,4), %r10d
	movl	(%rbx), %r9d
	andl	$-2147483648, %r9d
	orl	%r9d, %r11d
	shrq	%r11
	xorl	%r11d, %r10d
	movl	%r10d, (%rbx)
L8:
	movl	-40(%rsp), %ebx
	testl	%ebx, %ebx
	je	L31
	cmpl	$3, %ebx
	jne	L32
	movl	912(%rdi), %r10d
	movl	908(%rdi), %r11d
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	andl	$-2147483648, %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	andl	$-2147483648, %r11d
	negl	%r9d
	orl	%ebx, %r11d
	andl	$-1727483681, %r9d
	shrq	%r11
	xorl	(%rdi), %r9d
	xorl	%r11d, %r9d
	movl	916(%rdi), %r11d
	movl	%r9d, 908(%rdi)
	movl	%r11d, %r9d
	movl	%r11d, %ebx
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r10d
	andl	$-1727483681, %r9d
	shrq	%r10
	xorl	4(%rdi), %r9d
	xorl	%r10d, %r9d
	cmpl	$3, -28(%rsp)
	movl	%r9d, 912(%rdi)
	jbe	L33
	movl	920(%rdi), %r10d
	andl	$-2147483648, %r11d
	movl	$6, -36(%rsp)
	movl	$390, %r12d
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	andl	$-2147483648, %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r11d
	andl	$-1727483681, %r9d
	shrq	%r11
	xorl	8(%rdi), %r9d
	xorl	%r11d, %r9d
	movl	924(%rdi), %r11d
	movl	%r9d, 916(%rdi)
	movl	%r11d, %r9d
	movl	%r11d, %ebx
	andl	$-2147483648, %r11d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r10d
	andl	$-1727483681, %r9d
	shrq	%r10
	xorl	12(%rdi), %r9d
	xorl	%r10d, %r9d
	movl	928(%rdi), %r10d
	movl	%r9d, 920(%rdi)
	movl	%r10d, %r9d
	movl	%r10d, %ebx
	andl	$-2147483648, %r10d
	andl	$1, %r9d
	andl	$2147483647, %ebx
	negl	%r9d
	orl	%ebx, %r11d
	andl	$-1727483681, %r9d
	shrq	%r11
	xorl	16(%rdi), %r9d
	xorl	%r11d, %r9d
	movl	932(%rdi), %r11d
	movl	%r9d, 924(%rdi)
	movl	%r11d, %r9d
	andl	$2147483647, %r11d
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	20(%rdi), %r9d
	orl	%r11d, %r10d
	shrq	%r10
	xorl	%r10d, %r9d
	movl	$233, %r10d
	movl	%r9d, 928(%rdi)
L21:
	leal	1(%r10), %r13d
	movl	%r10d, %r9d
	leaq	(%rdi,%r9,4), %rbx
	movl	%r13d, %ebp
	movl	(%rdi,%rbp,4), %r9d
	leal	-227(%r10), %r11d
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	(%rdi,%r11,4), %r9d
	movl	(%rdi,%rbp,4), %r11d
	andl	$2147483647, %r11d
	movl	%r11d, -20(%rsp)
	movl	(%rbx), %r11d
	andl	$-2147483648, %r11d
	orl	-20(%rsp), %r11d
	shrq	%r11
	xorl	%r11d, %r9d
	movl	%r9d, (%rbx)
	movl	-36(%rsp), %ebx
	leal	-1(%r12), %r9d
	leal	1(%rbx), %r11d
	cmpl	%r11d, -40(%rsp)
	jbe	L20
	leal	2(%r10), %r13d
	leal	-226(%r10), %r11d
	movl	%r13d, %ebx
	movl	(%rdi,%rbx,4), %r9d
	leaq	(%rdi,%rbp,4), %rbp
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	(%rdi,%r11,4), %r9d
	movl	(%rdi,%rbx,4), %r11d
	andl	$2147483647, %r11d
	movl	%r11d, -20(%rsp)
	movl	0(%rbp), %r11d
	andl	$-2147483648, %r11d
	orl	-20(%rsp), %r11d
	shrq	%r11
	xorl	%r11d, %r9d
	movl	-36(%rsp), %r11d
	movl	%r9d, 0(%rbp)
	leal	-2(%r12), %r9d
	addl	$2, %r11d
	cmpl	%r11d, -40(%rsp)
	jbe	L20
	leal	3(%r10), %r13d
	subl	$225, %r10d
	leaq	(%rdi,%rbx,4), %rbx
	movl	%r13d, %r9d
	movl	(%rdi,%r9,4), %r11d
	movl	%r11d, %r9d
	andl	$2147483647, %r11d
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	(%rdi,%r10,4), %r9d
	movl	(%rbx), %r10d
	andl	$-2147483648, %r10d
	orl	%r10d, %r11d
	shrq	%r11
	xorl	%r11d, %r9d
	movl	%r9d, (%rbx)
	leal	-3(%r12), %r9d
L20:
	movl	-16(%rsp), %ebx
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	.align 4,0x90
L23:
	vmovdqa	(%rcx,%r10), %xmm0
	addl	$1, %r11d
	vmovdqu	(%rax,%r10), %xmm7
	vpand	%xmm5, %xmm0, %xmm6
	vpcmpeqd	%xmm1, %xmm6, %xmm6
	vpand	%xmm3, %xmm0, %xmm0
	vpand	%xmm2, %xmm7, %xmm7
	vpor	%xmm7, %xmm0, %xmm0
	vpsrld	$1, %xmm0, %xmm0
	vpandn	%xmm4, %xmm6, %xmm6
	vpxor	(%r14,%r10), %xmm6, %xmm6
	vpxor	%xmm0, %xmm6, %xmm0
	vmovdqu	%xmm0, (%rax,%r10)
	addq	$16, %r10
	cmpl	%r11d, %ebx
	ja	L23
	movl	-12(%rsp), %ebx
	movl	-32(%rsp), %r11d
	subl	%ebx, %r9d
	addl	%ebx, %r13d
	cmpl	%r11d, %ebx
	movl	%r9d, %r12d
	je	L12
	leal	1(%r13), %r11d
	movl	%r13d, %r9d
	movl	(%rdi,%r11,4), %ebp
	leaq	(%rdi,%r9,4), %rbx
	leal	-227(%r13), %r10d
	movl	%ebp, %r9d
	andl	$2147483647, %ebp
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	(%rdi,%r10,4), %r9d
	movl	(%rbx), %r10d
	andl	$-2147483648, %r10d
	orl	%ebp, %r10d
	shrq	%r10
	xorl	%r10d, %r9d
	cmpl	$1, %r12d
	movl	%r9d, (%rbx)
	je	L12
	leal	2(%r13), %ebx
	movl	(%rdi,%rbx,4), %ebp
	leaq	(%rdi,%r11,4), %r11
	leal	-226(%r13), %r10d
	movl	%ebp, %r9d
	andl	$2147483647, %ebp
	andl	$1, %r9d
	negl	%r9d
	andl	$-1727483681, %r9d
	xorl	(%rdi,%r10,4), %r9d
	movl	(%r11), %r10d
	andl	$-2147483648, %r10d
	orl	%ebp, %r10d
	shrq	%r10
	xorl	%r10d, %r9d
	cmpl	$2, %r12d
	movl	%r9d, (%r11)
	je	L12
	leal	3(%r13), %r9d
	subl	$225, %r13d
	movl	(%rdi,%r9,4), %r10d
	leaq	(%rdi,%rbx,4), %r11
	movl	(%r11), %ebx
	movl	%r10d, %r9d
	andl	$2147483647, %r10d
	andl	$1, %r9d
	andl	$-2147483648, %ebx
	negl	%r9d
	orl	%ebx, %r10d
	andl	$-1727483681, %r9d
	shrq	%r10
	xorl	(%rdi,%r13,4), %r9d
	xorl	%r10d, %r9d
	movl	%r9d, (%r11)
L12:
	movl	(%rdi), %r9d
	movl	2492(%rdi), %r10d
	movl	%r9d, %r11d
	andl	$1, %r9d
	andl	$2147483647, %r11d
	andl	$-2147483648, %r10d
	negl	%r9d
	orl	%r11d, %r10d
	andl	$-1727483681, %r9d
	movl	$1, %r11d
	shrq	%r10
	xorl	1584(%rdi), %r10d
	xorl	%r9d, %r10d
	xorl	%r9d, %r9d
	movl	%r10d, 2492(%rdi)
	jmp	L5
	.align 4,0x90
L49:
	popq	%rbx
LCFI6:
	movl	%r15d, %eax
	popq	%rbp
LCFI7:
	popq	%r12
LCFI8:
	popq	%r13
LCFI9:
	popq	%r14
LCFI10:
	popq	%r15
LCFI11:
	ret
	.align 4,0x90
L34:
LCFI12:
	movl	$227, %ebx
	xorl	%r9d, %r9d
	jmp	L24
	.align 4,0x90
L28:
	vmovss	LC0(%rip), %xmm9
	vmovss	LC2(%rip), %xmm8
	jmp	L2
	.align 4,0x90
L31:
	movl	$396, %r9d
	movl	$227, %r13d
	jmp	L20
	.align 4,0x90
L33:
	movl	$2, -36(%rsp)
	movl	$394, %r12d
	movl	$229, %r10d
	jmp	L21
	.align 4,0x90
L36:
	movl	$225, %r13d
	movl	$2, %r11d
	jmp	L25
L32:
	movl	$0, -36(%rsp)
	movl	$396, %r12d
	movl	$227, %r10d
	jmp	L21
L35:
	movl	$227, %r13d
	xorl	%r11d, %r11d
	jmp	L25
LFE3748:
	.align 4,0x90
	.globl __Z4VPoiRKSt5arrayIfLm8EERN3vdt15MersenneTwisterE
__Z4VPoiRKSt5arrayIfLm8EERN3vdt15MersenneTwisterE:
LFB3749:
	pushq	%rbp
LCFI13:
	movq	%rdx, %r11
	movq	%rdi, %rax
	movq	%rsp, %rbp
LCFI14:
	pushq	%r13
	andl	$15, %r11d
	pushq	%r12
	shrq	$2, %r11
	pushq	%rbx
	negq	%r11
	andq	$-32, %rsp
	subq	$2440, %rsp
	andl	$3, %r11d
LCFI15:
	vmovups	(%rsi), %xmm1
	vmovaps	LC19(%rip), %xmm13
	vmovaps	LC21(%rip), %xmm2
	vmovaps	LC20(%rip), %xmm12
	vxorps	%xmm13, %xmm1, %xmm4
	vmovaps	%xmm4, %xmm0
	vmovaps	LC22(%rip), %xmm11
	vfmadd132ps	%xmm12, %xmm2, %xmm0
	vmovaps	LC23(%rip), %xmm10
	vcvttps2dq	%xmm0, %xmm14
	vpsrld	$31, %xmm0, %xmm0
	vmovaps	LC28(%rip), %xmm3
	vpsubd	%xmm0, %xmm14, %xmm14
	vmovaps	LC27(%rip), %xmm9
	vcvtdq2ps	%xmm14, %xmm0
	vfmsub231ps	%xmm11, %xmm0, %xmm1
	vfnmadd231ps	%xmm10, %xmm0, %xmm1
	vaddps	LC26(%rip), %xmm1, %xmm7
	vmulps	%xmm1, %xmm1, %xmm5
	vmovaps	%xmm1, %xmm0
	vmovaps	LC30(%rip), %xmm8
	vfmadd132ps	%xmm9, %xmm3, %xmm0
	vfmadd132ps	%xmm1, %xmm3, %xmm0
	vfmadd213ps	LC29(%rip), %xmm1, %xmm0
	vmovaps	LC25(%rip), %xmm6
	vmovaps	%xmm7, -104(%rsp)
	vmovaps	LC31(%rip), %xmm7
	vfmadd132ps	%xmm1, %xmm8, %xmm0
	vmovaps	LC24(%rip), %xmm15
	vfmadd132ps	%xmm1, %xmm7, %xmm0
	vfmadd132ps	%xmm1, %xmm2, %xmm0
	vmovdqa	LC32(%rip), %xmm1
	vfmadd213ps	-104(%rsp), %xmm0, %xmm5
	vpaddd	%xmm1, %xmm14, %xmm0
	vcmpltps	%xmm4, %xmm6, %xmm14
	vpslld	$23, %xmm0, %xmm0
	vmulps	%xmm0, %xmm5, %xmm5
	vcmpltps	%xmm15, %xmm4, %xmm4
	vmovaps	LC33(%rip), %xmm0
	vblendvps	%xmm14, %xmm0, %xmm5, %xmm5
	vandnps	%xmm5, %xmm4, %xmm4
	vmovaps	%xmm4, -88(%rsp)
	vmovups	16(%rsi), %xmm4
	vxorps	%xmm13, %xmm4, %xmm13
	vfmadd132ps	%xmm13, %xmm2, %xmm12
	vcvttps2dq	%xmm12, %xmm5
	vpsrld	$31, %xmm12, %xmm12
	vpsubd	%xmm12, %xmm5, %xmm5
	vcmpltps	%xmm13, %xmm6, %xmm6
	vcvtdq2ps	%xmm5, %xmm12
	vfmsub231ps	%xmm11, %xmm12, %xmm4
	vfnmadd132ps	%xmm12, %xmm4, %xmm10
	vmulps	%xmm10, %xmm10, %xmm4
	vaddps	LC26(%rip), %xmm10, %xmm11
	vpaddd	%xmm1, %xmm5, %xmm1
	vpslld	$23, %xmm1, %xmm1
	vfmadd132ps	%xmm10, %xmm3, %xmm9
	vfmadd132ps	%xmm10, %xmm3, %xmm9
	vfmadd213ps	LC29(%rip), %xmm10, %xmm9
	vfmadd132ps	%xmm10, %xmm8, %xmm9
	vfmadd132ps	%xmm10, %xmm7, %xmm9
	vfmadd132ps	%xmm10, %xmm2, %xmm9
	vcmpltps	%xmm15, %xmm13, %xmm13
	vfmadd132ps	%xmm9, %xmm11, %xmm4
	vmulps	%xmm1, %xmm4, %xmm4
	vblendvps	%xmm6, %xmm0, %xmm4, %xmm0
	vandnps	%xmm0, %xmm13, %xmm0
	vmovaps	%xmm0, -72(%rsp)
	je	L112
	cmpl	$3, %r11d
	jne	L95
	movl	4(%rdx), %esi
	movl	(%rdx), %edi
	movl	%esi, %ecx
	movl	%esi, %r8d
	andl	$-2147483648, %esi
	andl	$1, %ecx
	andl	$2147483647, %r8d
	andl	$-2147483648, %edi
	negl	%ecx
	orl	%r8d, %edi
	andl	$-1727483681, %ecx
	shrq	%rdi
	xorl	1588(%rdx), %ecx
	xorl	%edi, %ecx
	movl	8(%rdx), %edi
	movl	%ecx, (%rdx)
	movl	%edi, %ecx
	andl	$2147483647, %edi
	andl	$1, %ecx
	orl	%edi, %esi
	movl	$225, %edi
	negl	%ecx
	shrq	%rsi
	andl	$-1727483681, %ecx
	xorl	1592(%rdx), %ecx
	xorl	%esi, %ecx
	movl	$2, %esi
	movl	%ecx, 4(%rdx)
L86:
	leal	1(%rsi), %r10d
	movl	%esi, %ecx
	leaq	(%rdx,%rcx,4), %rbx
	movl	%r10d, %r9d
	movl	(%rdx,%r9,4), %r8d
	leal	397(%rsi), %r12d
	movl	%r8d, %ecx
	andl	$2147483647, %r8d
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r12,4), %ecx
	movl	%r8d, %r12d
	movl	(%rbx), %r8d
	andl	$-2147483648, %r8d
	orl	%r12d, %r8d
	shrq	%r8
	xorl	%r8d, %ecx
	cmpl	%r10d, %r11d
	movl	%ecx, (%rbx)
	leal	-1(%rdi), %ebx
	jbe	L87
	leal	2(%rsi), %r10d
	addl	$398, %esi
	leaq	(%rdx,%r9,4), %r9
	movl	%r10d, %r8d
	movl	(%rdx,%r8,4), %ebx
	movl	%ebx, %ecx
	andl	$2147483647, %ebx
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%rsi,4), %ecx
	movl	(%r9), %esi
	andl	$-2147483648, %esi
	orl	%ebx, %esi
	leal	-2(%rdi), %ebx
	shrq	%rsi
	xorl	%esi, %ecx
	cmpl	%r10d, %r11d
	movl	%ecx, (%r9)
	jbe	L87
	movl	12(%rdx), %esi
	leaq	(%rdx,%r8,4), %r8
	movl	$3, %r10d
	leal	-3(%rdi), %ebx
	movl	%esi, %ecx
	andl	$2147483647, %esi
	movl	%esi, %r9d
	andl	$1, %ecx
	movl	(%r8), %esi
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	1596(%rdx), %ecx
	andl	$-2147483648, %esi
	orl	%r9d, %esi
	shrq	%rsi
	xorl	%esi, %ecx
	movl	%ecx, (%r8)
L87:
	movl	$227, %r12d
	movl	%r11d, %ecx
	subl	%r11d, %r12d
L56:
	vmovdqa	LC15(%rip), %xmm4
	xorl	%esi, %esi
	vpxor	%xmm7, %xmm7, %xmm7
	leaq	1588(,%rcx,4), %rcx
	vmovdqa	LC16(%rip), %xmm3
	vmovdqa	LC17(%rip), %xmm2
	leaq	(%rdx,%rcx), %r9
	vmovdqa	LC18(%rip), %xmm1
	leaq	-1584(%rdx,%rcx), %r8
	leaq	-1588(%rdx,%rcx), %rdi
	xorl	%ecx, %ecx
	.align 4,0x90
L88:
	vmovdqu	(%r8,%rcx), %xmm0
	addl	$1, %esi
	vmovdqu	(%r9,%rcx), %xmm6
	vpand	%xmm4, %xmm0, %xmm5
	vpcmpeqd	%xmm7, %xmm5, %xmm5
	vpand	%xmm2, %xmm0, %xmm0
	vpandn	%xmm3, %xmm5, %xmm5
	vpxor	%xmm5, %xmm6, %xmm5
	vpand	(%rdi,%rcx), %xmm1, %xmm6
	vpor	%xmm6, %xmm0, %xmm0
	vpsrld	$1, %xmm0, %xmm0
	vpxor	%xmm0, %xmm5, %xmm0
	vmovdqa	%xmm0, (%rdi,%rcx)
	addq	$16, %rcx
	cmpl	$55, %esi
	jbe	L88
	leal	224(%r10), %ecx
	cmpl	$224, %r12d
	leal	-224(%rbx), %r8d
	je	L60
	leal	225(%r10), %edi
	movl	(%rdx,%rdi,4), %esi
	leaq	(%rdx,%rcx,4), %r9
	leal	621(%r10), %r12d
	movl	%esi, %ecx
	andl	$2147483647, %esi
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r12,4), %ecx
	movl	%esi, %r12d
	movl	(%r9), %esi
	andl	$-2147483648, %esi
	orl	%r12d, %esi
	shrq	%rsi
	xorl	%esi, %ecx
	cmpl	$225, %ebx
	movl	%ecx, (%r9)
	je	L60
	leal	226(%r10), %r9d
	movl	(%rdx,%r9,4), %esi
	leaq	(%rdx,%rdi,4), %rdi
	leal	622(%r10), %ebx
	movl	%esi, %ecx
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%rbx,4), %ecx
	movl	%esi, %ebx
	movl	(%rdi), %esi
	andl	$2147483647, %ebx
	andl	$-2147483648, %esi
	orl	%ebx, %esi
	shrq	%rsi
	xorl	%esi, %ecx
	cmpl	$2, %r8d
	movl	%ecx, (%rdi)
	je	L60
	leal	227(%r10), %ecx
	addl	$623, %r10d
	movl	(%rdx,%rcx,4), %esi
	leaq	(%rdx,%r9,4), %rdi
	movl	%esi, %ecx
	andl	$2147483647, %esi
	movl	%esi, %r8d
	andl	$1, %ecx
	movl	(%rdi), %esi
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r10,4), %ecx
	andl	$-2147483648, %esi
	orl	%r8d, %esi
	shrq	%rsi
	xorl	%esi, %ecx
	movl	%ecx, (%rdi)
L60:
	testl	%r11d, %r11d
	je	L92
	leal	-1(%r11), %r9d
	cmpl	$3, %r11d
	jne	L93
	movl	912(%rdx), %esi
	movl	908(%rdx), %edi
	movl	%esi, %ecx
	movl	%esi, %r8d
	andl	$-2147483648, %esi
	andl	$1, %ecx
	andl	$2147483647, %r8d
	andl	$-2147483648, %edi
	negl	%ecx
	orl	%r8d, %edi
	movl	916(%rdx), %r8d
	andl	$-1727483681, %ecx
	shrq	%rdi
	xorl	(%rdx), %ecx
	xorl	%edi, %ecx
	movl	%r8d, %edi
	movl	%ecx, 908(%rdx)
	movl	%r8d, %ecx
	andl	$2147483647, %edi
	andl	$1, %ecx
	orl	%edi, %esi
	negl	%ecx
	shrq	%rsi
	andl	$-1727483681, %ecx
	xorl	4(%rdx), %ecx
	xorl	%esi, %ecx
	cmpl	$3, %r9d
	movl	%ecx, 912(%rdx)
	jbe	L94
	movl	920(%rdx), %esi
	andl	$-2147483648, %r8d
	movl	$6, %r9d
	movl	%esi, %ecx
	movl	%esi, %edi
	andl	$-2147483648, %esi
	andl	$1, %ecx
	andl	$2147483647, %edi
	negl	%ecx
	orl	%r8d, %edi
	movl	924(%rdx), %r8d
	andl	$-1727483681, %ecx
	shrq	%rdi
	xorl	8(%rdx), %ecx
	xorl	%edi, %ecx
	movl	%r8d, %edi
	movl	%ecx, 916(%rdx)
	movl	%r8d, %ecx
	andl	$2147483647, %edi
	andl	$1, %ecx
	orl	%edi, %esi
	andl	$-2147483648, %r8d
	negl	%ecx
	shrq	%rsi
	andl	$-1727483681, %ecx
	xorl	12(%rdx), %ecx
	xorl	%esi, %ecx
	movl	928(%rdx), %esi
	movl	%ecx, 920(%rdx)
	movl	%esi, %ecx
	movl	%esi, %edi
	andl	$-2147483648, %esi
	andl	$1, %ecx
	andl	$2147483647, %edi
	negl	%ecx
	orl	%r8d, %edi
	andl	$-1727483681, %ecx
	shrq	%rdi
	xorl	16(%rdx), %ecx
	xorl	%edi, %ecx
	movl	932(%rdx), %edi
	movl	%ecx, 924(%rdx)
	movl	%edi, %ecx
	andl	$2147483647, %edi
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	20(%rdx), %ecx
	orl	%edi, %esi
	movl	$390, %edi
	shrq	%rsi
	xorl	%esi, %ecx
	movl	$233, %esi
	movl	%ecx, 928(%rdx)
L83:
	leal	1(%rsi), %ebx
	movl	%esi, %ecx
	leaq	(%rdx,%rcx,4), %r10
	movl	%ebx, %r12d
	movl	(%rdx,%r12,4), %r8d
	leal	-227(%rsi), %r13d
	movl	%r8d, %ecx
	andl	$2147483647, %r8d
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r13,4), %ecx
	movl	%r8d, %r13d
	movl	(%r10), %r8d
	andl	$-2147483648, %r8d
	orl	%r13d, %r8d
	leal	-1(%rdi), %r13d
	shrq	%r8
	xorl	%r8d, %ecx
	movl	%ecx, (%r10)
	leal	1(%r9), %ecx
	cmpl	%ecx, %r11d
	jbe	L84
	leal	2(%rsi), %ebx
	addl	$2, %r9d
	leaq	(%rdx,%r12,4), %r12
	movl	%ebx, %r10d
	movl	(%rdx,%r10,4), %r8d
	leal	-226(%rsi), %r13d
	movl	%r8d, %ecx
	andl	$2147483647, %r8d
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r13,4), %ecx
	movl	%r8d, %r13d
	movl	(%r12), %r8d
	andl	$-2147483648, %r8d
	orl	%r13d, %r8d
	leal	-2(%rdi), %r13d
	shrq	%r8
	xorl	%r8d, %ecx
	cmpl	%r9d, %r11d
	movl	%ecx, (%r12)
	jbe	L84
	leal	3(%rsi), %ebx
	subl	$225, %esi
	leaq	(%rdx,%r10,4), %r8
	movl	%ebx, %ecx
	movl	(%rdx,%rcx,4), %r9d
	leal	-3(%rdi), %r13d
	movl	%r9d, %ecx
	andl	$2147483647, %r9d
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%rsi,4), %ecx
	movl	(%r8), %esi
	andl	$-2147483648, %esi
	orl	%r9d, %esi
	shrq	%rsi
	xorl	%esi, %ecx
	movl	%ecx, (%r8)
L84:
	movl	$396, %r12d
	movl	%r11d, %ecx
	subl	%r11d, %r12d
	movl	%r12d, %r8d
	shrl	$2, %r8d
	leal	0(,%r8,4), %r11d
L82:
	salq	$2, %rcx
	xorl	%esi, %esi
	vpxor	%xmm7, %xmm7, %xmm7
	leaq	(%rdx,%rcx), %r10
	leaq	912(%rdx,%rcx), %r9
	leaq	908(%rdx,%rcx), %rdi
	xorl	%ecx, %ecx
	.align 4,0x90
L85:
	vmovdqa	(%r9,%rcx), %xmm0
	addl	$1, %esi
	vmovdqu	(%rdi,%rcx), %xmm6
	vpand	%xmm4, %xmm0, %xmm5
	vpcmpeqd	%xmm7, %xmm5, %xmm5
	vpand	%xmm2, %xmm0, %xmm0
	vpand	%xmm1, %xmm6, %xmm6
	vpor	%xmm0, %xmm6, %xmm0
	vpsrld	$1, %xmm0, %xmm0
	vpandn	%xmm3, %xmm5, %xmm5
	vpxor	(%r10,%rcx), %xmm5, %xmm5
	vpxor	%xmm0, %xmm5, %xmm0
	vmovdqu	%xmm0, (%rdi,%rcx)
	addq	$16, %rcx
	cmpl	%esi, %r8d
	ja	L85
	leal	(%rbx,%r11), %esi
	subl	%r11d, %r13d
	cmpl	%r12d, %r11d
	je	L64
	leal	1(%rsi), %r8d
	movl	%esi, %ecx
	movl	(%rdx,%r8,4), %edi
	leaq	(%rdx,%rcx,4), %r9
	leal	-227(%rsi), %r10d
	movl	%edi, %ecx
	andl	$2147483647, %edi
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r10,4), %ecx
	movl	%edi, %r10d
	movl	(%r9), %edi
	andl	$-2147483648, %edi
	orl	%r10d, %edi
	shrq	%rdi
	xorl	%edi, %ecx
	cmpl	$1, %r13d
	movl	%ecx, (%r9)
	je	L64
	leal	2(%rsi), %r9d
	movl	(%rdx,%r9,4), %edi
	leaq	(%rdx,%r8,4), %r8
	leal	-226(%rsi), %r10d
	movl	%edi, %ecx
	andl	$2147483647, %edi
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%r10,4), %ecx
	movl	%edi, %r10d
	movl	(%r8), %edi
	andl	$-2147483648, %edi
	orl	%r10d, %edi
	shrq	%rdi
	xorl	%edi, %ecx
	cmpl	$2, %r13d
	movl	%ecx, (%r8)
	je	L64
	leal	3(%rsi), %ecx
	subl	$225, %esi
	movl	(%rdx,%rcx,4), %r8d
	leaq	(%rdx,%r9,4), %rdi
	movl	%r8d, %ecx
	andl	$2147483647, %r8d
	andl	$1, %ecx
	negl	%ecx
	andl	$-1727483681, %ecx
	xorl	(%rdx,%rsi,4), %ecx
	movl	(%rdi), %esi
	andl	$-2147483648, %esi
	orl	%r8d, %esi
	shrq	%rsi
	xorl	%esi, %ecx
	movl	%ecx, (%rdi)
L64:
	movl	(%rdx), %ecx
	leaq	-56(%rsp), %r10
	movl	$0, 2496(%rdx)
	movl	2492(%rdx), %esi
	vmovdqa	LC34(%rip), %xmm4
	vmovdqa	LC35(%rip), %xmm3
	movl	%ecx, %edi
	andl	$1, %ecx
	andl	$2147483647, %edi
	andl	$-2147483648, %esi
	negl	%ecx
	orl	%edi, %esi
	andl	$-1727483681, %ecx
	shrq	%rsi
	xorl	1584(%rdx), %esi
	xorl	%ecx, %esi
	xorl	%ecx, %ecx
	movl	%esi, 2492(%rdx)
	.align 4,0x90
L67:
	vmovdqu	(%rdx,%rcx), %xmm0
	addq	$16, %rcx
	vpsrld	$11, %xmm0, %xmm2
	vpxor	%xmm0, %xmm2, %xmm2
	vpslld	$7, %xmm2, %xmm1
	vpand	%xmm4, %xmm1, %xmm1
	vpxor	%xmm2, %xmm1, %xmm1
	vpslld	$15, %xmm1, %xmm0
	vpand	%xmm3, %xmm0, %xmm0
	vpxor	%xmm1, %xmm0, %xmm0
	vpsrld	$18, %xmm0, %xmm1
	vpxor	%xmm0, %xmm1, %xmm0
	vmovdqa	%xmm0, -16(%rcx,%r10)
	cmpq	$2496, %rcx
	jne	L67
	vmovdqa	-40(%rsp), %xmm0
	leaq	32(%r10), %rcx
	xorl	%edx, %edx
	vmovdqa	-56(%rsp), %xmm1
	jmp	L70
	.align 4,0x90
L73:
	movq	%rdi, %rcx
L70:
	addl	$2, %edx
	vpmulld	(%rcx), %xmm1, %xmm1
	vmovdqa	%xmm1, (%rcx)
	vpmulld	16(%rcx), %xmm0, %xmm0
	leaq	32(%rcx), %rdi
	vmovdqa	%xmm0, 16(%rcx)
	cmpl	$152, %edx
	jne	L73
	movl	$152, %esi
	xorl	%edx, %edx
	.align 4,0x90
L72:
	vmovdqa	(%rcx,%rdx), %xmm0
	addl	$1, %esi
	vpmulld	(%rdi,%rdx), %xmm0, %xmm0
	vmovdqa	%xmm0, (%rdi,%rdx)
	addq	$16, %rdx
	cmpl	$153, %esi
	je	L72
	leaq	2464(%r10), %r11
	xorl	%edi, %edi
	.align 4,0x90
L81:
	leaq	(%r11,%rdi), %rdx
	movq	%r10, %rsi
	subq	%r10, %rdx
	sarq	$2, %rdx
	testq	%rdx, %rdx
	jle	L91
	vmovss	-88(%rsp,%rdi), %xmm1
	.align 4,0x90
L79:
	movq	%rdx, %rcx
	sarq	%rcx
	leaq	(%rsi,%rcx,4), %r8
	subq	%rcx, %rdx
	movl	(%r8), %r9d
	subq	$1, %rdx
	addq	$4, %r8
	vcvtsi2ssq	%r9, %xmm0, %xmm0
	vcomiss	%xmm1, %xmm0
	cmovbe	%rcx, %rdx
	cmova	%r8, %rsi
	testq	%rdx, %rdx
	jg	L79
	subq	%r10, %rsi
	shrq	$2, %rsi
L75:
	movl	%esi, (%rdi,%rax)
	addq	$4, %rdi
	addq	$4, %r10
	cmpq	$32, %rdi
	jne	L81
	leaq	-24(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
LCFI16:
	ret
L91:
LCFI17:
	xorl	%esi, %esi
	jmp	L75
L92:
	movl	$227, %ebx
	movl	$396, %r13d
	xorl	%ecx, %ecx
	movl	$396, %r12d
	movl	$99, %r8d
	movl	$396, %r11d
	jmp	L82
L112:
	xorl	%r10d, %r10d
	movl	$227, %ebx
	xorl	%ecx, %ecx
	movl	$227, %r12d
	jmp	L56
L94:
	movl	$2, %r9d
	movl	$394, %edi
	movl	$229, %esi
	jmp	L83
L95:
	movl	$227, %edi
	xorl	%esi, %esi
	jmp	L86
L93:
	xorl	%r9d, %r9d
	movl	$396, %edi
	movl	$227, %esi
	jmp	L83
LFE3749:
	.cstring
LC42:
	.ascii " \0"
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB4181:
	pushq	%r13
LCFI18:
	movl	$5489, %edi
	movl	$1, %eax
	pushq	%r12
LCFI19:
	pushq	%rbp
LCFI20:
	pushq	%rbx
LCFI21:
	subq	$5048, %rsp
LCFI22:
	leaq	32(%rsp), %rbx
	movl	$0x3f000000, (%rsp)
	movl	$0x3f800000, 4(%rsp)
	leaq	4(%rbx), %rcx
	movl	$0x40400000, 8(%rsp)
	movl	$0x40800000, 12(%rsp)
	movl	$0x40a00000, 16(%rsp)
	movl	$0x40c00000, 20(%rsp)
	movl	$0x41000000, 24(%rsp)
	movl	$0x41200000, 28(%rsp)
	movl	$5489, 32(%rsp)
	.align 4
L114:
	movl	%edi, %edx
	addq	$4, %rcx
	shrl	$30, %edx
	xorl	%edi, %edx
	imull	$1812433253, %edx, %edi
	addl	%eax, %edi
	addl	$1, %eax
	movl	%edi, -4(%rcx)
	cmpl	$624, %eax
	jne	L114
	leaq	2528(%rsp), %r12
	movl	$2496, %edx
	movq	%rbx, %rsi
	movq	%r12, %rdi
	xorl	%ebp, %ebp
	movq	%rsp, %r13
	call	_memcpy
	movl	$624, 5024(%rsp)
	.align 4
L120:
	vmovss	0(%r13,%rbp), %xmm0
	movq	%r12, %rdi
	call	__Z8knuthPoifRN3vdt15MersenneTwisterE
	movl	%eax, (%rbx,%rbp)
	addq	$4, %rbp
	cmpq	$32, %rbp
	jne	L120
	movq	__ZSt4cout@GOTPCREL(%rip), %rbp
	leaq	32(%rbx), %r12
	.align 4
L119:
	movl	(%rbx), %esi
	movq	%rbp, %rdi
	addq	$4, %rbx
	call	__ZNSolsEi
	movl	$1, %edx
	leaq	LC42(%rip), %rsi
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	cmpq	%rbx, %r12
	jne	L119
	movq	%rbp, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	addq	$5048, %rsp
LCFI23:
	xorl	%eax, %eax
	popq	%rbx
LCFI24:
	popq	%rbp
LCFI25:
	popq	%r12
LCFI26:
	popq	%r13
LCFI27:
	ret
LFE4181:
	.align 4
__GLOBAL__sub_I_poiss.cpp:
LFB4467:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI28:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	addq	$8, %rsp
LCFI29:
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE4467:
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal4
	.align 2
LC0:
	.long	2139095040
	.align 2
LC2:
	.long	1065353216
	.literal16
	.align 4
LC3:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.literal4
	.align 2
LC4:
	.long	1118925336
	.align 2
LC5:
	.long	1069066811
	.align 2
LC6:
	.long	1056964608
	.align 2
LC7:
	.long	962494595
	.align 2
LC8:
	.long	1060208640
	.align 2
LC9:
	.long	961571175
	.align 2
LC10:
	.long	985088974
	.align 2
LC11:
	.long	1007192328
	.align 2
LC12:
	.long	1026206145
	.align 2
LC13:
	.long	1042983594
	.align 2
LC14:
	.long	3266314240
	.literal16
	.align 4
LC15:
	.long	1
	.long	1
	.long	1
	.long	1
	.align 4
LC16:
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.long	-1727483681
	.align 4
LC17:
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.align 4
LC18:
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.long	-2147483648
	.align 4
LC19:
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.align 4
LC20:
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.align 4
LC21:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.align 4
LC22:
	.long	962494595
	.long	962494595
	.long	962494595
	.long	962494595
	.align 4
LC23:
	.long	1060208640
	.long	1060208640
	.long	1060208640
	.long	1060208640
	.align 4
LC24:
	.long	3266314240
	.long	3266314240
	.long	3266314240
	.long	3266314240
	.align 4
LC25:
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.align 4
LC26:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 4
LC27:
	.long	961571175
	.long	961571175
	.long	961571175
	.long	961571175
	.align 4
LC28:
	.long	985088974
	.long	985088974
	.long	985088974
	.long	985088974
	.align 4
LC29:
	.long	1007192328
	.long	1007192328
	.long	1007192328
	.long	1007192328
	.align 4
LC30:
	.long	1026206145
	.long	1026206145
	.long	1026206145
	.long	1026206145
	.align 4
LC31:
	.long	1042983594
	.long	1042983594
	.long	1042983594
	.long	1042983594
	.align 4
LC32:
	.long	127
	.long	127
	.long	127
	.long	127
	.align 4
LC33:
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.align 4
LC34:
	.long	-1658038656
	.long	-1658038656
	.long	-1658038656
	.long	-1658038656
	.align 4
LC35:
	.long	-272236544
	.long	-272236544
	.long	-272236544
	.long	-272236544
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
	.quad	LFB3748-.
	.set L$set$2,LFE3748-LFB3748
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB3748
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$8,LCFI5-LCFI4
	.long L$set$8
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$9,LCFI6-LCFI5
	.long L$set$9
	.byte	0xa
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$10,LCFI7-LCFI6
	.long L$set$10
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$11,LCFI8-LCFI7
	.long L$set$11
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$12,LCFI9-LCFI8
	.long L$set$12
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$13,LCFI10-LCFI9
	.long L$set$13
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$14,LCFI11-LCFI10
	.long L$set$14
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$15,LCFI12-LCFI11
	.long L$set$15
	.byte	0xb
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$16,LEFDE3-LASFDE3
	.long L$set$16
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB3749-.
	.set L$set$17,LFE3749-LFB3749
	.quad L$set$17
	.byte	0
	.byte	0x4
	.set L$set$18,LCFI13-LFB3749
	.long L$set$18
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$19,LCFI14-LCFI13
	.long L$set$19
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$20,LCFI15-LCFI14
	.long L$set$20
	.byte	0x8d
	.byte	0x3
	.byte	0x8c
	.byte	0x4
	.byte	0x83
	.byte	0x5
	.byte	0x4
	.set L$set$21,LCFI16-LCFI15
	.long L$set$21
	.byte	0xa
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$22,LCFI17-LCFI16
	.long L$set$22
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$23,LEFDE5-LASFDE5
	.long L$set$23
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB4181-.
	.set L$set$24,LFE4181-LFB4181
	.quad L$set$24
	.byte	0
	.byte	0x4
	.set L$set$25,LCFI18-LFB4181
	.long L$set$25
	.byte	0xe
	.byte	0x10
	.byte	0x8d
	.byte	0x2
	.byte	0x4
	.set L$set$26,LCFI19-LCFI18
	.long L$set$26
	.byte	0xe
	.byte	0x18
	.byte	0x8c
	.byte	0x3
	.byte	0x4
	.set L$set$27,LCFI20-LCFI19
	.long L$set$27
	.byte	0xe
	.byte	0x20
	.byte	0x86
	.byte	0x4
	.byte	0x4
	.set L$set$28,LCFI21-LCFI20
	.long L$set$28
	.byte	0xe
	.byte	0x28
	.byte	0x83
	.byte	0x5
	.byte	0x4
	.set L$set$29,LCFI22-LCFI21
	.long L$set$29
	.byte	0xe
	.byte	0xe0,0x27
	.byte	0x4
	.set L$set$30,LCFI23-LCFI22
	.long L$set$30
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$31,LCFI24-LCFI23
	.long L$set$31
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$32,LCFI25-LCFI24
	.long L$set$32
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$33,LCFI26-LCFI25
	.long L$set$33
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$34,LCFI27-LCFI26
	.long L$set$34
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$35,LEFDE7-LASFDE7
	.long L$set$35
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB4467-.
	.set L$set$36,LFE4467-LFB4467
	.quad L$set$36
	.byte	0
	.byte	0x4
	.set L$set$37,LCFI28-LFB4467
	.long L$set$37
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$38,LCFI29-LCFI28
	.long L$set$38
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE7:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_poiss.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
