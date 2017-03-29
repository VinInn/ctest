	.file	"gather.cpp"
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB1280:
	.cfi_startproc
	movl	%esi, %eax
	ret
	.cfi_endproc
.LFE1280:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC14:
	.string	"for dis "
	.text
	.p2align 4,,15
	.type	_Z5time3PK6float3iiii.constprop.3, @function
_Z5time3PK6float3iiii.constprop.3:
.LFB2610:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	leal	-1(%rsi), %eax
	movl	%esi, %r8d
	pushq	-8(%r10)
	andl	$-8, %r8d
	movl	%eax, %ecx
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	movq	%rdi, %r15
	pushq	%r14
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	movq	%rdi, %r14
	pushq	%r13
	pushq	%r12
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	movl	%esi, %r12d
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	shrl	$3, %r12d
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	%esi, %ebx
	movslq	%eax, %rsi
	subq	$96, %rsp
	movl	%eax, -80(%rbp)
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	movq	%rdi, -88(%rbp)
	leaq	1200000(%rdi), %rdi
	.p2align 4,,10
	.p2align 3
.L8:
	vmovss	(%r14), %xmm4
	vmovss	4(%r14), %xmm9
	vmovss	8(%r14), %xmm11
	testl	%ebx, %ebx
	jle	.L4
	cmpl	$6, %ecx
	jbe	.L33
	vbroadcastss	%xmm4, %ymm8
	vbroadcastss	%xmm9, %ymm7
	xorl	%eax, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm11, %ymm5
	.p2align 4,,10
	.p2align 3
.L6:
	vmovaps	%ymm6, %ymm3
	vmovaps	%ymm6, %ymm1
	vmovaps	%ymm6, %ymm12
	addl	$1, %edx
	vpmulld	neighList(%rax), %ymm10, %ymm0
	addq	$32, %rax
	vgatherdps	%ymm3, position(,%ymm0,4), %ymm2
	vgatherdps	%ymm1, position+4(,%ymm0,4), %ymm3
	vsubps	%ymm3, %ymm7, %ymm3
	vsubps	%ymm2, %ymm8, %ymm2
	vgatherdps	%ymm12, position+8(,%ymm0,4), %ymm1
	vsubps	%ymm1, %ymm5, %ymm0
	vmulps	%ymm3, %ymm3, %ymm3
	vfmadd132ps	%ymm2, %ymm3, %ymm2
	vfmadd132ps	%ymm0, %ymm2, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edx, %r12d
	ja	.L6
	cmpl	%ebx, %r8d
	je	.L4
	movl	%r8d, %eax
.L5:
	movslq	%eax, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	1(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L4
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	2(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L4
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	3(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L4
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	4(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L4
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	5(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L4
	movslq	%r9d, %r9
	addl	$6, %eax
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	cmpl	%eax, %ebx
	jle	.L4
	cltq
	movl	neighList(,%rax,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm4
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm11
	vsubss	position(,%rdx,4), %xmm9, %xmm9
	vmulss	%xmm11, %xmm11, %xmm11
	vfmadd132ss	%xmm9, %xmm11, %xmm9
	vfmadd132ss	%xmm4, %xmm9, %xmm4
	vmovss	%xmm4, r2inv(,%rax,4)
.L4:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r14
	vmovss	%xmm0, res(%rip)
	cmpq	%rdi, %r14
	jne	.L8
	movl	%r8d, -104(%rbp)
	movq	%rsi, -72(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-88(%rbp), %r13
	movl	-80(%rbp), %edx
	movq	%rax, -96(%rbp)
	vmovdqa	.LC2(%rip), %ymm15
	vmovdqa	.LC3(%rip), %ymm14
	vmovdqa	.LC5(%rip), %ymm13
	vmovdqa	.LC9(%rip), %ymm12
	movl	-104(%rbp), %r8d
	movq	-72(%rbp), %rsi
	.p2align 4,,10
	.p2align 3
.L13:
	vmovss	8(%r13), %xmm5
	movl	0(%r13), %ecx
	movl	4(%r13), %edi
	vmovss	%xmm5, -72(%rbp)
	testl	%ebx, %ebx
	jle	.L9
	cmpl	$6, %edx
	jbe	.L34
	vmovd	%ecx, %xmm7
	vmovdqa	.LC12(%rip), %ymm6
	movl	$position, %eax
	xorl	%r9d, %r9d
	vbroadcastss	%xmm7, %ymm9
	vmovd	%edi, %xmm7
	movl	$r2inv, %r10d
	vbroadcastss	%xmm7, %ymm8
	vbroadcastss	%xmm5, %ymm7
	vmovdqa	.LC13(%rip), %ymm5
	.p2align 4,,10
	.p2align 3
.L11:
	vmovaps	(%rax), %ymm0
	addl	$1, %r9d
	addq	$32, %r10
	addq	$96, %rax
	vmovaps	-64(%rax), %ymm10
	vmovaps	-32(%rax), %ymm4
	vpshufb	.LC7(%rip), %ymm0, %ymm11
	vpshufb	%ymm15, %ymm0, %ymm3
	vpshufb	.LC4(%rip), %ymm10, %ymm2
	vpermq	$78, %ymm3, %ymm1
	vpshufb	%ymm14, %ymm0, %ymm3
	vpor	%ymm1, %ymm3, %ymm3
	vpshufb	.LC8(%rip), %ymm10, %ymm1
	vpshufb	%ymm6, %ymm10, %ymm10
	vpor	%ymm2, %ymm3, %ymm3
	vpermd	%ymm4, %ymm13, %ymm2
	vblendps	$192, %ymm2, %ymm3, %ymm3
	vpshufb	.LC6(%rip), %ymm0, %ymm2
	vpermq	$78, %ymm2, %ymm2
	vsubps	%ymm3, %ymm9, %ymm3
	vpor	%ymm2, %ymm11, %ymm2
	vpor	%ymm1, %ymm2, %ymm2
	vpermd	%ymm4, %ymm12, %ymm1
	vblendps	$224, %ymm1, %ymm2, %ymm2
	vpshufb	.LC10(%rip), %ymm0, %ymm1
	vpshufb	.LC11(%rip), %ymm0, %ymm0
	vsubps	%ymm2, %ymm8, %ymm2
	vpermq	$78, %ymm1, %ymm1
	vpor	%ymm1, %ymm0, %ymm1
	vpermd	%ymm4, %ymm5, %ymm0
	vpor	%ymm10, %ymm1, %ymm1
	vmulps	%ymm2, %ymm2, %ymm2
	vblendps	$224, %ymm0, %ymm1, %ymm1
	vsubps	%ymm1, %ymm7, %ymm0
	vfmadd132ps	%ymm3, %ymm2, %ymm3
	vfmadd132ps	%ymm0, %ymm3, %ymm0
	vmovaps	%ymm0, -32(%r10)
	cmpl	%r9d, %r12d
	ja	.L11
	cmpl	%ebx, %r8d
	je	.L9
	vmovss	-72(%rbp), %xmm7
	movl	%r8d, %eax
.L10:
	movslq	%eax, %r9
	vmovd	%ecx, %xmm5
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vmovd	%edi, %xmm5
	vsubss	4(%r10), %xmm5, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	1(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L9
	movslq	%r9d, %r9
	vmovd	%edi, %xmm6
	vmovd	%ecx, %xmm5
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	2(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L9
	movslq	%r9d, %r9
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	3(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L9
	movslq	%r9d, %r9
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vmovd	%edi, %xmm5
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	4(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L9
	movslq	%r9d, %r9
	vmovd	%ecx, %xmm6
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm6, %xmm1
	vmovd	%edi, %xmm6
	vsubss	4(%r10), %xmm5, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	5(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L9
	movslq	%r9d, %r9
	vmovd	%ecx, %xmm5
	addl	$6, %eax
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vmovd	%edi, %xmm5
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	cmpl	%eax, %ebx
	jle	.L9
	cltq
	vmovd	%ecx, %xmm6
	leaq	(%rax,%rax,2), %r10
	leaq	position(,%r10,4), %r9
	vsubss	position(,%r10,4), %xmm6, %xmm1
	vsubss	4(%r9), %xmm5, %xmm2
	vsubss	8(%r9), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rax,4)
.L9:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r13
	vmovss	%xmm0, res(%rip)
	cmpq	%r14, %r13
	jne	.L13
	movl	%r8d, -112(%rbp)
	movq	%rsi, -104(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	-96(%rbp), %rax
	movabsq	$2361183241434822607, %rdx
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -72(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-88(%rbp), %rdx
	vmovaps	.LC1(%rip), %ymm6
	movl	-112(%rbp), %r8d
	movl	-80(%rbp), %edi
	movq	%rax, %r14
	movq	-104(%rbp), %rsi
	.p2align 4,,10
	.p2align 3
.L18:
	vmovss	(%rdx), %xmm7
	vmovss	4(%rdx), %xmm4
	testl	%ebx, %ebx
	jle	.L14
	cmpl	$6, %edi
	jbe	.L35
	vbroadcastss	%xmm7, %ymm8
	vbroadcastss	%xmm4, %ymm5
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L16:
	vmovaps	%ymm6, %ymm2
	vmovaps	%ymm6, %ymm0
	vmovaps	%ymm6, %ymm13
	addl	$1, %ecx
	vmovdqa	neighList(%rax), %ymm3
	addq	$32, %rax
	vgatherdps	%ymm2, fx(,%ymm3,4), %ymm1
	vgatherdps	%ymm0, fy(,%ymm3,4), %ymm2
	vgatherdps	%ymm13, fz(,%ymm3,4), %ymm0
	vsubps	%ymm2, %ymm5, %ymm2
	vsubps	%ymm1, %ymm8, %ymm1
	vsubps	%ymm0, %ymm5, %ymm0
	vmulps	%ymm2, %ymm2, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%ecx, %r12d
	ja	.L16
	cmpl	%ebx, %r8d
	je	.L14
	movl	%r8d, %eax
.L15:
	movslq	%eax, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	1(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L14
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L14
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L14
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L14
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L14
	movslq	%ecx, %rcx
	addl	$6, %eax
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %ebx
	jle	.L14
	cltq
	movslq	neighList(,%rax,4), %rcx
	vsubss	fy(,%rcx,4), %xmm4, %xmm0
	vsubss	fx(,%rcx,4), %xmm7, %xmm7
	vsubss	fz(,%rcx,4), %xmm4, %xmm4
	vmulss	%xmm0, %xmm0, %xmm0
	vfmadd132ss	%xmm7, %xmm0, %xmm7
	vfmadd132ss	%xmm4, %xmm7, %xmm4
	vmovss	%xmm4, r2inv(,%rax,4)
.L14:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %rdx
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdx
	jne	.L18
	movl	%r8d, -112(%rbp)
	movq	%rsi, -104(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -96(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-88(%rbp), %r9
	movl	-112(%rbp), %r8d
	movq	%rax, %r14
	movl	-80(%rbp), %eax
	movq	-104(%rbp), %rsi
	leaq	4(,%rax,4), %rcx
	.p2align 4,,10
	.p2align 3
.L21:
	vmovss	(%r9), %xmm4
	vmovss	4(%r9), %xmm3
	testl	%ebx, %ebx
	jle	.L19
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L20:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rdi
	leaq	position(,%rdi,4), %rdx
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L20
.L19:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r9
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %r9
	jne	.L21
	movl	%r8d, -124(%rbp)
	movq	%rcx, -120(%rbp)
	movq	%rsi, -112(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rdi
	imulq	%rdx
	movq	%rdi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -104(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-88(%rbp), %rdx
	movl	-124(%rbp), %r8d
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	movq	%rax, %r14
	movl	-80(%rbp), %r9d
	movq	-120(%rbp), %rcx
	movq	-112(%rbp), %rsi
	.p2align 4,,10
	.p2align 3
.L26:
	vmovss	(%rdx), %xmm3
	vmovss	4(%rdx), %xmm4
	vmovss	8(%rdx), %xmm5
	testl	%ebx, %ebx
	jle	.L22
	cmpl	$6, %r9d
	jbe	.L36
	vbroadcastss	%xmm3, %ymm9
	vbroadcastss	%xmm4, %ymm8
	xorl	%eax, %eax
	xorl	%edi, %edi
	vbroadcastss	%xmm5, %ymm7
	.p2align 4,,10
	.p2align 3
.L24:
	vmovaps	%ymm6, %ymm2
	vmovaps	%ymm6, %ymm0
	vmovaps	%ymm6, %ymm14
	addl	$1, %edi
	vpmulld	neighList(%rax), %ymm10, %ymm11
	addq	$32, %rax
	vgatherdps	%ymm2, position(,%ymm11,4), %ymm1
	vgatherdps	%ymm0, position+4(,%ymm11,4), %ymm2
	vsubps	%ymm2, %ymm8, %ymm2
	vsubps	%ymm1, %ymm9, %ymm1
	vgatherdps	%ymm14, position+8(,%ymm11,4), %ymm0
	vsubps	%ymm0, %ymm7, %ymm0
	vmulps	%ymm2, %ymm2, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edi, %r12d
	ja	.L24
	cmpl	%ebx, %r8d
	je	.L22
	movl	%r8d, %eax
.L23:
	movslq	%eax, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	1(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L22
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	2(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L22
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	3(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L22
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	4(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L22
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	5(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L22
	movslq	%r10d, %r10
	addl	$6, %eax
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	cmpl	%eax, %ebx
	jle	.L22
	cltq
	movl	neighList(,%rax,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm3
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm5
	vsubss	position(,%rdi,4), %xmm4, %xmm4
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, r2inv(,%rax,4)
.L22:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %rdx
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdx
	jne	.L26
	movq	%rcx, -88(%rbp)
	movq	%rsi, -80(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rdi
	imulq	%rdx
	movq	%rdi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r12
	subq	%rax, %r12
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-88(%rbp), %rcx
	movq	-80(%rbp), %rsi
	movq	%rax, %r14
	.p2align 4,,10
	.p2align 3
.L29:
	vmovss	(%r15), %xmm3
	vmovss	4(%r15), %xmm4
	vmovss	8(%r15), %xmm5
	testl	%ebx, %ebx
	jle	.L27
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L28:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rdi
	leaq	position(,%rdi,4), %rdx
	vsubss	position(,%rdi,4), %xmm3, %xmm1
	vsubss	4(%rdx), %xmm4, %xmm2
	vsubss	8(%rdx), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L28
.L27:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r15
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %r15
	jne	.L29
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movl	$.LC14, %esi
	movl	$_ZSt4cout, %edi
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r13
	movl	$8, %edx
	subq	%rax, %r13
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%ebx, %esi
	movl	$_ZSt4cout, %edi
	call	_ZNSolsEi
	movl	$1, %edx
	leaq	-49(%rbp), %rsi
	movb	$32, -49(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-72(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-50(%rbp), %rsi
	movb	$32, -50(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-96(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-51(%rbp), %rsi
	movb	$32, -51(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-104(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-52(%rbp), %rsi
	movb	$32, -52(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-53(%rbp), %rsi
	movb	$32, -53(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r13, %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movq	%rax, %r12
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r12,%rax), %rbx
	testq	%rbx, %rbx
	je	.L52
	cmpb	$0, 56(%rbx)
	je	.L31
	movsbl	67(%rbx), %esi
.L32:
	movq	%r12, %rdi
	call	_ZNSo3putEc
	movq	%rax, %rdi
	call	_ZNSo5flushEv
	addq	$96, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L33:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L5
	.p2align 4,,10
	.p2align 3
.L36:
	xorl	%eax, %eax
	jmp	.L23
	.p2align 4,,10
	.p2align 3
.L35:
	xorl	%eax, %eax
	jmp	.L15
	.p2align 4,,10
	.p2align 3
.L34:
	xorl	%eax, %eax
	vmovaps	%xmm5, %xmm7
	jmp	.L10
	.p2align 4,,10
	.p2align 3
.L31:
	movq	%rbx, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	$_ZNKSt5ctypeIcE8do_widenEc, %rax
	je	.L32
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L32
.L52:
	call	_ZSt16__throw_bad_castv
	.cfi_endproc
.LFE2610:
	.size	_Z5time3PK6float3iiii.constprop.3, .-_Z5time3PK6float3iiii.constprop.3
	.p2align 4,,15
	.globl	_Z3f20v
	.type	_Z3f20v, @function
_Z3f20v:
.LFB0:
	.cfi_startproc
	vmovaps	.LC1(%rip), %ymm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L54:
	vmovaps	%ymm2, %ymm5
	vmovaps	%ymm2, %ymm6
	vmovaps	%ymm2, %ymm7
	addq	$32, %rax
	vmovdqa	k-32(%rax), %ymm1
	vgatherdps	%ymm5, fx(,%ymm1,4), %ymm4
	vgatherdps	%ymm6, fy(,%ymm1,4), %ymm0
	vgatherdps	%ymm7, fz(,%ymm1,4), %ymm3
	vaddps	%ymm4, %ymm0, %ymm0
	vaddps	%ymm3, %ymm0, %ymm0
	vmovaps	%ymm0, g-32(%rax)
	cmpq	$4096, %rax
	jne	.L54
	vzeroupper
	ret
	.cfi_endproc
.LFE0:
	.size	_Z3f20v, .-_Z3f20v
	.p2align 4,,15
	.globl	_Z3f21v
	.type	_Z3f21v, @function
_Z3f21v:
.LFB1:
	.cfi_startproc
	vmovdqa	.LC0(%rip), %ymm7
	vmovaps	.LC1(%rip), %ymm2
	xorl	%eax, %eax
	vmovdqa	.LC15(%rip), %ymm6
	vmovdqa	.LC16(%rip), %ymm5
	.p2align 4,,10
	.p2align 3
.L57:
	vmovaps	%ymm2, %ymm3
	vmovaps	%ymm2, %ymm8
	vmovaps	%ymm2, %ymm9
	addq	$32, %rax
	vpmulld	k-32(%rax), %ymm7, %ymm1
	vgatherdps	%ymm3, ff(,%ymm1,4), %ymm4
	vpaddd	%ymm6, %ymm1, %ymm3
	vpaddd	%ymm5, %ymm1, %ymm1
	vgatherdps	%ymm8, ff(,%ymm3,4), %ymm0
	vgatherdps	%ymm9, ff(,%ymm1,4), %ymm3
	vaddps	%ymm4, %ymm0, %ymm0
	vaddps	%ymm3, %ymm0, %ymm0
	vmovaps	%ymm0, g-32(%rax)
	cmpq	$4096, %rax
	jne	.L57
	vzeroupper
	ret
	.cfi_endproc
.LFE1:
	.size	_Z3f21v, .-_Z3f21v
	.p2align 4,,15
	.globl	_Z4f21bv
	.type	_Z4f21bv, @function
_Z4f21bv:
.LFB2:
	.cfi_startproc
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L60:
	movl	k(%rdx), %eax
	addq	$4, %rdx
	leal	(%rax,%rax,2), %eax
	cltq
	vmovss	ff(,%rax,4), %xmm0
	vaddss	ff+4(,%rax,4), %xmm0, %xmm0
	vaddss	ff+8(,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, g-4(%rdx)
	cmpq	$4096, %rdx
	jne	.L60
	ret
	.cfi_endproc
.LFE2:
	.size	_Z4f21bv, .-_Z4f21bv
	.p2align 4,,15
	.globl	_Z3f22v
	.type	_Z3f22v, @function
_Z3f22v:
.LFB3:
	.cfi_startproc
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L63:
	movslq	k(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rdx
	vmovss	f3(,%rdx,4), %xmm0
	vaddss	f3+4(,%rdx,4), %xmm0, %xmm0
	vaddss	f3+8(,%rdx,4), %xmm0, %xmm0
	vmovss	%xmm0, g-4(%rax)
	cmpq	$4096, %rax
	jne	.L63
	ret
	.cfi_endproc
.LFE3:
	.size	_Z3f22v, .-_Z3f22v
	.p2align 4,,15
	.globl	_Z3seq6float3iii
	.type	_Z3seq6float3iii, @function
_Z3seq6float3iii:
.LFB4:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	leal	-1(%rdi), %r10d
	vmovq	%xmm0, -32(%rbp)
	testl	%edi, %edi
	jle	.L66
	leal	-1(%rdi), %r10d
	vmovd	%xmm1, %esi
	movl	-32(%rbp), %r8d
	movl	-28(%rbp), %r9d
	cmpl	$6, %r10d
	jbe	.L70
	vmovd	%r8d, %xmm7
	movl	%edi, %r11d
	vmovdqa	.LC2(%rip), %ymm13
	vmovdqa	.LC3(%rip), %ymm12
	vbroadcastss	%xmm7, %ymm1
	vmovd	%r9d, %xmm7
	vmovdqa	.LC4(%rip), %ymm11
	vmovdqa	.LC5(%rip), %ymm10
	vbroadcastss	%xmm7, %ymm15
	vmovd	%esi, %xmm7
	shrl	$3, %r11d
	xorl	%edx, %edx
	vbroadcastss	%xmm7, %ymm14
	movl	$position, %eax
	movl	$r2inv, %ecx
	vmovdqa	.LC9(%rip), %ymm9
	vmovdqa	.LC13(%rip), %ymm8
	.p2align 4,,10
	.p2align 3
.L68:
	vmovaps	(%rax), %ymm0
	vmovaps	32(%rax), %ymm6
	addl	$1, %edx
	addq	$96, %rax
	vmovaps	-32(%rax), %ymm5
	addq	$32, %rcx
	vpshufb	%ymm13, %ymm0, %ymm3
	vpshufb	%ymm11, %ymm6, %ymm4
	vpermq	$78, %ymm3, %ymm2
	vpshufb	%ymm12, %ymm0, %ymm3
	vpor	%ymm2, %ymm3, %ymm3
	vpermd	%ymm5, %ymm10, %ymm2
	vpor	%ymm4, %ymm3, %ymm3
	vpshufb	.LC6(%rip), %ymm0, %ymm4
	vpermq	$78, %ymm4, %ymm7
	vblendps	$192, %ymm2, %ymm3, %ymm3
	vpshufb	.LC7(%rip), %ymm0, %ymm4
	vpshufb	.LC8(%rip), %ymm6, %ymm2
	vpor	%ymm7, %ymm4, %ymm4
	vsubps	%ymm3, %ymm1, %ymm3
	vpshufb	.LC12(%rip), %ymm6, %ymm6
	vpor	%ymm2, %ymm4, %ymm4
	vpermd	%ymm5, %ymm9, %ymm2
	vblendps	$224, %ymm2, %ymm4, %ymm4
	vpshufb	.LC10(%rip), %ymm0, %ymm2
	vpshufb	.LC11(%rip), %ymm0, %ymm0
	vsubps	%ymm4, %ymm15, %ymm4
	vpermq	$78, %ymm2, %ymm2
	vpor	%ymm2, %ymm0, %ymm2
	vpermd	%ymm5, %ymm8, %ymm0
	vpor	%ymm6, %ymm2, %ymm2
	vmulps	%ymm4, %ymm4, %ymm4
	vblendps	$224, %ymm0, %ymm2, %ymm2
	vsubps	%ymm2, %ymm14, %ymm0
	vfmadd132ps	%ymm3, %ymm4, %ymm3
	vfmadd132ps	%ymm0, %ymm3, %ymm0
	vmovaps	%ymm0, -32(%rcx)
	cmpl	%edx, %r11d
	ja	.L68
	movl	%edi, %eax
	andl	$-8, %eax
	cmpl	%edi, %eax
	je	.L76
	vzeroupper
.L67:
	movslq	%eax, %rcx
	vmovd	%r8d, %xmm7
	leaq	(%rcx,%rcx,2), %rdx
	salq	$2, %rdx
	vsubss	position(%rdx), %xmm7, %xmm1
	vmovd	%r9d, %xmm7
	vsubss	position+4(%rdx), %xmm7, %xmm2
	vmovd	%esi, %xmm7
	vsubss	position+8(%rdx), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	1(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L66
	movslq	%ecx, %rcx
	vmovd	%r9d, %xmm5
	vmovd	%r8d, %xmm7
	leaq	(%rcx,%rcx,2), %rdx
	salq	$2, %rdx
	vsubss	position+4(%rdx), %xmm5, %xmm2
	vsubss	position(%rdx), %xmm7, %xmm1
	vmovd	%esi, %xmm7
	vsubss	position+8(%rdx), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L66
	movslq	%ecx, %rcx
	vmovd	%r8d, %xmm5
	leaq	(%rcx,%rcx,2), %rdx
	salq	$2, %rdx
	vsubss	position(%rdx), %xmm5, %xmm1
	vmovd	%r9d, %xmm5
	vsubss	position+4(%rdx), %xmm5, %xmm2
	vsubss	position+8(%rdx), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L66
	movslq	%ecx, %rcx
	vmovd	%r8d, %xmm1
	leaq	(%rcx,%rcx,2), %rdx
	salq	$2, %rdx
	vsubss	position+4(%rdx), %xmm5, %xmm2
	vsubss	position(%rdx), %xmm1, %xmm1
	vsubss	position+8(%rdx), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L66
	movslq	%ecx, %rcx
	vmovd	%r8d, %xmm1
	leaq	(%rcx,%rcx,2), %rdx
	salq	$2, %rdx
	vsubss	position+4(%rdx), %xmm5, %xmm2
	vsubss	position(%rdx), %xmm1, %xmm1
	vsubss	position+8(%rdx), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L66
	movslq	%ecx, %rcx
	vmovd	%r8d, %xmm1
	addl	$6, %eax
	leaq	(%rcx,%rcx,2), %rdx
	salq	$2, %rdx
	vsubss	position+4(%rdx), %xmm5, %xmm2
	vsubss	position(%rdx), %xmm1, %xmm1
	vsubss	position+8(%rdx), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %edi
	jle	.L66
	cltq
	vmovd	%r8d, %xmm1
	leaq	(%rax,%rax,2), %rdx
	salq	$2, %rdx
	vsubss	position+4(%rdx), %xmm5, %xmm2
	vsubss	position(%rdx), %xmm1, %xmm0
	vsubss	position+8(%rdx), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vfmadd132ss	%xmm1, %xmm0, %xmm1
	vmovss	%xmm1, r2inv(,%rax,4)
.L66:
	movslq	%r10d, %r10
	vmovss	r2inv(,%r10,4), %xmm0
	vmovss	%xmm0, res(%rip)
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L76:
	.cfi_restore_state
	vzeroupper
	jmp	.L66
	.p2align 4,,10
	.p2align 3
.L70:
	xorl	%eax, %eax
	jmp	.L67
	.cfi_endproc
.LFE4:
	.size	_Z3seq6float3iii, .-_Z3seq6float3iii
	.p2align 4,,15
	.globl	_Z3soa6float3iii
	.type	_Z3soa6float3iii, @function
_Z3soa6float3iii:
.LFB5:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	leal	-1(%rdi), %esi
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	vmovq	%xmm0, -32(%rbp)
	vmovss	%xmm1, -24(%rbp)
	vmovss	-28(%rbp), %xmm5
	testl	%edi, %edi
	jle	.L78
	leal	-1(%rdi), %esi
	vmovss	-32(%rbp), %xmm7
	cmpl	$6, %esi
	jbe	.L82
	movl	%edi, %ecx
	vmovaps	.LC1(%rip), %ymm4
	xorl	%eax, %eax
	xorl	%edx, %edx
	shrl	$3, %ecx
	vbroadcastss	%xmm7, %ymm8
	vbroadcastss	%xmm5, %ymm6
	.p2align 4,,10
	.p2align 3
.L80:
	vmovaps	%ymm4, %ymm1
	vmovaps	%ymm4, %ymm9
	addl	$1, %edx
	addq	$32, %rax
	vmovdqa	neighList-32(%rax), %ymm0
	vgatherdps	%ymm1, fx(,%ymm0,4), %ymm2
	vmovaps	%ymm4, %ymm1
	vgatherdps	%ymm1, fy(,%ymm0,4), %ymm3
	vsubps	%ymm2, %ymm8, %ymm2
	vgatherdps	%ymm9, fz(,%ymm0,4), %ymm1
	vsubps	%ymm3, %ymm6, %ymm3
	vsubps	%ymm1, %ymm6, %ymm0
	vmulps	%ymm3, %ymm3, %ymm3
	vfmadd132ps	%ymm2, %ymm3, %ymm2
	vfmadd132ps	%ymm0, %ymm2, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edx, %ecx
	ja	.L80
	movl	%edi, %eax
	andl	$-8, %eax
	cmpl	%edi, %eax
	je	.L88
	vzeroupper
.L79:
	movslq	%eax, %rcx
	movslq	neighList(,%rcx,4), %rdx
	vsubss	fy(,%rdx,4), %xmm5, %xmm2
	vsubss	fx(,%rdx,4), %xmm7, %xmm1
	vsubss	fz(,%rdx,4), %xmm5, %xmm0
	leal	1(%rax), %edx
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%edx, %edi
	jle	.L78
	movslq	%edx, %rdx
	movslq	neighList(,%rdx,4), %rcx
	vsubss	fy(,%rcx,4), %xmm5, %xmm2
	vsubss	fx(,%rcx,4), %xmm7, %xmm1
	vsubss	fz(,%rcx,4), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	leal	2(%rax), %edx
	cmpl	%edx, %edi
	jle	.L78
	movslq	%edx, %rdx
	movslq	neighList(,%rdx,4), %rcx
	vsubss	fy(,%rcx,4), %xmm5, %xmm2
	vsubss	fx(,%rcx,4), %xmm7, %xmm1
	vsubss	fz(,%rcx,4), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	leal	3(%rax), %edx
	cmpl	%edx, %edi
	jle	.L78
	movslq	%edx, %rdx
	movslq	neighList(,%rdx,4), %rcx
	vsubss	fy(,%rcx,4), %xmm5, %xmm2
	vsubss	fx(,%rcx,4), %xmm7, %xmm1
	vsubss	fz(,%rcx,4), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	leal	4(%rax), %edx
	cmpl	%edx, %edi
	jle	.L78
	movslq	%edx, %rdx
	movslq	neighList(,%rdx,4), %rcx
	vsubss	fy(,%rcx,4), %xmm5, %xmm2
	vsubss	fx(,%rcx,4), %xmm7, %xmm1
	vsubss	fz(,%rcx,4), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	leal	5(%rax), %edx
	cmpl	%edx, %edi
	jle	.L78
	movslq	%edx, %rdx
	addl	$6, %eax
	movslq	neighList(,%rdx,4), %rcx
	vsubss	fy(,%rcx,4), %xmm5, %xmm2
	vsubss	fx(,%rcx,4), %xmm7, %xmm1
	vsubss	fz(,%rcx,4), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	cmpl	%eax, %edi
	jle	.L78
	cltq
	movslq	neighList(,%rax,4), %rdx
	vsubss	fy(,%rdx,4), %xmm5, %xmm0
	vsubss	fx(,%rdx,4), %xmm7, %xmm7
	vsubss	fz(,%rdx,4), %xmm5, %xmm5
	vmulss	%xmm0, %xmm0, %xmm0
	vfmadd132ss	%xmm7, %xmm0, %xmm7
	vfmadd132ss	%xmm5, %xmm7, %xmm5
	vmovss	%xmm5, r2inv(,%rax,4)
.L78:
	movslq	%esi, %rsi
	vmovss	r2inv(,%rsi,4), %xmm0
	vmovss	%xmm0, res(%rip)
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L88:
	.cfi_restore_state
	vzeroupper
	jmp	.L78
	.p2align 4,,10
	.p2align 3
.L82:
	xorl	%eax, %eax
	jmp	.L79
	.cfi_endproc
.LFE5:
	.size	_Z3soa6float3iii, .-_Z3soa6float3iii
	.p2align 4,,15
	.globl	_Z3bar6float3iii
	.type	_Z3bar6float3iii, @function
_Z3bar6float3iii:
.LFB6:
	.cfi_startproc
	vmovq	%xmm0, -16(%rsp)
	vmovss	-12(%rsp), %xmm3
	testl	%edi, %edi
	jle	.L95
	leal	-1(%rdi), %eax
	vmovss	-16(%rsp), %xmm4
	movq	%rax, %rdi
	leaq	4(,%rax,4), %rsi
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L91:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rcx
	leaq	position(,%rcx,4), %rdx
	vsubss	position(,%rcx,4), %xmm4, %xmm1
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rax, %rsi
	jne	.L91
.L90:
	movslq	%edi, %rdi
	vmovss	r2inv(,%rdi,4), %xmm0
	vmovss	%xmm0, res(%rip)
	ret
	.p2align 4,,10
	.p2align 3
.L95:
	subl	$1, %edi
	jmp	.L90
	.cfi_endproc
.LFE6:
	.size	_Z3bar6float3iii, .-_Z3bar6float3iii
	.p2align 4,,15
	.globl	_Z4bar26float3iii
	.type	_Z4bar26float3iii, @function
_Z4bar26float3iii:
.LFB7:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	leal	-1(%rdi), %ecx
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	vmovq	%xmm0, -32(%rbp)
	vmovss	%xmm1, -24(%rbp)
	testl	%edi, %edi
	jle	.L97
	leal	-1(%rdi), %ecx
	vmovss	-32(%rbp), %xmm10
	vmovss	-28(%rbp), %xmm11
	cmpl	$6, %ecx
	jbe	.L101
	movl	%edi, %esi
	vmovdqa	.LC0(%rip), %ymm6
	xorl	%eax, %eax
	xorl	%edx, %edx
	vmovaps	.LC1(%rip), %ymm5
	shrl	$3, %esi
	vbroadcastss	%xmm10, %ymm9
	vbroadcastss	%xmm11, %ymm8
	vbroadcastss	%xmm1, %ymm7
	.p2align 4,,10
	.p2align 3
.L99:
	vmovaps	%ymm5, %ymm2
	vmovaps	%ymm5, %ymm12
	addl	$1, %edx
	addq	$32, %rax
	vpmulld	neighList-32(%rax), %ymm6, %ymm0
	vgatherdps	%ymm2, position(,%ymm0,4), %ymm3
	vmovaps	%ymm5, %ymm2
	vgatherdps	%ymm2, position+4(,%ymm0,4), %ymm4
	vsubps	%ymm3, %ymm9, %ymm3
	vgatherdps	%ymm12, position+8(,%ymm0,4), %ymm2
	vsubps	%ymm4, %ymm8, %ymm4
	vsubps	%ymm2, %ymm7, %ymm0
	vmulps	%ymm4, %ymm4, %ymm4
	vfmadd132ps	%ymm3, %ymm4, %ymm3
	vfmadd132ps	%ymm0, %ymm3, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edx, %esi
	ja	.L99
	movl	%edi, %eax
	andl	$-8, %eax
	cmpl	%edi, %eax
	je	.L107
	vzeroupper
.L98:
	movslq	%eax, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm3
	vsubss	position(,%rdx,4), %xmm11, %xmm2
	leal	1(%rax), %edx
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%edx, %edi
	jle	.L97
	movslq	%edx, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm3
	vsubss	position(,%rdx,4), %xmm11, %xmm2
	leal	2(%rax), %edx
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%edx, %edi
	jle	.L97
	movslq	%edx, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm3
	vsubss	position(,%rdx,4), %xmm11, %xmm2
	leal	3(%rax), %edx
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%edx, %edi
	jle	.L97
	movslq	%edx, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm3
	vsubss	position(,%rdx,4), %xmm11, %xmm2
	leal	4(%rax), %edx
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%edx, %edi
	jle	.L97
	movslq	%edx, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm3
	vsubss	position(,%rdx,4), %xmm11, %xmm2
	leal	5(%rax), %edx
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%edx, %edi
	jle	.L97
	movslq	%edx, %rsi
	addl	$6, %eax
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm3
	vsubss	position(,%rdx,4), %xmm11, %xmm2
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%eax, %edi
	jle	.L97
	cltq
	movl	neighList(,%rax,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm10, %xmm10
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm1, %xmm1
	vsubss	position(,%rdx,4), %xmm11, %xmm11
	vmulss	%xmm1, %xmm1, %xmm1
	vfmadd132ss	%xmm11, %xmm1, %xmm11
	vfmadd132ss	%xmm10, %xmm11, %xmm10
	vmovss	%xmm10, r2inv(,%rax,4)
.L97:
	movslq	%ecx, %rcx
	vmovss	r2inv(,%rcx,4), %xmm0
	vmovss	%xmm0, res(%rip)
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L107:
	.cfi_restore_state
	vzeroupper
	jmp	.L97
	.p2align 4,,10
	.p2align 3
.L101:
	xorl	%eax, %eax
	jmp	.L98
	.cfi_endproc
.LFE7:
	.size	_Z4bar26float3iii, .-_Z4bar26float3iii
	.p2align 4,,15
	.globl	_Z3foo6float3iii
	.type	_Z3foo6float3iii, @function
_Z3foo6float3iii:
.LFB8:
	.cfi_startproc
	vmovq	%xmm0, -16(%rsp)
	testl	%edi, %edi
	jle	.L114
	leal	-1(%rdi), %eax
	vmovss	-16(%rsp), %xmm5
	vmovss	-12(%rsp), %xmm4
	movq	%rax, %rdi
	leaq	4(,%rax,4), %rsi
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L110:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rcx
	leaq	position(,%rcx,4), %rdx
	vsubss	position(,%rcx,4), %xmm5, %xmm2
	vsubss	4(%rdx), %xmm4, %xmm3
	vsubss	8(%rdx), %xmm1, %xmm0
	vmulss	%xmm3, %xmm3, %xmm3
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm0, %xmm2, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rax, %rsi
	jne	.L110
.L109:
	movslq	%edi, %rdi
	vmovss	r2inv(,%rdi,4), %xmm0
	vmovss	%xmm0, res(%rip)
	ret
	.p2align 4,,10
	.p2align 3
.L114:
	subl	$1, %edi
	jmp	.L109
	.cfi_endproc
.LFE8:
	.size	_Z3foo6float3iii, .-_Z3foo6float3iii
	.p2align 4,,15
	.globl	_Z5time3PK6float3iiii
	.type	_Z5time3PK6float3iiii, @function
_Z5time3PK6float3iiii:
.LFB2100:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	%edx, %ebx
	subq	$96, %rsp
	movq	%rdi, -80(%rbp)
	testl	%esi, %esi
	jle	.L116
	leal	-1(%rdx), %ecx
	leal	-1(%rsi), %eax
	movl	%edx, %r12d
	movl	%edx, %r8d
	movl	%ecx, -88(%rbp)
	leaq	3(%rax,%rax,2), %rax
	movq	%rdi, %r14
	movq	%rdi, %r15
	shrl	$3, %r12d
	leaq	(%rdi,%rax,4), %rdi
	andl	$-8, %r8d
	movslq	%ecx, %rsi
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	.p2align 4,,10
	.p2align 3
.L121:
	vmovss	(%r14), %xmm4
	vmovss	4(%r14), %xmm9
	vmovss	8(%r14), %xmm11
	testl	%ebx, %ebx
	jle	.L117
	cmpl	$6, %ecx
	jbe	.L152
	vbroadcastss	%xmm4, %ymm8
	vbroadcastss	%xmm9, %ymm7
	xorl	%eax, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm11, %ymm5
	.p2align 4,,10
	.p2align 3
.L119:
	vmovaps	%ymm6, %ymm3
	vmovaps	%ymm6, %ymm1
	vmovaps	%ymm6, %ymm12
	addl	$1, %edx
	vpmulld	neighList(%rax), %ymm10, %ymm0
	addq	$32, %rax
	vgatherdps	%ymm3, position(,%ymm0,4), %ymm2
	vgatherdps	%ymm1, position+4(,%ymm0,4), %ymm3
	vsubps	%ymm3, %ymm7, %ymm3
	vsubps	%ymm2, %ymm8, %ymm2
	vgatherdps	%ymm12, position+8(,%ymm0,4), %ymm1
	vsubps	%ymm1, %ymm5, %ymm0
	vmulps	%ymm3, %ymm3, %ymm3
	vfmadd132ps	%ymm2, %ymm3, %ymm2
	vfmadd132ps	%ymm0, %ymm2, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edx, %r12d
	ja	.L119
	cmpl	%ebx, %r8d
	je	.L117
	movl	%r8d, %eax
.L118:
	movslq	%eax, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	1(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L117
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	2(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L117
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	3(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L117
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	4(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L117
	movslq	%r9d, %r9
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	5(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L117
	movslq	%r9d, %r9
	addl	$6, %eax
	movl	neighList(,%r9,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	cmpl	%eax, %ebx
	jle	.L117
	cltq
	movl	neighList(,%rax,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm4
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm11
	vsubss	position(,%rdx,4), %xmm9, %xmm9
	vmulss	%xmm11, %xmm11, %xmm11
	vfmadd132ss	%xmm9, %xmm11, %xmm9
	vfmadd132ss	%xmm4, %xmm9, %xmm4
	vmovss	%xmm4, r2inv(,%rax,4)
.L117:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r14
	vmovss	%xmm0, res(%rip)
	cmpq	%rdi, %r14
	jne	.L121
	movq	%rsi, -104(%rbp)
	movl	%r8d, -72(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-80(%rbp), %r13
	movl	-88(%rbp), %edx
	movq	%rax, -96(%rbp)
	vmovdqa	.LC2(%rip), %ymm15
	vmovdqa	.LC3(%rip), %ymm14
	vmovdqa	.LC5(%rip), %ymm13
	vmovdqa	.LC9(%rip), %ymm12
	movl	-72(%rbp), %r8d
	movq	-104(%rbp), %rsi
	.p2align 4,,10
	.p2align 3
.L127:
	vmovss	8(%r13), %xmm5
	movl	0(%r13), %ecx
	movl	4(%r13), %edi
	vmovss	%xmm5, -72(%rbp)
	testl	%ebx, %ebx
	jle	.L123
	cmpl	$6, %edx
	jbe	.L153
	vmovd	%ecx, %xmm7
	vmovdqa	.LC12(%rip), %ymm6
	movl	$position, %eax
	xorl	%r9d, %r9d
	vbroadcastss	%xmm7, %ymm9
	vmovd	%edi, %xmm7
	movl	$r2inv, %r10d
	vbroadcastss	%xmm7, %ymm8
	vbroadcastss	%xmm5, %ymm7
	vmovdqa	.LC13(%rip), %ymm5
	.p2align 4,,10
	.p2align 3
.L125:
	vmovaps	(%rax), %ymm0
	addl	$1, %r9d
	addq	$32, %r10
	addq	$96, %rax
	vmovaps	-64(%rax), %ymm10
	vmovaps	-32(%rax), %ymm4
	vpshufb	.LC7(%rip), %ymm0, %ymm11
	vpshufb	%ymm15, %ymm0, %ymm3
	vpshufb	.LC4(%rip), %ymm10, %ymm2
	vpermq	$78, %ymm3, %ymm1
	vpshufb	%ymm14, %ymm0, %ymm3
	vpor	%ymm1, %ymm3, %ymm3
	vpshufb	.LC8(%rip), %ymm10, %ymm1
	vpshufb	%ymm6, %ymm10, %ymm10
	vpor	%ymm2, %ymm3, %ymm3
	vpermd	%ymm4, %ymm13, %ymm2
	vblendps	$192, %ymm2, %ymm3, %ymm3
	vpshufb	.LC6(%rip), %ymm0, %ymm2
	vpermq	$78, %ymm2, %ymm2
	vsubps	%ymm3, %ymm9, %ymm3
	vpor	%ymm2, %ymm11, %ymm2
	vpor	%ymm1, %ymm2, %ymm2
	vpermd	%ymm4, %ymm12, %ymm1
	vblendps	$224, %ymm1, %ymm2, %ymm2
	vpshufb	.LC10(%rip), %ymm0, %ymm1
	vpshufb	.LC11(%rip), %ymm0, %ymm0
	vsubps	%ymm2, %ymm8, %ymm2
	vpermq	$78, %ymm1, %ymm1
	vpor	%ymm1, %ymm0, %ymm1
	vpermd	%ymm4, %ymm5, %ymm0
	vpor	%ymm10, %ymm1, %ymm1
	vmulps	%ymm2, %ymm2, %ymm2
	vblendps	$224, %ymm0, %ymm1, %ymm1
	vsubps	%ymm1, %ymm7, %ymm0
	vfmadd132ps	%ymm3, %ymm2, %ymm3
	vfmadd132ps	%ymm0, %ymm3, %ymm0
	vmovaps	%ymm0, -32(%r10)
	cmpl	%r9d, %r12d
	ja	.L125
	cmpl	%ebx, %r8d
	je	.L123
	vmovss	-72(%rbp), %xmm7
	movl	%r8d, %eax
.L124:
	movslq	%eax, %r9
	vmovd	%ecx, %xmm5
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vmovd	%edi, %xmm5
	vsubss	4(%r10), %xmm5, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	1(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L123
	movslq	%r9d, %r9
	vmovd	%edi, %xmm6
	vmovd	%ecx, %xmm5
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	2(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L123
	movslq	%r9d, %r9
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	3(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L123
	movslq	%r9d, %r9
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vmovd	%edi, %xmm5
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	4(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L123
	movslq	%r9d, %r9
	vmovd	%ecx, %xmm6
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm6, %xmm1
	vmovd	%edi, %xmm6
	vsubss	4(%r10), %xmm5, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	leal	5(%rax), %r9d
	cmpl	%r9d, %ebx
	jle	.L123
	movslq	%r9d, %r9
	vmovd	%ecx, %xmm5
	addl	$6, %eax
	leaq	(%r9,%r9,2), %r11
	leaq	position(,%r11,4), %r10
	vsubss	position(,%r11,4), %xmm5, %xmm1
	vmovd	%edi, %xmm5
	vsubss	4(%r10), %xmm6, %xmm2
	vsubss	8(%r10), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r9,4)
	cmpl	%eax, %ebx
	jle	.L123
	cltq
	vmovd	%ecx, %xmm6
	leaq	(%rax,%rax,2), %r10
	leaq	position(,%r10,4), %r9
	vsubss	position(,%r10,4), %xmm6, %xmm1
	vsubss	4(%r9), %xmm5, %xmm2
	vsubss	8(%r9), %xmm7, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rax,4)
.L123:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r13
	vmovss	%xmm0, res(%rip)
	cmpq	%r14, %r13
	jne	.L127
	movq	%rsi, -112(%rbp)
	movl	%r8d, -104(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	-96(%rbp), %rax
	movabsq	$2361183241434822607, %rdx
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -72(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-80(%rbp), %rdx
	vmovaps	.LC1(%rip), %ymm6
	movl	-104(%rbp), %r8d
	movl	-88(%rbp), %edi
	movq	%rax, %r14
	movq	-112(%rbp), %rsi
	.p2align 4,,10
	.p2align 3
.L133:
	vmovss	(%rdx), %xmm7
	vmovss	4(%rdx), %xmm4
	testl	%ebx, %ebx
	jle	.L129
	cmpl	$6, %edi
	jbe	.L154
	vbroadcastss	%xmm7, %ymm8
	vbroadcastss	%xmm4, %ymm5
	xorl	%eax, %eax
	xorl	%ecx, %ecx
	.p2align 4,,10
	.p2align 3
.L131:
	vmovaps	%ymm6, %ymm2
	vmovaps	%ymm6, %ymm0
	vmovaps	%ymm6, %ymm13
	addl	$1, %ecx
	vmovdqa	neighList(%rax), %ymm3
	addq	$32, %rax
	vgatherdps	%ymm2, fx(,%ymm3,4), %ymm1
	vgatherdps	%ymm0, fy(,%ymm3,4), %ymm2
	vgatherdps	%ymm13, fz(,%ymm3,4), %ymm0
	vsubps	%ymm2, %ymm5, %ymm2
	vsubps	%ymm1, %ymm8, %ymm1
	vsubps	%ymm0, %ymm5, %ymm0
	vmulps	%ymm2, %ymm2, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%ecx, %r12d
	ja	.L131
	cmpl	%ebx, %r8d
	je	.L129
	movl	%r8d, %eax
.L130:
	movslq	%eax, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	1(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L129
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L129
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L129
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L129
	movslq	%ecx, %rcx
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L129
	movslq	%ecx, %rcx
	addl	$6, %eax
	movslq	neighList(,%rcx,4), %r9
	vsubss	fy(,%r9,4), %xmm4, %xmm2
	vsubss	fx(,%r9,4), %xmm7, %xmm1
	vsubss	fz(,%r9,4), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %ebx
	jle	.L129
	cltq
	movslq	neighList(,%rax,4), %rcx
	vsubss	fy(,%rcx,4), %xmm4, %xmm0
	vsubss	fx(,%rcx,4), %xmm7, %xmm7
	vsubss	fz(,%rcx,4), %xmm4, %xmm4
	vmulss	%xmm0, %xmm0, %xmm0
	vfmadd132ss	%xmm7, %xmm0, %xmm7
	vfmadd132ss	%xmm4, %xmm7, %xmm4
	vmovss	%xmm4, r2inv(,%rax,4)
.L129:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %rdx
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdx
	jne	.L133
	movq	%rsi, -112(%rbp)
	movl	%r8d, -104(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -96(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-80(%rbp), %r9
	movl	-104(%rbp), %r8d
	movq	%rax, %r14
	movl	-88(%rbp), %eax
	movq	-112(%rbp), %rsi
	leaq	4(,%rax,4), %rcx
	.p2align 4,,10
	.p2align 3
.L137:
	vmovss	(%r9), %xmm4
	vmovss	4(%r9), %xmm3
	testl	%ebx, %ebx
	jle	.L135
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L136:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rdi
	leaq	position(,%rdi,4), %rdx
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L136
.L135:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r9
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %r9
	jne	.L137
	movq	%rsi, -128(%rbp)
	movq	%rcx, -120(%rbp)
	movl	%r8d, -112(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rdi
	imulq	%rdx
	movq	%rdi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -104(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-80(%rbp), %rdx
	movl	-112(%rbp), %r8d
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	movq	%rax, %r14
	movl	-88(%rbp), %r9d
	movq	-120(%rbp), %rcx
	movq	-128(%rbp), %rsi
	.p2align 4,,10
	.p2align 3
.L143:
	vmovss	(%rdx), %xmm3
	vmovss	4(%rdx), %xmm4
	vmovss	8(%rdx), %xmm5
	testl	%ebx, %ebx
	jle	.L139
	cmpl	$6, %r9d
	jbe	.L155
	vbroadcastss	%xmm3, %ymm9
	vbroadcastss	%xmm4, %ymm8
	xorl	%eax, %eax
	xorl	%edi, %edi
	vbroadcastss	%xmm5, %ymm7
	.p2align 4,,10
	.p2align 3
.L141:
	vmovaps	%ymm6, %ymm2
	vmovaps	%ymm6, %ymm0
	vmovaps	%ymm6, %ymm14
	addl	$1, %edi
	vpmulld	neighList(%rax), %ymm10, %ymm11
	addq	$32, %rax
	vgatherdps	%ymm2, position(,%ymm11,4), %ymm1
	vgatherdps	%ymm0, position+4(,%ymm11,4), %ymm2
	vsubps	%ymm2, %ymm8, %ymm2
	vsubps	%ymm1, %ymm9, %ymm1
	vgatherdps	%ymm14, position+8(,%ymm11,4), %ymm0
	vsubps	%ymm0, %ymm7, %ymm0
	vmulps	%ymm2, %ymm2, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edi, %r12d
	ja	.L141
	cmpl	%ebx, %r8d
	je	.L139
	movl	%r8d, %eax
.L140:
	movslq	%eax, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	1(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L139
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	2(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L139
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	3(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L139
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	4(%rax), %r10d
	cmpl	%r10d, %ebx
	jle	.L139
	movslq	%r10d, %r10
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	leal	5(%rax), %r10d
	cmpl	%ebx, %r10d
	jge	.L139
	movslq	%r10d, %r10
	addl	$6, %eax
	movl	neighList(,%r10,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm0
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm2
	vsubss	position(,%rdi,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%r10,4)
	cmpl	%eax, %ebx
	jle	.L139
	cltq
	movl	neighList(,%rax,4), %edi
	leal	(%rdi,%rdi,2), %edi
	movslq	%edi, %rdi
	vsubss	position(,%rdi,4), %xmm3, %xmm3
	addq	$1, %rdi
	vsubss	position+4(,%rdi,4), %xmm5, %xmm5
	vsubss	position(,%rdi,4), %xmm4, %xmm4
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, r2inv(,%rax,4)
.L139:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %rdx
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdx
	jne	.L143
	movq	%rsi, -88(%rbp)
	movq	%rcx, -80(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rdi
	imulq	%rdx
	movq	%rdi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r12
	subq	%rax, %r12
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-80(%rbp), %rcx
	movq	-88(%rbp), %rsi
	movq	%rax, %r14
	.p2align 4,,10
	.p2align 3
.L147:
	vmovss	(%r15), %xmm5
	vmovss	4(%r15), %xmm4
	vmovss	8(%r15), %xmm3
	testl	%ebx, %ebx
	jle	.L145
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L146:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rdi
	leaq	position(,%rdi,4), %rdx
	vsubss	position(,%rdi,4), %xmm5, %xmm1
	vsubss	4(%rdx), %xmm4, %xmm2
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L146
.L145:
	vmovss	r2inv(,%rsi,4), %xmm0
	addq	$12, %r15
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %r15
	jne	.L147
.L151:
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movl	$.LC14, %esi
	movl	$_ZSt4cout, %edi
	movabsq	$2361183241434822607, %rdx
	subq	%r14, %rax
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r13
	movl	$8, %edx
	subq	%rax, %r13
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%ebx, %esi
	movl	$_ZSt4cout, %edi
	call	_ZNSolsEi
	movl	$1, %edx
	leaq	-53(%rbp), %rsi
	movb	$32, -53(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-72(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-52(%rbp), %rsi
	movb	$32, -52(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-96(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-51(%rbp), %rsi
	movb	$32, -51(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-104(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-50(%rbp), %rsi
	movb	$32, -50(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-49(%rbp), %rsi
	movb	$32, -49(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r13, %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movq	%rax, %r12
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r12,%rax), %rbx
	testq	%rbx, %rbx
	je	.L178
	cmpb	$0, 56(%rbx)
	je	.L149
	movsbl	67(%rbx), %esi
.L150:
	movq	%r12, %rdi
	call	_ZNSo3putEc
	movq	%rax, %rdi
	call	_ZNSo5flushEv
	addq	$96, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L152:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L118
	.p2align 4,,10
	.p2align 3
.L155:
	xorl	%eax, %eax
	jmp	.L140
	.p2align 4,,10
	.p2align 3
.L154:
	xorl	%eax, %eax
	jmp	.L130
	.p2align 4,,10
	.p2align 3
.L153:
	xorl	%eax, %eax
	vmovaps	%xmm5, %xmm7
	jmp	.L124
	.p2align 4,,10
	.p2align 3
.L149:
	movq	%rbx, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	$_ZNKSt5ctypeIcE8do_widenEc, %rax
	je	.L150
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L150
.L116:
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %r12
	movq	%rax, %r13
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	%r13, %rax
	movq	%rax, %rcx
	imulq	%r12
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -72(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, %r13
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	%r13, %rax
	movq	%rax, %rcx
	imulq	%r12
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -96(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, %r13
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	%r13, %rax
	movq	%rax, %rcx
	imulq	%r12
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -104(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, %r13
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	%r13, %rax
	movq	%rax, %rcx
	imulq	%r12
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r12
	subq	%rax, %r12
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, %r14
	jmp	.L151
.L178:
	call	_ZSt16__throw_bad_castv
	.cfi_endproc
.LFE2100:
	.size	_Z5time3PK6float3iiii, .-_Z5time3PK6float3iiii
	.section	.rodata.str1.1
.LC18:
	.string	"tivially sequential"
.LC20:
	.string	"small stride"
.LC22:
	.string	"big stride"
.LC23:
	.string	"random"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB2109:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	movl	$19, %edx
	movl	$.LC18, %esi
	pushq	-8(%r10)
	movl	$_ZSt4cout, %edi
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	movl	$neighList, %r14d
	pushq	%r13
	pushq	%r12
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	movq	%r14, %r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	pushq	%rbx
	subq	$1200048, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movq	%rsp, %r13
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	vmovdqa	.LC17(%rip), %ymm0
	.p2align 4,,10
	.p2align 3
.L180:
	vmovdqa	%ymm0, (%r12)
	addq	$32, %r12
	vpaddd	.LC19(%rip), %ymm0, %ymm0
	cmpq	$neighList+4096, %r12
	jne	.L180
	movl	$9, %r15d
	movl	$2, %ebx
	vzeroupper
	.p2align 4,,10
	.p2align 3
.L181:
	movl	%ebx, %esi
	movq	%r13, %rdi
	addl	%ebx, %ebx
	call	_Z5time3PK6float3iiii.constprop.3
	subl	$1, %r15d
	jne	.L181
	movl	$12, %edx
	movl	$.LC20, %esi
	movl	$_ZSt4cout, %edi
	movl	$neighList, %ebx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	vmovdqa	.LC17(%rip), %ymm1
	vmovdqa	.LC21(%rip), %ymm2
	.p2align 4,,10
	.p2align 3
.L182:
	vpslld	$1, %ymm1, %ymm0
	addq	$32, %rbx
	vpaddd	.LC19(%rip), %ymm1, %ymm1
	vpand	%ymm2, %ymm0, %ymm0
	vmovdqa	%ymm0, -32(%rbx)
	cmpq	%rbx, %r12
	jne	.L182
	movl	$9, %r15d
	movl	$2, %r12d
	.p2align 4,,10
	.p2align 3
.L183:
	movl	%r12d, %esi
	movq	%r13, %rdi
	vmovdqa	%ymm2, -80(%rbp)
	vzeroupper
	call	_Z5time3PK6float3iiii.constprop.3
	addl	%r12d, %r12d
	subl	$1, %r15d
	vmovdqa	-80(%rbp), %ymm2
	jne	.L183
	movl	$10, %edx
	movl	$.LC22, %esi
	movl	$_ZSt4cout, %edi
	vzeroupper
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	vmovdqa	.LC17(%rip), %ymm1
	vmovdqa	-80(%rbp), %ymm2
	.p2align 4,,10
	.p2align 3
.L184:
	vpslld	$5, %ymm1, %ymm0
	addq	$32, %r14
	vpaddd	.LC19(%rip), %ymm1, %ymm1
	vpand	%ymm2, %ymm0, %ymm0
	vmovdqa	%ymm0, -32(%r14)
	cmpq	%rbx, %r14
	jne	.L184
	movl	$9, %r14d
	movl	$2, %r12d
	vzeroupper
	.p2align 4,,10
	.p2align 3
.L185:
	movl	%r12d, %esi
	movq	%r13, %rdi
	addl	%r12d, %r12d
	call	_Z5time3PK6float3iiii.constprop.3
	subl	$1, %r14d
	jne	.L185
	movl	$6, %edx
	movl	$.LC23, %esi
	movl	$_ZSt4cout, %edi
	subq	$neighList+8, %rbx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$neighList+4, %r12d
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rbx, %rax
	shrq	$2, %rax
	leaq	neighList+8(,%rax,4), %rbx
	.p2align 4,,10
	.p2align 3
.L186:
	call	rand
	movq	%r12, %rcx
	subq	$neighList, %rcx
	cltq
	sarq	$2, %rcx
	cqto
	addq	$1, %rcx
	idivq	%rcx
	leaq	neighList(,%rdx,4), %rax
	cmpq	%r12, %rax
	je	.L187
	movl	neighList(,%rdx,4), %ecx
	movl	(%r12), %eax
	addq	$4, %r12
	movl	%ecx, -4(%r12)
	movl	%eax, neighList(,%rdx,4)
	cmpq	%r12, %rbx
	jne	.L186
.L189:
	movl	$9, %r12d
	movl	$2, %ebx
	.p2align 4,,10
	.p2align 3
.L190:
	movl	%ebx, %esi
	movq	%r13, %rdi
	addl	%ebx, %ebx
	call	_Z5time3PK6float3iiii.constprop.3
	subl	$1, %r12d
	jne	.L190
	leaq	-48(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L187:
	.cfi_restore_state
	addq	$4, %r12
	cmpq	%r12, %rbx
	jne	.L186
	jmp	.L189
	.cfi_endproc
.LFE2109:
	.size	main, .-main
	.p2align 4,,15
	.type	_GLOBAL__sub_I_fx, @function
_GLOBAL__sub_I_fx:
.LFB2604:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	call	_ZNSt8ios_base4InitC1Ev
	movl	$__dso_handle, %edx
	movl	$_ZStL8__ioinit, %esi
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit
	.cfi_endproc
.LFE2604:
	.size	_GLOBAL__sub_I_fx, .-_GLOBAL__sub_I_fx
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_fx
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.globl	res
	.bss
	.align 4
	.type	res, @object
	.size	res, 4
res:
	.zero	4
	.globl	r2inv
	.align 32
	.type	r2inv, @object
	.size	r2inv, 4096
r2inv:
	.zero	4096
	.globl	maxNeighbors
	.align 4
	.type	maxNeighbors, @object
	.size	maxNeighbors, 4
maxNeighbors:
	.zero	4
	.globl	neighList
	.align 32
	.type	neighList, @object
	.size	neighList, 4096
neighList:
	.zero	4096
	.globl	position
	.align 32
	.type	position, @object
	.size	position, 12288
position:
	.zero	12288
	.globl	f3
	.align 32
	.type	f3, @object
	.size	f3, 12288
f3:
	.zero	12288
	.globl	ff
	.align 32
	.type	ff, @object
	.size	ff, 12288
ff:
	.zero	12288
	.globl	k
	.align 32
	.type	k, @object
	.size	k, 4096
k:
	.zero	4096
	.globl	fz
	.align 32
	.type	fz, @object
	.size	fz, 4096
fz:
	.zero	4096
	.globl	fy
	.align 32
	.type	fy, @object
	.size	fy, 4096
fy:
	.zero	4096
	.globl	g
	.align 32
	.type	g, @object
	.size	g, 4096
g:
	.zero	4096
	.globl	fx
	.align 32
	.type	fx, @object
	.size	fx, 4096
fx:
	.zero	4096
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC0:
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.long	3
	.align 32
.LC1:
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.align 32
.LC2:
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
	.byte	2
	.byte	3
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC3:
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC4:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC5:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	2
	.long	5
	.align 32
.LC6:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC7:
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC8:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC9:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	0
	.long	3
	.long	6
	.align 32
.LC10:
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	0
	.byte	1
	.byte	2
	.byte	3
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	4
	.byte	5
	.byte	6
	.byte	7
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC11:
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC12:
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
	.byte	2
	.byte	3
	.byte	12
	.byte	13
	.byte	14
	.byte	15
	.byte	8
	.byte	9
	.byte	10
	.byte	11
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.byte	-128
	.align 32
.LC13:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	1
	.long	4
	.long	7
	.align 32
.LC15:
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.align 32
.LC16:
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.long	2
	.align 32
.LC17:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.align 32
.LC19:
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.align 32
.LC21:
	.long	1023
	.long	1023
	.long	1023
	.long	1023
	.long	1023
	.long	1023
	.long	1023
	.long	1023
	.hidden	__dso_handle
	.ident	"GCC: (GNU) 7.0.1 20170326 (experimental) [trunk revision 246485]"
	.section	.note.GNU-stack,"",@progbits
