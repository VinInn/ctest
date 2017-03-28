	.file	"gather.cpp"
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4,,15
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB1279:
	.cfi_startproc
	movl	%esi, %eax
	ret
	.cfi_endproc
.LFE1279:
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
	pushq	-8(%r10)
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
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	leal	-1(%rsi), %r10d
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	%esi, %ebx
	movslq	%r10d, %r8
	leaq	1200000(%rdi), %rsi
	movl	%ebx, %r12d
	movl	%ebx, %r9d
	shrl	$3, %r12d
	andl	$-8, %r9d
	subq	$64, %rsp
	movq	%rdi, -72(%rbp)
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	.p2align 4,,10
	.p2align 3
.L8:
	vmovss	(%r15), %xmm4
	vmovss	4(%r15), %xmm9
	vmovss	8(%r15), %xmm11
	testl	%ebx, %ebx
	jle	.L4
	cmpl	$6, %r10d
	jbe	.L28
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
	cmpl	%ebx, %r9d
	je	.L4
	movl	%r9d, %eax
.L5:
	movslq	%eax, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	1(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L4
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L4
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L4
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L4
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L4
	movslq	%ecx, %rcx
	addl	$6, %eax
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
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
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %r15
	vmovss	%xmm0, res(%rip)
	cmpq	%rsi, %r15
	jne	.L8
	movl	%r9d, -104(%rbp)
	movl	%r10d, -96(%rbp)
	movq	%r8, -88(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %r13
	vmovdqa	.LC2(%rip), %ymm15
	vmovdqa	.LC3(%rip), %ymm14
	movl	-104(%rbp), %r9d
	movq	%rax, -80(%rbp)
	vmovdqa	.LC5(%rip), %ymm13
	vmovdqa	.LC9(%rip), %ymm12
	movl	-96(%rbp), %r10d
	movq	-88(%rbp), %r8
	.p2align 4,,10
	.p2align 3
.L13:
	movl	0(%r13), %esi
	movl	4(%r13), %edi
	movl	8(%r13), %r11d
	testl	%ebx, %ebx
	jle	.L9
	cmpl	$6, %r10d
	jbe	.L29
	vmovd	%esi, %xmm7
	movl	$r2inv, %ecx
	movl	$position, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm7, %ymm9
	vmovd	%edi, %xmm7
	vmovdqa	.LC12(%rip), %ymm6
	vmovdqa	.LC13(%rip), %ymm5
	vbroadcastss	%xmm7, %ymm8
	vmovd	%r11d, %xmm7
	vbroadcastss	%xmm7, %ymm7
	.p2align 4,,10
	.p2align 3
.L11:
	vmovaps	(%rax), %ymm0
	vmovaps	32(%rax), %ymm11
	addl	$1, %edx
	addq	$32, %rcx
	vmovaps	64(%rax), %ymm10
	addq	$96, %rax
	vpshufb	%ymm15, %ymm0, %ymm1
	vpshufb	%ymm14, %ymm0, %ymm2
	vpshufb	.LC4(%rip), %ymm11, %ymm4
	vpshufb	.LC7(%rip), %ymm0, %ymm3
	vpermq	$78, %ymm1, %ymm1
	vpor	%ymm1, %ymm2, %ymm1
	vpshufb	.LC6(%rip), %ymm0, %ymm2
	vpor	%ymm4, %ymm1, %ymm1
	vpermd	%ymm10, %ymm13, %ymm4
	vpermq	$78, %ymm2, %ymm2
	vblendps	$192, %ymm4, %ymm1, %ymm1
	vpor	%ymm2, %ymm3, %ymm2
	vpshufb	.LC10(%rip), %ymm0, %ymm3
	vsubps	%ymm1, %ymm9, %ymm4
	vpshufb	.LC8(%rip), %ymm11, %ymm1
	vpshufb	%ymm6, %ymm11, %ymm11
	vpor	%ymm1, %ymm2, %ymm2
	vpermd	%ymm10, %ymm12, %ymm1
	vblendps	$224, %ymm1, %ymm2, %ymm2
	vpermq	$78, %ymm3, %ymm1
	vpshufb	.LC11(%rip), %ymm0, %ymm3
	vsubps	%ymm2, %ymm8, %ymm2
	vpor	%ymm1, %ymm3, %ymm3
	vpermd	%ymm10, %ymm5, %ymm1
	vpor	%ymm11, %ymm3, %ymm3
	vblendps	$224, %ymm1, %ymm3, %ymm3
	vmulps	%ymm2, %ymm2, %ymm2
	vsubps	%ymm3, %ymm7, %ymm0
	vfmadd132ps	%ymm4, %ymm2, %ymm4
	vfmadd132ps	%ymm0, %ymm4, %ymm0
	vmovaps	%ymm0, -32(%rcx)
	cmpl	%edx, %r12d
	ja	.L11
	cmpl	%ebx, %r9d
	je	.L9
	movl	%r9d, %eax
.L10:
	leal	(%rax,%rax,2), %edx
	vmovd	%esi, %xmm7
	leal	1(%rax), %ecx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%edi, %xmm7
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmovd	%r11d, %xmm7
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	movslq	%eax, %rdx
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	cmpl	%ecx, %ebx
	jle	.L9
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm7
	vmovd	%edi, %xmm5
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%r11d, %xmm7
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L9
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm5
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm5, %xmm0
	addq	$1, %rdx
	vmovd	%edi, %xmm5
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L9
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm5
	vmovd	%edi, %xmm7
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm5, %xmm0
	addq	$1, %rdx
	vmovd	%r11d, %xmm5
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L9
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm7
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%edi, %xmm7
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L9
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm7
	vmovd	%r11d, %xmm6
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vmovd	%edi, %xmm5
	addl	$6, %eax
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm6, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %ebx
	jle	.L9
	leal	(%rax,%rax,2), %edx
	cltq
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm6, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rax,4)
.L9:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %r13
	vmovss	%xmm0, res(%rip)
	cmpq	%r15, %r13
	jne	.L13
	movl	%r9d, -104(%rbp)
	movl	%r10d, -96(%rbp)
	movq	%r8, -88(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	-80(%rbp), %rax
	movabsq	$2361183241434822607, %rdx
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -80(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %rdi
	movl	-104(%rbp), %r9d
	movq	%rax, %r15
	movl	-96(%rbp), %eax
	movq	-88(%rbp), %r8
	movq	%rax, %r10
	leaq	4(,%rax,4), %rcx
	.p2align 4,,10
	.p2align 3
.L16:
	vmovss	(%rdi), %xmm4
	vmovss	4(%rdi), %xmm3
	testl	%ebx, %ebx
	jle	.L14
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L15:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rsi
	leaq	position(,%rsi,4), %rdx
	vsubss	position(,%rsi,4), %xmm4, %xmm1
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L15
.L14:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %rdi
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdi
	jne	.L16
	movl	%r9d, -112(%rbp)
	movl	%r10d, -108(%rbp)
	movq	%rcx, -104(%rbp)
	movq	%r8, -96(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r15, %rax
	movq	%rax, %rsi
	imulq	%rdx
	movq	%rsi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -88(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %rdi
	movl	-112(%rbp), %r9d
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	movq	%rax, %r15
	movl	-108(%rbp), %r10d
	movq	-104(%rbp), %rcx
	movq	-96(%rbp), %r8
	.p2align 4,,10
	.p2align 3
.L21:
	vmovss	(%rdi), %xmm3
	vmovss	4(%rdi), %xmm4
	vmovss	8(%rdi), %xmm5
	testl	%ebx, %ebx
	jle	.L17
	cmpl	$6, %r10d
	jbe	.L30
	vbroadcastss	%xmm3, %ymm9
	vbroadcastss	%xmm4, %ymm8
	xorl	%eax, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm5, %ymm7
	.p2align 4,,10
	.p2align 3
.L19:
	vmovaps	%ymm6, %ymm0
	vmovaps	%ymm6, %ymm13
	addl	$1, %edx
	addq	$32, %rax
	vpmulld	neighList-32(%rax), %ymm10, %ymm11
	vgatherdps	%ymm0, position(,%ymm11,4), %ymm1
	vmovaps	%ymm6, %ymm0
	vgatherdps	%ymm0, position+4(,%ymm11,4), %ymm2
	vsubps	%ymm1, %ymm9, %ymm1
	vgatherdps	%ymm13, position+8(,%ymm11,4), %ymm0
	vsubps	%ymm2, %ymm8, %ymm2
	vsubps	%ymm0, %ymm7, %ymm0
	vmulps	%ymm2, %ymm2, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edx, %r12d
	ja	.L19
	cmpl	%ebx, %r9d
	je	.L17
	movl	%r9d, %eax
.L18:
	movslq	%eax, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	1(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L17
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	2(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L17
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	3(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L17
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	4(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L17
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	5(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L17
	movslq	%esi, %rsi
	addl	$6, %eax
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%eax, %ebx
	jle	.L17
	cltq
	movl	neighList(,%rax,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm3
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm5
	vsubss	position(,%rdx,4), %xmm4, %xmm4
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, r2inv(,%rax,4)
.L17:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %rdi
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdi
	jne	.L21
	movq	%rcx, -96(%rbp)
	movq	%r8, -72(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r15, %rax
	movq	%rax, %rsi
	imulq	%rdx
	movq	%rsi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r12
	subq	%rax, %r12
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-96(%rbp), %rcx
	movq	-72(%rbp), %r8
	movq	%rax, %r15
	.p2align 4,,10
	.p2align 3
.L24:
	vmovss	(%r14), %xmm3
	vmovss	4(%r14), %xmm4
	vmovss	8(%r14), %xmm5
	testl	%ebx, %ebx
	jle	.L22
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L23:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rsi
	leaq	position(,%rsi,4), %rdx
	vsubss	position(,%rsi,4), %xmm3, %xmm1
	vsubss	4(%rdx), %xmm4, %xmm2
	vsubss	8(%rdx), %xmm5, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L23
.L22:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %r14
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %r14
	jne	.L24
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movl	$.LC14, %esi
	movl	$_ZSt4cout, %edi
	movabsq	$2361183241434822607, %rdx
	subq	%r15, %rax
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
	movq	-80(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-50(%rbp), %rsi
	movb	$32, -50(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-88(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-51(%rbp), %rsi
	movb	$32, -51(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%r12, %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-52(%rbp), %rsi
	movb	$32, -52(%rbp)
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
	je	.L44
	cmpb	$0, 56(%rbx)
	je	.L26
	movsbl	67(%rbx), %esi
.L27:
	movq	%r12, %rdi
	call	_ZNSo3putEc
	movq	%rax, %rdi
	call	_ZNSo5flushEv
	addq	$64, %rsp
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
.L28:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L5
	.p2align 4,,10
	.p2align 3
.L30:
	xorl	%eax, %eax
	jmp	.L18
	.p2align 4,,10
	.p2align 3
.L29:
	xorl	%eax, %eax
	jmp	.L10
	.p2align 4,,10
	.p2align 3
.L26:
	movq	%rbx, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	$_ZNKSt5ctypeIcE8do_widenEc, %rax
	je	.L27
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L27
.L44:
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
.L46:
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
	jne	.L46
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
.L49:
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
	jne	.L49
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
.L52:
	movl	k(%rdx), %eax
	addq	$4, %rdx
	leal	(%rax,%rax,2), %eax
	cltq
	vmovss	ff(,%rax,4), %xmm0
	vaddss	ff+4(,%rax,4), %xmm0, %xmm0
	vaddss	ff+8(,%rax,4), %xmm0, %xmm0
	vmovss	%xmm0, g-4(%rdx)
	cmpq	$4096, %rdx
	jne	.L52
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
.L55:
	movslq	k(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rdx
	vmovss	f3(,%rdx,4), %xmm0
	vaddss	f3+4(,%rdx,4), %xmm0, %xmm0
	vaddss	f3+8(,%rdx,4), %xmm0, %xmm0
	vmovss	%xmm0, g-4(%rax)
	cmpq	$4096, %rax
	jne	.L55
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
	jle	.L58
	leal	-1(%rdi), %r10d
	vmovd	%xmm1, %esi
	movl	-32(%rbp), %r8d
	movl	-28(%rbp), %r9d
	cmpl	$6, %r10d
	jbe	.L62
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
.L60:
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
	ja	.L60
	movl	%edi, %eax
	andl	$-8, %eax
	cmpl	%edi, %eax
	je	.L68
	vzeroupper
.L59:
	leal	(%rax,%rax,2), %edx
	vmovd	%r8d, %xmm7
	leal	1(%rax), %ecx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%r9d, %xmm7
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmovd	%esi, %xmm7
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	movslq	%eax, %rdx
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	cmpl	%ecx, %edi
	jle	.L58
	leal	(%rcx,%rcx,2), %edx
	vmovd	%r8d, %xmm7
	vmovd	%r9d, %xmm5
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%esi, %xmm7
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L58
	leal	(%rcx,%rcx,2), %edx
	vmovd	%r8d, %xmm5
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm5, %xmm0
	addq	$1, %rdx
	vmovd	%r9d, %xmm5
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L58
	leal	(%rcx,%rcx,2), %edx
	vmovd	%r8d, %xmm5
	vmovd	%r9d, %xmm7
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm5, %xmm0
	addq	$1, %rdx
	vmovd	%esi, %xmm5
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L58
	leal	(%rcx,%rcx,2), %edx
	vmovd	%r8d, %xmm1
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm1, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %edi
	jle	.L58
	leal	(%rcx,%rcx,2), %edx
	vmovd	%r8d, %xmm1
	vmovd	%r9d, %xmm7
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	addl	$6, %eax
	vsubss	position(,%rdx,4), %xmm1, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %edi
	jle	.L58
	leal	(%rax,%rax,2), %edx
	vmovd	%r8d, %xmm1
	cltq
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm1, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm1
	vsubss	position(,%rdx,4), %xmm7, %xmm2
	vmulss	%xmm1, %xmm1, %xmm1
	vfmadd231ss	%xmm2, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rax,4)
.L58:
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
.L68:
	.cfi_restore_state
	vzeroupper
	jmp	.L58
	.p2align 4,,10
	.p2align 3
.L62:
	xorl	%eax, %eax
	jmp	.L59
	.cfi_endproc
.LFE4:
	.size	_Z3seq6float3iii, .-_Z3seq6float3iii
	.p2align 4,,15
	.globl	_Z3bar6float3iii
	.type	_Z3bar6float3iii, @function
_Z3bar6float3iii:
.LFB5:
	.cfi_startproc
	vmovq	%xmm0, -16(%rsp)
	vmovss	-12(%rsp), %xmm3
	testl	%edi, %edi
	jle	.L75
	leal	-1(%rdi), %eax
	vmovss	-16(%rsp), %xmm4
	movq	%rax, %rdi
	leaq	4(,%rax,4), %rsi
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L71:
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
	jne	.L71
.L70:
	movslq	%edi, %rdi
	vmovss	r2inv(,%rdi,4), %xmm0
	vmovss	%xmm0, res(%rip)
	ret
	.p2align 4,,10
	.p2align 3
.L75:
	subl	$1, %edi
	jmp	.L70
	.cfi_endproc
.LFE5:
	.size	_Z3bar6float3iii, .-_Z3bar6float3iii
	.p2align 4,,15
	.globl	_Z4bar26float3iii
	.type	_Z4bar26float3iii, @function
_Z4bar26float3iii:
.LFB6:
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
	jle	.L77
	leal	-1(%rdi), %ecx
	vmovss	-32(%rbp), %xmm10
	vmovss	-28(%rbp), %xmm11
	cmpl	$6, %ecx
	jbe	.L81
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
.L79:
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
	ja	.L79
	movl	%edi, %eax
	andl	$-8, %eax
	cmpl	%edi, %eax
	je	.L87
	vzeroupper
.L78:
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
	jle	.L77
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
	jle	.L77
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
	jle	.L77
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
	jle	.L77
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
	jle	.L77
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
	jle	.L77
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
.L77:
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
.L87:
	.cfi_restore_state
	vzeroupper
	jmp	.L77
	.p2align 4,,10
	.p2align 3
.L81:
	xorl	%eax, %eax
	jmp	.L78
	.cfi_endproc
.LFE6:
	.size	_Z4bar26float3iii, .-_Z4bar26float3iii
	.p2align 4,,15
	.globl	_Z3foo6float3iii
	.type	_Z3foo6float3iii, @function
_Z3foo6float3iii:
.LFB7:
	.cfi_startproc
	vmovq	%xmm0, -16(%rsp)
	testl	%edi, %edi
	jle	.L94
	leal	-1(%rdi), %eax
	vmovss	-16(%rsp), %xmm5
	vmovss	-12(%rsp), %xmm4
	movq	%rax, %rdi
	leaq	4(,%rax,4), %rsi
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L90:
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
	jne	.L90
.L89:
	movslq	%edi, %rdi
	vmovss	r2inv(,%rdi,4), %xmm0
	vmovss	%xmm0, res(%rip)
	ret
	.p2align 4,,10
	.p2align 3
.L94:
	subl	$1, %edi
	jmp	.L89
	.cfi_endproc
.LFE7:
	.size	_Z3foo6float3iii, .-_Z3foo6float3iii
	.p2align 4,,15
	.globl	_Z5time3PK6float3iiii
	.type	_Z5time3PK6float3iiii, @function
_Z5time3PK6float3iiii:
.LFB2099:
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
	movq	%rdi, -72(%rbp)
	testl	%esi, %esi
	jle	.L96
	leal	-1(%rsi), %eax
	leal	-1(%rdx), %r10d
	movl	%edx, %r12d
	movl	%edx, %r9d
	leaq	3(%rax,%rax,2), %rax
	movq	%rdi, %r15
	movq	%rdi, %r14
	shrl	$3, %r12d
	vmovdqa	.LC0(%rip), %ymm10
	leaq	(%rdi,%rax,4), %rsi
	andl	$-8, %r9d
	movslq	%r10d, %r8
	vmovaps	.LC1(%rip), %ymm6
	.p2align 4,,10
	.p2align 3
.L101:
	vmovss	(%r15), %xmm4
	vmovss	4(%r15), %xmm9
	vmovss	8(%r15), %xmm11
	testl	%ebx, %ebx
	jle	.L97
	cmpl	$6, %r10d
	jbe	.L126
	vbroadcastss	%xmm4, %ymm8
	vbroadcastss	%xmm9, %ymm7
	xorl	%eax, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm11, %ymm5
	.p2align 4,,10
	.p2align 3
.L99:
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
	ja	.L99
	cmpl	%ebx, %r9d
	je	.L97
	movl	%r9d, %eax
.L98:
	movslq	%eax, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	1(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L97
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L97
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L97
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L97
	movslq	%ecx, %rcx
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L97
	movslq	%ecx, %rcx
	addl	$6, %eax
	movl	neighList(,%rcx,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm4, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm11, %xmm2
	vsubss	position(,%rdx,4), %xmm9, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %ebx
	jle	.L97
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
.L97:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %r15
	vmovss	%xmm0, res(%rip)
	cmpq	%rsi, %r15
	jne	.L101
	movq	%r8, -104(%rbp)
	movl	%r10d, -96(%rbp)
	movl	%r9d, -88(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %r13
	vmovdqa	.LC2(%rip), %ymm15
	vmovdqa	.LC3(%rip), %ymm14
	movl	-88(%rbp), %r9d
	movq	%rax, -80(%rbp)
	vmovdqa	.LC5(%rip), %ymm13
	vmovdqa	.LC9(%rip), %ymm12
	movl	-96(%rbp), %r10d
	movq	-104(%rbp), %r8
	.p2align 4,,10
	.p2align 3
.L107:
	movl	0(%r13), %esi
	movl	4(%r13), %edi
	movl	8(%r13), %r11d
	testl	%ebx, %ebx
	jle	.L103
	cmpl	$6, %r10d
	jbe	.L127
	vmovd	%esi, %xmm7
	movl	$r2inv, %ecx
	movl	$position, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm7, %ymm9
	vmovd	%edi, %xmm7
	vmovdqa	.LC12(%rip), %ymm6
	vmovdqa	.LC13(%rip), %ymm5
	vbroadcastss	%xmm7, %ymm8
	vmovd	%r11d, %xmm7
	vbroadcastss	%xmm7, %ymm7
	.p2align 4,,10
	.p2align 3
.L105:
	vmovaps	(%rax), %ymm0
	vmovaps	32(%rax), %ymm11
	addl	$1, %edx
	addq	$32, %rcx
	vmovaps	64(%rax), %ymm10
	addq	$96, %rax
	vpshufb	%ymm15, %ymm0, %ymm1
	vpshufb	%ymm14, %ymm0, %ymm2
	vpshufb	.LC4(%rip), %ymm11, %ymm4
	vpshufb	.LC7(%rip), %ymm0, %ymm3
	vpermq	$78, %ymm1, %ymm1
	vpor	%ymm1, %ymm2, %ymm1
	vpshufb	.LC6(%rip), %ymm0, %ymm2
	vpor	%ymm4, %ymm1, %ymm1
	vpermd	%ymm10, %ymm13, %ymm4
	vpermq	$78, %ymm2, %ymm2
	vblendps	$192, %ymm4, %ymm1, %ymm1
	vpor	%ymm2, %ymm3, %ymm2
	vpshufb	.LC10(%rip), %ymm0, %ymm3
	vsubps	%ymm1, %ymm9, %ymm4
	vpshufb	.LC8(%rip), %ymm11, %ymm1
	vpshufb	%ymm6, %ymm11, %ymm11
	vpor	%ymm1, %ymm2, %ymm2
	vpermd	%ymm10, %ymm12, %ymm1
	vblendps	$224, %ymm1, %ymm2, %ymm2
	vpermq	$78, %ymm3, %ymm1
	vpshufb	.LC11(%rip), %ymm0, %ymm3
	vsubps	%ymm2, %ymm8, %ymm2
	vpor	%ymm1, %ymm3, %ymm3
	vpermd	%ymm10, %ymm5, %ymm1
	vpor	%ymm11, %ymm3, %ymm3
	vblendps	$224, %ymm1, %ymm3, %ymm3
	vmulps	%ymm2, %ymm2, %ymm2
	vsubps	%ymm3, %ymm7, %ymm0
	vfmadd132ps	%ymm4, %ymm2, %ymm4
	vfmadd132ps	%ymm0, %ymm4, %ymm0
	vmovaps	%ymm0, -32(%rcx)
	cmpl	%edx, %r12d
	ja	.L105
	cmpl	%ebx, %r9d
	je	.L103
	movl	%r9d, %eax
.L104:
	leal	(%rax,%rax,2), %edx
	vmovd	%esi, %xmm7
	leal	1(%rax), %ecx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%edi, %xmm7
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmovd	%r11d, %xmm7
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	movslq	%eax, %rdx
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rdx,4)
	cmpl	%ecx, %ebx
	jle	.L103
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm7
	vmovd	%edi, %xmm5
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%r11d, %xmm7
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	2(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L103
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm5
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm5, %xmm0
	addq	$1, %rdx
	vmovd	%edi, %xmm5
	vsubss	position+4(,%rdx,4), %xmm7, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	3(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L103
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm5
	vmovd	%edi, %xmm7
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm5, %xmm0
	addq	$1, %rdx
	vmovd	%r11d, %xmm5
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	4(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L103
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm7
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vmovd	%edi, %xmm7
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm7, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	leal	5(%rax), %ecx
	cmpl	%ecx, %ebx
	jle	.L103
	leal	(%rcx,%rcx,2), %edx
	vmovd	%esi, %xmm7
	vmovd	%r11d, %xmm6
	movslq	%ecx, %rcx
	movslq	%edx, %rdx
	vmovd	%edi, %xmm5
	addl	$6, %eax
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm6, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rcx,4)
	cmpl	%eax, %ebx
	jle	.L103
	leal	(%rax,%rax,2), %edx
	cltq
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm7, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm6, %xmm2
	vsubss	position(,%rdx,4), %xmm5, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rax,4)
.L103:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %r13
	vmovss	%xmm0, res(%rip)
	cmpq	%r15, %r13
	jne	.L107
	movq	%r8, -104(%rbp)
	movl	%r10d, -96(%rbp)
	movl	%r9d, -88(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	subq	-80(%rbp), %rax
	movabsq	$2361183241434822607, %rdx
	movq	%rax, %rcx
	imulq	%rdx
	movq	%rcx, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -80(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %rdi
	movl	-88(%rbp), %r9d
	movq	%rax, %r15
	movl	-96(%rbp), %eax
	movq	-104(%rbp), %r8
	movq	%rax, %r10
	leaq	4(,%rax,4), %rcx
	.p2align 4,,10
	.p2align 3
.L111:
	vmovss	(%rdi), %xmm4
	vmovss	4(%rdi), %xmm3
	testl	%ebx, %ebx
	jle	.L109
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L110:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rsi
	leaq	position(,%rsi,4), %rdx
	vsubss	position(,%rsi,4), %xmm4, %xmm1
	vsubss	4(%rdx), %xmm3, %xmm2
	vsubss	8(%rdx), %xmm3, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L110
.L109:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %rdi
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdi
	jne	.L111
	movq	%r8, -120(%rbp)
	movq	%rcx, -112(%rbp)
	movl	%r10d, -104(%rbp)
	movl	%r9d, -96(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r15, %rax
	movq	%rax, %rsi
	imulq	%rdx
	movq	%rsi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	subq	%rax, %rdx
	movq	%rdx, -88(%rbp)
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %rdi
	movl	-96(%rbp), %r9d
	vmovdqa	.LC0(%rip), %ymm10
	vmovaps	.LC1(%rip), %ymm6
	movq	%rax, %r15
	movl	-104(%rbp), %r10d
	movq	-112(%rbp), %rcx
	movq	-120(%rbp), %r8
	.p2align 4,,10
	.p2align 3
.L117:
	vmovss	(%rdi), %xmm3
	vmovss	4(%rdi), %xmm4
	vmovss	8(%rdi), %xmm5
	testl	%ebx, %ebx
	jle	.L113
	cmpl	$6, %r10d
	jbe	.L128
	vbroadcastss	%xmm3, %ymm9
	vbroadcastss	%xmm4, %ymm8
	xorl	%eax, %eax
	xorl	%edx, %edx
	vbroadcastss	%xmm5, %ymm7
	.p2align 4,,10
	.p2align 3
.L115:
	vmovaps	%ymm6, %ymm0
	vmovaps	%ymm6, %ymm13
	addl	$1, %edx
	addq	$32, %rax
	vpmulld	neighList-32(%rax), %ymm10, %ymm11
	vgatherdps	%ymm0, position(,%ymm11,4), %ymm1
	vmovaps	%ymm6, %ymm0
	vgatherdps	%ymm0, position+4(,%ymm11,4), %ymm2
	vsubps	%ymm1, %ymm9, %ymm1
	vgatherdps	%ymm13, position+8(,%ymm11,4), %ymm0
	vsubps	%ymm2, %ymm8, %ymm2
	vsubps	%ymm0, %ymm7, %ymm0
	vmulps	%ymm2, %ymm2, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, r2inv-32(%rax)
	cmpl	%edx, %r12d
	ja	.L115
	cmpl	%ebx, %r9d
	je	.L113
	movl	%r9d, %eax
.L114:
	movslq	%eax, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	1(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L113
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	2(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L113
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	3(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L113
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	4(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L113
	movslq	%esi, %rsi
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	leal	5(%rax), %esi
	cmpl	%esi, %ebx
	jle	.L113
	movslq	%esi, %rsi
	addl	$6, %eax
	movl	neighList(,%rsi,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm0
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm2
	vsubss	position(,%rdx,4), %xmm4, %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv(,%rsi,4)
	cmpl	%eax, %ebx
	jle	.L113
	cltq
	movl	neighList(,%rax,4), %edx
	leal	(%rdx,%rdx,2), %edx
	movslq	%edx, %rdx
	vsubss	position(,%rdx,4), %xmm3, %xmm3
	addq	$1, %rdx
	vsubss	position+4(,%rdx,4), %xmm5, %xmm5
	vsubss	position(,%rdx,4), %xmm4, %xmm4
	vmulss	%xmm5, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vfmadd132ss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm3, r2inv(,%rax,4)
.L113:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %rdi
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %rdi
	jne	.L117
	movq	%r8, -96(%rbp)
	movq	%rcx, -72(%rbp)
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movabsq	$2361183241434822607, %rdx
	subq	%r15, %rax
	movq	%rax, %rsi
	imulq	%rdx
	movq	%rsi, %rax
	sarq	$63, %rax
	sarq	$7, %rdx
	movq	%rdx, %r12
	subq	%rax, %r12
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movq	-72(%rbp), %rcx
	movq	-96(%rbp), %r8
	movq	%rax, %r15
	.p2align 4,,10
	.p2align 3
.L121:
	vmovss	(%r14), %xmm3
	vmovss	4(%r14), %xmm5
	vmovss	8(%r14), %xmm4
	testl	%ebx, %ebx
	jle	.L119
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L120:
	movslq	neighList(%rax), %rdx
	addq	$4, %rax
	leaq	(%rdx,%rdx,2), %rsi
	leaq	position(,%rsi,4), %rdx
	vsubss	position(,%rsi,4), %xmm3, %xmm1
	vsubss	4(%rdx), %xmm5, %xmm2
	vsubss	8(%rdx), %xmm4, %xmm0
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vmovss	%xmm0, r2inv-4(%rax)
	cmpq	%rcx, %rax
	jne	.L120
.L119:
	vmovss	r2inv(,%r8,4), %xmm0
	addq	$12, %r14
	vmovss	%xmm0, res(%rip)
	cmpq	%r13, %r14
	jne	.L121
.L125:
	call	_ZNSt6chrono3_V212system_clock3nowEv
	movl	$.LC14, %esi
	movl	$_ZSt4cout, %edi
	movabsq	$2361183241434822607, %rdx
	subq	%r15, %rax
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
	leaq	-52(%rbp), %rsi
	movb	$32, -52(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-80(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZNSo9_M_insertIlEERSoT_
	movl	$1, %edx
	leaq	-51(%rbp), %rsi
	movb	$32, -51(%rbp)
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	-88(%rbp), %rsi
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
	je	.L148
	cmpb	$0, 56(%rbx)
	je	.L123
	movsbl	67(%rbx), %esi
.L124:
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
.L126:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L98
	.p2align 4,,10
	.p2align 3
.L128:
	xorl	%eax, %eax
	jmp	.L114
	.p2align 4,,10
	.p2align 3
.L127:
	xorl	%eax, %eax
	jmp	.L104
	.p2align 4,,10
	.p2align 3
.L123:
	movq	%rbx, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	$_ZNKSt5ctypeIcE8do_widenEc, %rax
	je	.L124
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	.L124
.L96:
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
	movq	%rdx, -80(%rbp)
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
	movq	%rdx, -88(%rbp)
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
	movq	%rax, %r15
	jmp	.L125
.L148:
	call	_ZSt16__throw_bad_castv
	.cfi_endproc
.LFE2099:
	.size	_Z5time3PK6float3iiii, .-_Z5time3PK6float3iiii
	.section	.rodata.str1.1
.LC18:
	.string	"tivially sequential"
.LC20:
	.string	"random"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB2108:
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
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x60,0x6
	.cfi_escape 0x10,0xe,0x2,0x76,0x78
	.cfi_escape 0x10,0xd,0x2,0x76,0x70
	.cfi_escape 0x10,0xc,0x2,0x76,0x68
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x58
	movl	$neighList, %ebx
	subq	$1200024, %rsp
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movq	%rsp, %r12
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	vmovdqa	.LC17(%rip), %ymm0
	movl	$neighList+4096, %eax
	vmovdqa	.LC19(%rip), %ymm1
	.p2align 4,,10
	.p2align 3
.L150:
	vmovdqa	%ymm0, (%rbx)
	addq	$32, %rbx
	vpaddd	%ymm1, %ymm0, %ymm0
	cmpq	%rbx, %rax
	jne	.L150
	movl	$9, %r14d
	movl	$2, %r13d
	vzeroupper
	.p2align 4,,10
	.p2align 3
.L151:
	movl	%r13d, %esi
	movq	%r12, %rdi
	addl	%r13d, %r13d
	call	_Z5time3PK6float3iiii.constprop.3
	subl	$1, %r14d
	jne	.L151
	movl	$6, %edx
	movl	$.LC20, %esi
	movl	$_ZSt4cout, %edi
	subq	$neighList+8, %rbx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$neighList+4, %r13d
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movq	%rbx, %rax
	shrq	$2, %rax
	leaq	neighList+8(,%rax,4), %rbx
	.p2align 4,,10
	.p2align 3
.L152:
	call	rand
	movq	%r13, %rcx
	subq	$neighList, %rcx
	cltq
	sarq	$2, %rcx
	cqto
	addq	$1, %rcx
	idivq	%rcx
	leaq	neighList(,%rdx,4), %rax
	cmpq	%r13, %rax
	je	.L153
	movl	neighList(,%rdx,4), %ecx
	movl	0(%r13), %eax
	addq	$4, %r13
	movl	%ecx, -4(%r13)
	movl	%eax, neighList(,%rdx,4)
	cmpq	%r13, %rbx
	jne	.L152
.L155:
	movl	$9, %r13d
	movl	$2, %ebx
	.p2align 4,,10
	.p2align 3
.L156:
	movl	%ebx, %esi
	movq	%r12, %rdi
	addl	%ebx, %ebx
	call	_Z5time3PK6float3iiii.constprop.3
	subl	$1, %r13d
	jne	.L156
	leaq	-40(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L153:
	.cfi_restore_state
	addq	$4, %r13
	cmpq	%rbx, %r13
	jne	.L152
	jmp	.L155
	.cfi_endproc
.LFE2108:
	.size	main, .-main
	.p2align 4,,15
	.type	_GLOBAL__sub_I_fx, @function
_GLOBAL__sub_I_fx:
.LFB2603:
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
.LFE2603:
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
	.hidden	__dso_handle
	.ident	"GCC: (GNU) 7.0.1 20170326 (experimental) [trunk revision 246485]"
	.section	.note.GNU-stack,"",@progbits
