	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 1
	.align 4
	.globl __ZNSt6vectorIfSaIfEED1Ev
	.weak_definition __ZNSt6vectorIfSaIfEED1Ev
__ZNSt6vectorIfSaIfEED1Ev:
LFB4047:
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	L3
	jmp	__ZdlPv
	.align 4
L3:
	rep; ret
LFE4047:
	.text
	.align 4,0x90
	.globl __Z18chargeDistributioni
__Z18chargeDistributioni:
LFB3763:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	movslq	%edi, %rax
	pushq	-8(%r10)
	leaq	18(,%rax,4), %rax
	andq	$-16, %rax
	pushq	%rbp
LCFI1:
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
LCFI2:
	pushq	%rbx
	subq	$64, %rsp
LCFI3:
	movl	%edi, -52(%rbp)
	subq	%rax, %rsp
	movq	%rsp, %r15
	subq	%rax, %rsp
	movq	%rsp, %r14
	movq	%rsp, -80(%rbp)
	subq	%rax, %rsp
	movq	%rsp, -64(%rbp)
	subq	%rax, %rsp
	leaq	3(%rsp), %rbx
	shrq	$2, %rbx
	testl	%edi, %edi
	leaq	0(,%rbx,4), %rdx
	movq	%rdx, -72(%rbp)
	je	L6
	movl	%edi, %eax
	xorl	%r13d, %r13d
	subl	$1, %eax
	leaq	4(,%rax,4), %r12
	.align 4,0x90
L8:
	leaq	(%r14,%r13), %rsi
	leaq	(%r15,%r13), %rdi
	addq	$4, %r13
	call	__Z4compRfS_
	cmpq	%r12, %r13
	jne	L8
	movl	_Nstrips(%rip), %eax
	movl	-52(%rbp), %ecx
	vmovss	_Nsigma(%rip), %xmm3
	movl	%eax, -92(%rbp)
	movq	-64(%rbp), %rax
	movl	%ecx, %esi
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	cmpl	%eax, %ecx
	cmovbe	%ecx, %eax
	cmpl	$8, %ecx
	ja	L209
L97:
	movl	-92(%rbp), %r8d
	xorl	%eax, %eax
	xorl	%edi, %edi
	movq	-80(%rbp), %r9
	movq	-64(%rbp), %r10
	movq	-72(%rbp), %r11
	.align 4,0x90
L12:
	vmulss	(%r9,%rax,4), %xmm3, %xmm1
	vmovss	(%r15,%rax,4), %xmm0
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %edx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %ecx
	testl	%edx, %edx
	cmovs	%edi, %edx
	cmpl	%r8d, %ecx
	cmovg	%r8d, %ecx
	movl	%edx, (%r10,%rax,4)
	subl	%edx, %ecx
	movl	%ecx, (%r11,%rax,4)
	leal	1(%rax), %edx
	addq	$1, %rax
	cmpl	%eax, %esi
	ja	L12
	cmpl	%esi, -52(%rbp)
	je	L13
L10:
	movl	-52(%rbp), %r11d
	movl	%esi, %r10d
	subl	%esi, %r11d
	movl	%r11d, %edi
	shrl	$3, %edi
	leal	0(,%rdi,8), %ecx
	testl	%ecx, %ecx
	je	L14
	movq	-80(%rbp), %rax
	salq	$2, %r10
	vbroadcastss	%xmm3, %ymm6
	leaq	(%r15,%r10), %r9
	vbroadcastss	-92(%rbp), %ymm5
	xorl	%esi, %esi
	vpxor	%xmm4, %xmm4, %xmm4
	leaq	(%rax,%r10), %r8
	movq	-64(%rbp), %rax
	leaq	(%rax,%r10), %r12
	addq	-72(%rbp), %r10
	xorl	%eax, %eax
L19:
	vmovups	(%r9,%rax), %xmm1
	addl	$1, %esi
	vmovups	(%r8,%rax), %xmm2
	vinsertf128	$0x1, 16(%r9,%rax), %ymm1, %ymm1
	vinsertf128	$0x1, 16(%r8,%rax), %ymm2, %ymm2
	vmulps	%ymm6, %ymm2, %ymm2
	vsubps	%ymm2, %ymm1, %ymm0
	vaddps	%ymm1, %ymm2, %ymm1
	vroundps	$1, %ymm0, %ymm0
	vcvttps2dq	%ymm0, %ymm0
	vpmaxsd	%ymm4, %ymm0, %ymm0
	vroundps	$2, %ymm1, %ymm1
	vcvttps2dq	%ymm1, %ymm1
	vpminsd	%ymm5, %ymm1, %ymm1
	vmovdqa	%ymm0, (%r12,%rax)
	vpsubd	%ymm0, %ymm1, %ymm0
	vmovdqu	%xmm0, (%r10,%rax)
	vextracti128	$0x1, %ymm0, 16(%r10,%rax)
	addq	$32, %rax
	cmpl	%edi, %esi
	jb	L19
	addl	%ecx, %edx
	cmpl	%ecx, %r11d
	je	L17
	.align 4,0x90
L14:
	movq	-80(%rbp), %r9
	movslq	%edx, %rsi
	xorl	%eax, %eax
	vmovss	(%r15,%rsi,4), %xmm0
	movl	-92(%rbp), %r14d
	movq	-64(%rbp), %r11
	vmulss	(%r9,%rsi,4), %xmm3, %xmm1
	movq	-72(%rbp), %r10
	movl	-52(%rbp), %r8d
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %edi
	testl	%ecx, %ecx
	cmovs	%eax, %ecx
	cmpl	%r14d, %edi
	cmovg	%r14d, %edi
	movl	%ecx, (%r11,%rsi,4)
	subl	%ecx, %edi
	movl	%edi, (%r10,%rsi,4)
	leal	1(%rdx), %esi
	cmpl	%esi, %r8d
	je	L17
	movslq	%esi, %rsi
	vmulss	(%r9,%rsi,4), %xmm3, %xmm1
	vmovss	(%r15,%rsi,4), %xmm0
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %edi
	testl	%ecx, %ecx
	cmovs	%eax, %ecx
	cmpl	%r14d, %edi
	cmovg	%r14d, %edi
	movl	%ecx, (%r11,%rsi,4)
	subl	%ecx, %edi
	movl	%edi, (%r10,%rsi,4)
	leal	2(%rdx), %esi
	cmpl	%esi, %r8d
	je	L17
	movslq	%esi, %rsi
	vmulss	(%r9,%rsi,4), %xmm3, %xmm1
	vmovss	(%r15,%rsi,4), %xmm0
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %edi
	testl	%ecx, %ecx
	cmovs	%eax, %ecx
	cmpl	%r14d, %edi
	cmovg	%r14d, %edi
	movl	%ecx, (%r11,%rsi,4)
	subl	%ecx, %edi
	movl	%edi, (%r10,%rsi,4)
	leal	3(%rdx), %esi
	cmpl	%esi, %r8d
	je	L17
	movslq	%esi, %rsi
	vmulss	(%r9,%rsi,4), %xmm3, %xmm1
	vmovss	(%r15,%rsi,4), %xmm0
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %edi
	testl	%ecx, %ecx
	cmovs	%eax, %ecx
	cmpl	%r14d, %edi
	cmovg	%r14d, %edi
	movl	%ecx, (%r11,%rsi,4)
	subl	%ecx, %edi
	movl	%edi, (%r10,%rsi,4)
	leal	4(%rdx), %esi
	cmpl	%esi, %r8d
	je	L17
	movslq	%esi, %rsi
	vmulss	(%r9,%rsi,4), %xmm3, %xmm1
	vmovss	(%r15,%rsi,4), %xmm0
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %edi
	testl	%ecx, %ecx
	cmovs	%eax, %ecx
	cmpl	%r14d, %edi
	cmovg	%r14d, %edi
	movl	%ecx, (%r11,%rsi,4)
	subl	%ecx, %edi
	movl	%edi, (%r10,%rsi,4)
	leal	5(%rdx), %esi
	cmpl	%esi, %r8d
	je	L17
	movslq	%esi, %rsi
	vmulss	(%r9,%rsi,4), %xmm3, %xmm1
	vmovss	(%r15,%rsi,4), %xmm0
	vsubss	%xmm1, %xmm0, %xmm2
	vaddss	%xmm0, %xmm1, %xmm0
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	vcvttss2si	%xmm0, %edi
	testl	%ecx, %ecx
	cmovs	%eax, %ecx
	cmpl	%r14d, %edi
	cmovg	%r14d, %edi
	addl	$6, %edx
	movl	%ecx, (%r11,%rsi,4)
	subl	%ecx, %edi
	cmpl	%edx, %r8d
	movl	%edi, (%r10,%rsi,4)
	je	L17
	movslq	%edx, %rcx
	movq	-80(%rbp), %rdx
	vmovss	(%r15,%rcx,4), %xmm0
	movl	-92(%rbp), %esi
	vmulss	(%rdx,%rcx,4), %xmm3, %xmm3
	vsubss	%xmm3, %xmm0, %xmm1
	vaddss	%xmm0, %xmm3, %xmm0
	vroundss	$1, %xmm1, %xmm1, %xmm1
	vcvttss2si	%xmm1, %edx
	vroundss	$2, %xmm0, %xmm0, %xmm0
	testl	%edx, %edx
	cmovs	%eax, %edx
	movq	-64(%rbp), %rax
	movl	%edx, (%rax,%rcx,4)
	vcvttss2si	%xmm0, %eax
	cmpl	%esi, %eax
	cmovg	%esi, %eax
	subl	%edx, %eax
	movq	-72(%rbp), %rdx
	movl	%eax, (%rdx,%rcx,4)
L17:
	movq	-72(%rbp), %rax
	movl	-52(%rbp), %ecx
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	cmpl	%eax, %ecx
	cmovbe	%ecx, %eax
	cmpl	$17, %ecx
	ja	L20
L13:
	movl	-52(%rbp), %eax
L21:
	cmpl	$1, %eax
	movl	0(,%rbx,4), %r14d
	je	L104
	addl	4(,%rbx,4), %r14d
	cmpl	$2, %eax
	je	L105
	addl	8(,%rbx,4), %r14d
	cmpl	$3, %eax
	je	L106
	addl	12(,%rbx,4), %r14d
	cmpl	$4, %eax
	je	L107
	addl	16(,%rbx,4), %r14d
	cmpl	$5, %eax
	je	L108
	addl	20(,%rbx,4), %r14d
	cmpl	$6, %eax
	je	L109
	addl	24(,%rbx,4), %r14d
	cmpl	$7, %eax
	je	L110
	addl	28(,%rbx,4), %r14d
	cmpl	$8, %eax
	je	L111
	addl	32(,%rbx,4), %r14d
	cmpl	$9, %eax
	je	L112
	movl	36(,%rbx,4), %edx
	addl	%r14d, %edx
	cmpl	$10, %eax
	movl	%edx, %r14d
	je	L113
	addl	40(,%rbx,4), %edx
	cmpl	$11, %eax
	movl	%edx, %r14d
	je	L114
	addl	44(,%rbx,4), %r14d
	cmpl	$12, %eax
	je	L115
	addl	48(,%rbx,4), %r14d
	cmpl	$13, %eax
	je	L116
	addl	52(,%rbx,4), %r14d
	cmpl	$14, %eax
	je	L117
	addl	56(,%rbx,4), %r14d
	cmpl	$15, %eax
	je	L118
	addl	60(,%rbx,4), %r14d
	cmpl	$17, %eax
	jne	L119
	addl	64(,%rbx,4), %r14d
	movl	$17, %edx
L23:
	cmpl	%eax, -52(%rbp)
	je	L24
L22:
	movl	-52(%rbp), %r8d
	movl	%eax, %ecx
	subl	%eax, %r8d
	movl	%r8d, %edi
	shrl	$3, %edi
	leal	0(,%rdi,8), %eax
	testl	%eax, %eax
	je	L25
	movq	-72(%rbp), %rbx
	vpxor	%xmm0, %xmm0, %xmm0
	leaq	(%rbx,%rcx,4), %rsi
	xorl	%ecx, %ecx
L31:
	addl	$1, %ecx
	vpaddd	(%rsi), %ymm0, %ymm0
	addq	$32, %rsi
	cmpl	%edi, %ecx
	jb	L31
	vmovdqa	%xmm0, %xmm1
	vextracti128	$0x1, %ymm0, %xmm0
	addl	%eax, %edx
	vmovd	%xmm1, %ecx
	vpextrd	$1, %xmm1, %esi
	addl	%esi, %ecx
	vpextrd	$2, %xmm1, %esi
	addl	%esi, %ecx
	vpextrd	$3, %xmm1, %esi
	addl	%esi, %ecx
	vmovd	%xmm0, %esi
	addl	%esi, %ecx
	vpextrd	$1, %xmm0, %esi
	addl	%esi, %ecx
	vpextrd	$2, %xmm0, %esi
	addl	%esi, %ecx
	vpextrd	$3, %xmm0, %esi
	addl	%esi, %ecx
	addl	%ecx, %r14d
	cmpl	%eax, %r8d
	je	L24
	.align 4,0x90
L25:
	movq	-72(%rbp), %rbx
	movslq	%edx, %rax
	movl	-52(%rbp), %ecx
	addl	(%rbx,%rax,4), %r14d
	leal	1(%rdx), %eax
	cmpl	%eax, %ecx
	je	L24
	cltq
	addl	(%rbx,%rax,4), %r14d
	leal	2(%rdx), %eax
	cmpl	%eax, %ecx
	je	L24
	cltq
	addl	(%rbx,%rax,4), %r14d
	leal	3(%rdx), %eax
	cmpl	%eax, %ecx
	je	L24
	cltq
	addl	(%rbx,%rax,4), %r14d
	leal	4(%rdx), %eax
	cmpl	%eax, %ecx
	je	L24
	cltq
	addl	(%rbx,%rax,4), %r14d
	leal	5(%rdx), %eax
	cmpl	%eax, %ecx
	je	L24
	cltq
	addl	$6, %edx
	addl	(%rbx,%rax,4), %r14d
	cmpl	%edx, %ecx
	je	L24
	movq	-72(%rbp), %rax
	movslq	%edx, %rdx
	addl	(%rax,%rdx,4), %r14d
L24:
	movl	-52(%rbp), %eax
	movq	%r15, %r11
	xorl	%r12d, %r12d
	movq	-80(%rbp), %rbx
	xorl	%ecx, %ecx
	movl	%r14d, -96(%rbp)
	movq	-72(%rbp), %r15
	vmovss	LC0(%rip), %xmm4
	addl	%r14d, %eax
	vmovss	LC1(%rip), %xmm3
	cltq
	vmovdqa	LC8(%rip), %ymm5
	leaq	18(,%rax,4), %rax
	vmovss	LC2(%rip), %xmm11
	andq	$-16, %rax
	vmovss	LC3(%rip), %xmm10
	subq	%rax, %rsp
	vmovss	LC4(%rip), %xmm9
	vmovss	LC5(%rip), %xmm8
	movq	%rsp, %r13
	movq	%rsp, -104(%rbp)
	vmovss	LC6(%rip), %xmm7
	vmovss	LC7(%rip), %xmm6
	.align 4,0x90
L44:
	vmulss	(%rbx,%r12,4), %xmm4, %xmm0
	movq	-64(%rbp), %rax
	movl	(%r15,%r12,4), %esi
	vdivss	%xmm0, %xmm3, %xmm0
	vcvtsi2ss	(%rax,%r12,4), %xmm1, %xmm1
	testl	%esi, %esi
	vsubss	(%r11,%r12,4), %xmm1, %xmm1
	vmulss	%xmm0, %xmm1, %xmm1
	js	L32
	leal	1(%rsi), %r9d
	movslq	%ecx, %rdi
	leaq	0(%r13,%rdi,4), %rax
	movl	%r9d, %r8d
	andl	$31, %eax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	cmpl	%r9d, %eax
	cmova	%r9d, %eax
	cmpl	$9, %r9d
	ja	L210
L98:
	leal	1(%rcx), %r14d
	cmpl	$1, %r8d
	vmovss	%xmm1, 0(%r13,%rdi,4)
	movl	%r14d, %eax
	je	L121
	vaddss	%xmm1, %xmm0, %xmm2
	movslq	%r14d, %rax
	cmpl	$2, %r8d
	leal	2(%rcx), %edx
	vmovss	%xmm2, 0(%r13,%rax,4)
	je	L122
	vaddss	%xmm0, %xmm0, %xmm2
	movslq	%edx, %rdx
	cmpl	$3, %r8d
	leal	3(%rcx), %eax
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rdx,4)
	je	L123
	vmulss	%xmm11, %xmm0, %xmm2
	cltq
	cmpl	$4, %r8d
	leal	4(%rcx), %edx
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rax,4)
	je	L124
	vmulss	%xmm10, %xmm0, %xmm2
	movslq	%edx, %rdx
	cmpl	$5, %r8d
	leal	5(%rcx), %eax
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rdx,4)
	je	L125
	vmulss	%xmm9, %xmm0, %xmm2
	cltq
	cmpl	$6, %r8d
	leal	6(%rcx), %edx
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rax,4)
	je	L126
	vmulss	%xmm8, %xmm0, %xmm2
	movslq	%edx, %rdx
	cmpl	$7, %r8d
	leal	7(%rcx), %eax
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rdx,4)
	je	L127
	vmulss	%xmm7, %xmm0, %xmm2
	cltq
	cmpl	$9, %r8d
	leal	8(%rcx), %edx
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rax,4)
	jne	L128
	vmulss	%xmm6, %xmm0, %xmm2
	movslq	%edx, %rdx
	leal	9(%rcx), %eax
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rdx,4)
	movl	$9, %edx
L35:
	cmpl	%r8d, %r9d
	je	L36
L34:
	subl	%r8d, %r9d
	movl	%r8d, %ecx
	movl	%r9d, %r8d
	movq	%rcx, -80(%rbp)
	shrl	$3, %r8d
	movl	%r8d, %ecx
	movl	%r8d, -56(%rbp)
	sall	$3, %ecx
	testl	%ecx, %ecx
	je	L37
	leal	2(%rdx), %r8d
	addq	-80(%rbp), %rdi
	vmovd	%edx, %xmm15
	vmovd	%r8d, %xmm12
	leal	1(%rdx), %r10d
	leal	3(%rdx), %r8d
	movl	%r10d, -84(%rbp)
	movl	%r8d, -88(%rbp)
	leal	6(%rdx), %r10d
	vpinsrd	$1, -84(%rbp), %xmm15, %xmm14
	leal	4(%rdx), %r8d
	vmovd	%r10d, %xmm13
	vpinsrd	$1, -88(%rbp), %xmm12, %xmm12
	leal	7(%rdx), %r10d
	vmovd	%r8d, %xmm2
	vpunpcklqdq	%xmm12, %xmm14, %xmm12
	leal	5(%rdx), %r8d
	vpinsrd	$1, %r10d, %xmm13, %xmm13
	vbroadcastss	%xmm0, %ymm14
	vpinsrd	$1, %r8d, %xmm2, %xmm2
	leaq	0(%r13,%rdi,4), %r10
	vpunpcklqdq	%xmm13, %xmm2, %xmm2
	xorl	%edi, %edi
	vbroadcastss	%xmm1, %ymm13
	vinserti128	$0x1, %xmm2, %ymm12, %ymm2
	movl	-56(%rbp), %r8d
	jmp	L42
	.align 4,0x90
L38:
	vmovdqa	%ymm12, %ymm2
L42:
	vpaddd	%ymm5, %ymm2, %ymm12
	vcvtdq2ps	%ymm2, %ymm2
	vmulps	%ymm14, %ymm2, %ymm2
	vaddps	%ymm13, %ymm2, %ymm2
	addl	$1, %edi
	addq	$32, %r10
	vmovaps	%ymm2, -32(%r10)
	cmpl	%r8d, %edi
	jb	L38
	addl	%ecx, %eax
	addl	%ecx, %edx
	cmpl	%ecx, %r9d
	je	L36
	.align 4,0x90
L37:
	vcvtsi2ss	%edx, %xmm2, %xmm2
	leal	1(%rax), %r8d
	movslq	%eax, %rcx
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rcx,4)
	leal	1(%rdx), %ecx
	cmpl	%ecx, %esi
	jl	L36
	vcvtsi2ss	%ecx, %xmm2, %xmm2
	leal	2(%rdx), %ecx
	movslq	%r8d, %r8
	leal	2(%rax), %edi
	cmpl	%ecx, %esi
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%r8,4)
	jl	L36
	vcvtsi2ss	%ecx, %xmm2, %xmm2
	leal	3(%rdx), %ecx
	movslq	%edi, %rdi
	leal	3(%rax), %r8d
	cmpl	%ecx, %esi
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rdi,4)
	jl	L36
	vcvtsi2ss	%ecx, %xmm2, %xmm2
	leal	4(%rdx), %ecx
	movslq	%r8d, %r8
	leal	4(%rax), %edi
	cmpl	%ecx, %esi
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%r8,4)
	jl	L36
	vcvtsi2ss	%ecx, %xmm2, %xmm2
	leal	5(%rdx), %ecx
	movslq	%edi, %rdi
	leal	5(%rax), %r8d
	cmpl	%ecx, %esi
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%rdi,4)
	jl	L36
	vcvtsi2ss	%ecx, %xmm2, %xmm2
	addl	$6, %edx
	addl	$6, %eax
	movslq	%r8d, %r8
	cmpl	%edx, %esi
	vmulss	%xmm0, %xmm2, %xmm2
	vaddss	%xmm1, %xmm2, %xmm2
	vmovss	%xmm2, 0(%r13,%r8,4)
	jl	L36
	vcvtsi2ss	%edx, %xmm2, %xmm2
	cltq
	vmulss	%xmm0, %xmm2, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 0(%r13,%rax,4)
L36:
	leal	(%r14,%rsi), %ecx
L32:
	addq	$1, %r12
	cmpl	%r12d, -52(%rbp)
	jne	L44
	movl	-96(%rbp), %r14d
	movq	-104(%rbp), %rax
	testl	%r14d, %r14d
	js	L59
L45:
	leal	1(%r14), %ecx
	movq	%rax, %rdx
	andl	$31, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$7, %edx
	cmpl	%ecx, %edx
	cmova	%ecx, %edx
	cmpl	$8, %ecx
	movl	%edx, %esi
	movl	%ecx, %edx
	ja	L211
L100:
	vmovss	(%rax), %xmm0
	cmpl	$1, %edx
	vmulss	%xmm0, %xmm0, %xmm1
	vmovss	LC1(%rip), %xmm0
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rax)
	je	L138
	vmovss	4(%rax), %xmm1
	cmpl	$2, %edx
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 4(%rax)
	je	L139
	vmovss	8(%rax), %xmm1
	cmpl	$3, %edx
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 8(%rax)
	je	L140
	vmovss	12(%rax), %xmm1
	cmpl	$4, %edx
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 12(%rax)
	je	L141
	vmovss	16(%rax), %xmm1
	cmpl	$5, %edx
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 16(%rax)
	je	L142
	vmovss	20(%rax), %xmm1
	cmpl	$6, %edx
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 20(%rax)
	je	L143
	vmovss	24(%rax), %xmm1
	cmpl	$8, %edx
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 24(%rax)
	jne	L144
	vmovss	28(%rax), %xmm1
	movl	$8, %esi
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, 28(%rax)
L51:
	cmpl	%edx, %ecx
	je	L52
L50:
	subl	%edx, %ecx
	movl	%edx, %r8d
	movl	%ecx, %r9d
	shrl	$3, %r9d
	leal	0(,%r9,8), %edi
	testl	%edi, %edi
	je	L53
	vmovaps	LC9(%rip), %ymm6
	leaq	(%rax,%r8,4), %r10
	xorl	%edx, %edx
	vmovaps	LC10(%rip), %ymm5
	xorl	%r8d, %r8d
	vxorps	%xmm3, %xmm3, %xmm3
	vmovaps	LC11(%rip), %ymm4
L58:
	vmovaps	(%r10,%rdx), %ymm1
	addl	$1, %r8d
	vmulps	%ymm1, %ymm1, %ymm1
	vaddps	%ymm6, %ymm1, %ymm1
	vcmpneqps	%ymm1, %ymm3, %ymm2
	vrsqrtps	%ymm1, %ymm0
	vandps	%ymm2, %ymm0, %ymm0
	vmulps	%ymm1, %ymm0, %ymm1
	vmulps	%ymm0, %ymm1, %ymm0
	vaddps	%ymm5, %ymm0, %ymm0
	vmulps	%ymm4, %ymm1, %ymm1
	vmulps	%ymm1, %ymm0, %ymm0
	vmovaps	%ymm0, (%r10,%rdx)
	addq	$32, %rdx
	cmpl	%r9d, %r8d
	jb	L58
	addl	%edi, %esi
	cmpl	%edi, %ecx
	je	L52
	.align 4,0x90
L53:
	movslq	%esi, %rdx
	leaq	(%rax,%rdx,4), %rdx
	vmovss	(%rdx), %xmm0
	vmulss	%xmm0, %xmm0, %xmm1
	vmovss	LC1(%rip), %xmm0
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rdx)
	leal	1(%rsi), %edx
	cmpl	%edx, %r14d
	jl	L52
	movslq	%edx, %rdx
	leaq	(%rax,%rdx,4), %rdx
	vmovss	(%rdx), %xmm1
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rdx)
	leal	2(%rsi), %edx
	cmpl	%edx, %r14d
	jl	L52
	movslq	%edx, %rdx
	leaq	(%rax,%rdx,4), %rdx
	vmovss	(%rdx), %xmm1
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rdx)
	leal	3(%rsi), %edx
	cmpl	%edx, %r14d
	jl	L52
	movslq	%edx, %rdx
	leaq	(%rax,%rdx,4), %rdx
	vmovss	(%rdx), %xmm1
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rdx)
	leal	4(%rsi), %edx
	cmpl	%edx, %r14d
	jl	L52
	movslq	%edx, %rdx
	leaq	(%rax,%rdx,4), %rdx
	vmovss	(%rdx), %xmm1
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rdx)
	leal	5(%rsi), %edx
	cmpl	%edx, %r14d
	jl	L52
	movslq	%edx, %rdx
	addl	$6, %esi
	leaq	(%rax,%rdx,4), %rdx
	cmpl	%esi, %r14d
	vmovss	(%rdx), %xmm1
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, (%rdx)
	jl	L52
	movslq	%esi, %rsi
	leaq	(%rax,%rsi,4), %rdx
	vmovss	(%rdx), %xmm1
	vmulss	%xmm1, %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, (%rdx)
L52:
	testl	%r14d, %r14d
	je	L60
L59:
	movq	%rax, %rdx
	andl	$31, %edx
	shrq	$2, %rdx
	negq	%rdx
	andl	$7, %edx
	cmpl	%r14d, %edx
	movl	%edx, %ecx
	movl	%r14d, %edx
	cmova	%r14d, %ecx
	cmpl	$8, %r14d
	ja	L212
L99:
	vmovss	(%rax), %xmm1
	cmpl	$1, %edx
	vmovss	4(%rax), %xmm0
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, (%rax)
	je	L130
	vmovss	8(%rax), %xmm1
	cmpl	$2, %edx
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 4(%rax)
	je	L131
	vmovss	12(%rax), %xmm0
	cmpl	$3, %edx
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, 8(%rax)
	je	L132
	vmovss	16(%rax), %xmm1
	cmpl	$4, %edx
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 12(%rax)
	je	L133
	vmovss	20(%rax), %xmm0
	cmpl	$5, %edx
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, 16(%rax)
	je	L134
	vmovss	24(%rax), %xmm1
	cmpl	$6, %edx
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, 20(%rax)
	je	L135
	vmovss	28(%rax), %xmm0
	cmpl	$8, %edx
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, 24(%rax)
	jne	L136
	vsubss	32(%rax), %xmm0, %xmm0
	movl	$8, %ecx
	vmovss	%xmm0, 28(%rax)
L48:
	cmpl	%r14d, %edx
	je	L60
L47:
	movl	%r14d, %r9d
	movl	%edx, %edi
	subl	%edx, %r9d
	movl	%r9d, %r8d
	shrl	$3, %r8d
	leal	0(,%r8,8), %esi
	testl	%esi, %esi
	je	L62
	leaq	0(,%rdi,4), %rdx
	xorl	%edi, %edi
	leaq	(%rax,%rdx), %r11
	leaq	4(%rax,%rdx), %r10
	xorl	%edx, %edx
L67:
	vmovups	(%r10,%rdx), %xmm0
	addl	$1, %edi
	vmovaps	(%r11,%rdx), %ymm1
	vinsertf128	$0x1, 16(%r10,%rdx), %ymm0, %ymm0
	vsubps	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, (%r11,%rdx)
	addq	$32, %rdx
	cmpl	%r8d, %edi
	jb	L67
	addl	%esi, %ecx
	cmpl	%r9d, %esi
	je	L60
	.align 4,0x90
L62:
	movslq	%ecx, %rdx
	leaq	(%rax,%rdx,4), %rdi
	leal	1(%rcx), %edx
	vmovss	(%rdi), %xmm0
	movslq	%edx, %rsi
	cmpl	%r14d, %edx
	vmovss	(%rax,%rsi,4), %xmm1
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, (%rdi)
	je	L60
	leal	2(%rcx), %edx
	movslq	%edx, %rdi
	cmpl	%r14d, %edx
	vmovss	(%rax,%rdi,4), %xmm0
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, (%rax,%rsi,4)
	je	L60
	leal	3(%rcx), %edx
	movslq	%edx, %rsi
	cmpl	%r14d, %edx
	vmovss	(%rax,%rsi,4), %xmm1
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, (%rax,%rdi,4)
	je	L60
	leal	4(%rcx), %edx
	movslq	%edx, %rdi
	cmpl	%r14d, %edx
	vmovss	(%rax,%rdi,4), %xmm0
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, (%rax,%rsi,4)
	je	L60
	leal	5(%rcx), %edx
	movslq	%edx, %rsi
	cmpl	%r14d, %edx
	vmovss	(%rax,%rsi,4), %xmm1
	vsubss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, (%rax,%rdi,4)
	je	L60
	leal	6(%rcx), %edx
	movslq	%edx, %rdi
	cmpl	%r14d, %edx
	vmovss	(%rax,%rdi,4), %xmm0
	vsubss	%xmm0, %xmm1, %xmm1
	vmovss	%xmm1, (%rax,%rsi,4)
	je	L60
	leal	7(%rcx), %edx
	movslq	%edx, %rdx
	vsubss	(%rax,%rdx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rax,%rdi,4)
L60:
	movslq	-92(%rbp), %rdx
	leaq	18(,%rdx,4), %rdx
	andq	$-16, %rdx
	subq	%rdx, %rsp
	movl	-52(%rbp), %edx
	movq	%rsp, %r13
	testl	%edx, %edx
	je	L68
	xorl	%r9d, %r9d
	xorl	%edi, %edi
	.align 4,0x90
L69:
	movq	-72(%rbp), %rbx
	movl	(%rbx,%r9,4), %edx
	testl	%edx, %edx
	je	L74
	movq	-64(%rbp), %rbx
	movl	%edx, %r11d
	movl	(%rbx,%r9,4), %ecx
	movslq	%ecx, %r10
	leaq	0(%r13,%r10,4), %r8
	movq	%r8, %rsi
	andl	$31, %esi
	shrq	$2, %rsi
	negq	%rsi
	andl	$7, %esi
	cmpl	%edx, %esi
	cmova	%edx, %esi
	cmpl	$8, %edx
	ja	L213
L101:
	vmovss	(%r8), %xmm0
	movslq	%edi, %rbx
	cmpl	$1, %r11d
	leal	1(%rdi), %esi
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r8)
	je	L146
	leal	1(%rcx), %r8d
	movslq	%esi, %rsi
	cmpl	$2, %r11d
	leal	2(%rdi), %ebx
	movslq	%r8d, %r8
	leaq	0(%r13,%r8,4), %r8
	vmovss	(%r8), %xmm0
	vaddss	(%rax,%rsi,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r8)
	je	L147
	leal	2(%rcx), %r8d
	movslq	%ebx, %rbx
	cmpl	$3, %r11d
	leal	3(%rdi), %esi
	movslq	%r8d, %r8
	leaq	0(%r13,%r8,4), %r8
	vmovss	(%r8), %xmm0
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r8)
	je	L148
	leal	3(%rcx), %r8d
	movslq	%esi, %rsi
	cmpl	$4, %r11d
	leal	4(%rdi), %ebx
	movslq	%r8d, %r8
	leaq	0(%r13,%r8,4), %r8
	vmovss	(%r8), %xmm0
	vaddss	(%rax,%rsi,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r8)
	je	L149
	leal	4(%rcx), %r8d
	movslq	%ebx, %rbx
	cmpl	$5, %r11d
	leal	5(%rdi), %esi
	movslq	%r8d, %r8
	leaq	0(%r13,%r8,4), %r8
	vmovss	(%r8), %xmm0
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r8)
	je	L150
	leal	5(%rcx), %ebx
	movslq	%esi, %rsi
	cmpl	$6, %r11d
	leal	6(%rdi), %r8d
	movslq	%ebx, %rbx
	leaq	0(%r13,%rbx,4), %rbx
	vmovss	(%rbx), %xmm0
	vaddss	(%rax,%rsi,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rbx)
	je	L151
	leal	6(%rcx), %esi
	movslq	%r8d, %r8
	cmpl	$8, %r11d
	leal	7(%rdi), %ebx
	movslq	%esi, %rsi
	leaq	0(%r13,%rsi,4), %rsi
	vmovss	(%rsi), %xmm0
	vaddss	(%rax,%r8,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi)
	jne	L152
	leal	7(%rcx), %r8d
	movslq	%ebx, %rbx
	leal	8(%rdi), %esi
	movslq	%r8d, %r8
	leaq	0(%r13,%r8,4), %r8
	vmovss	(%r8), %xmm0
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r8)
	movl	$8, %r8d
L77:
	cmpl	%edx, %r11d
	je	L78
L76:
	movl	%edx, %r14d
	movl	%r11d, %ebx
	subl	%r11d, %r14d
	movl	%r14d, %r12d
	movl	%r14d, -80(%rbp)
	shrl	$3, %r12d
	leal	0(,%r12,8), %r11d
	testl	%r11d, %r11d
	je	L79
	movslq	%edi, %r14
	addq	%rbx, %r10
	addq	%rbx, %r14
	xorl	%ebx, %ebx
	leaq	(%rax,%r14,4), %r15
	leaq	0(%r13,%r10,4), %r14
	xorl	%r10d, %r10d
L80:
	vmovups	(%r15,%r10), %xmm0
	addl	$1, %ebx
	vinsertf128	$0x1, 16(%r15,%r10), %ymm0, %ymm0
	vaddps	(%r14,%r10), %ymm0, %ymm0
	vmovaps	%ymm0, (%r14,%r10)
	addq	$32, %r10
	cmpl	%r12d, %ebx
	jb	L80
	addl	%r11d, %esi
	addl	%r11d, %r8d
	cmpl	-80(%rbp), %r11d
	je	L78
	.align 4,0x90
L79:
	leal	(%r8,%rcx), %r10d
	movslq	%esi, %rbx
	leal	1(%rsi), %r11d
	movslq	%r10d, %r10
	leaq	0(%r13,%r10,4), %r10
	vmovss	(%r10), %xmm0
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	leal	1(%r8), %r10d
	cmpl	%edx, %r10d
	je	L78
	leal	2(%rsi), %ebx
	addl	%ecx, %r10d
	movslq	%r11d, %r11
	movslq	%r10d, %r10
	leaq	0(%r13,%r10,4), %r10
	vmovss	(%r10), %xmm0
	vaddss	(%rax,%r11,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	leal	2(%r8), %r10d
	cmpl	%edx, %r10d
	je	L78
	leal	3(%rsi), %r11d
	addl	%ecx, %r10d
	movslq	%ebx, %rbx
	movslq	%r10d, %r10
	leaq	0(%r13,%r10,4), %r10
	vmovss	(%r10), %xmm0
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	leal	3(%r8), %r10d
	cmpl	%edx, %r10d
	je	L78
	leal	4(%rsi), %ebx
	addl	%ecx, %r10d
	movslq	%r11d, %r11
	movslq	%r10d, %r10
	leaq	0(%r13,%r10,4), %r10
	vmovss	(%r10), %xmm0
	vaddss	(%rax,%r11,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	leal	4(%r8), %r10d
	cmpl	%edx, %r10d
	je	L78
	leal	5(%rsi), %r11d
	addl	%ecx, %r10d
	movslq	%ebx, %rbx
	movslq	%r10d, %r10
	leaq	0(%r13,%r10,4), %r10
	vmovss	(%r10), %xmm0
	vaddss	(%rax,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	leal	5(%r8), %r10d
	cmpl	%edx, %r10d
	je	L78
	addl	%ecx, %r10d
	movslq	%r11d, %r11
	addl	$6, %r8d
	movslq	%r10d, %r10
	addl	$6, %esi
	cmpl	%edx, %r8d
	leaq	0(%r13,%r10,4), %r10
	vmovss	(%r10), %xmm0
	vaddss	(%rax,%r11,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r10)
	je	L78
	addl	%r8d, %ecx
	movslq	%esi, %rsi
	movslq	%ecx, %rcx
	leaq	0(%r13,%rcx,4), %rcx
	vmovss	(%rcx), %xmm0
	vaddss	(%rax,%rsi,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rcx)
L78:
	addl	%edx, %edi
L74:
	addq	$1, %r9
	cmpl	%r9d, -52(%rbp)
	jne	L69
L68:
	movq	_coupling(%rip), %rcx
	xorl	%r12d, %r12d
	movq	8+_coupling(%rip), %r14
	movl	-92(%rbp), %eax
	movq	_localAmplitudes(%rip), %r15
	subq	%rcx, %r14
	sarq	$2, %r14
	testl	%eax, %eax
	je	L207
	movl	$1, -52(%rbp)
	vmovdqa	LC8(%rip), %ymm5
	subl	%r14d, -52(%rbp)
	movq	%r13, -80(%rbp)
	movl	%eax, %r13d
	vmovaps	LC12(%rip), %ymm4
	.align 4,0x90
L181:
	movl	-52(%rbp), %eax
	leal	(%r14,%r12), %esi
	movl	$0, %ebx
	movl	%r12d, %edx
	addl	%r12d, %eax
	cmovs	%ebx, %eax
	cmpl	%esi, %r13d
	cmovle	%r13d, %esi
	cmpl	%esi, %eax
	jge	L94
	movq	-80(%rbp), %rbx
	movl	%esi, %r9d
	subl	%eax, %r9d
	vmovss	(%rbx,%r12,4), %xmm0
	movslq	%eax, %rbx
	leaq	(%r15,%rbx,4), %r8
	movq	%r8, %rdi
	andl	$31, %edi
	shrq	$2, %rdi
	negq	%rdi
	andl	$7, %edi
	cmpl	%r9d, %edi
	cmova	%r9d, %edi
	cmpl	$8, %r9d
	cmovbe	%r9d, %edi
	testl	%edi, %edi
	je	L86
	movl	%eax, %r10d
	subl	%r12d, %r10d
	movl	%r10d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r10d
	subl	%r11d, %r10d
	cmpl	$1, %edi
	movslq	%r10d, %r10
	vmulss	(%rcx,%r10,4), %xmm0, %xmm1
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	leal	1(%rax), %r8d
	je	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	cmpl	$2, %edi
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	leal	2(%rax), %r8d
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
	je	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	cmpl	$3, %edi
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	leal	3(%rax), %r8d
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
	je	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	cmpl	$4, %edi
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	leal	4(%rax), %r8d
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
	je	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	cmpl	$5, %edi
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	leal	5(%rax), %r8d
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
	je	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	cmpl	$6, %edi
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	leal	6(%rax), %r8d
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
	je	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	cmpl	$8, %edi
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	leal	7(%rax), %r8d
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
	jne	L160
	movslq	%r8d, %r10
	subl	%r12d, %r8d
	addl	$8, %eax
	leaq	(%r15,%r10,4), %r10
	movl	%r8d, %r11d
	sarl	$31, %r11d
	xorl	%r11d, %r8d
	subl	%r11d, %r8d
	movslq	%r8d, %r8
	vmulss	(%rcx,%r8,4), %xmm0, %xmm1
	vaddss	(%r10), %xmm1, %xmm1
	vmovss	%xmm1, (%r10)
L87:
	cmpl	%edi, %r9d
	je	L94
L86:
	subl	%edi, %r9d
	movl	%edi, %r10d
	movl	%r9d, %r8d
	movq	%r10, -72(%rbp)
	shrl	$3, %r8d
	movl	%r8d, %edi
	movl	%r8d, -64(%rbp)
	sall	$3, %edi
	testl	%edi, %edi
	je	L89
	leal	4(%rax), %r10d
	vmovd	%eax, %xmm7
	leal	1(%rax), %r11d
	vmovd	%r10d, %xmm1
	leal	5(%rax), %r10d
	movl	%r11d, -56(%rbp)
	vpinsrd	$1, %r10d, %xmm1, %xmm1
	movq	-72(%rbp), %r10
	leal	2(%rax), %r11d
	leal	6(%rax), %r8d
	vpinsrd	$1, -56(%rbp), %xmm7, %xmm6
	vmovd	%r11d, %xmm2
	leal	3(%rax), %r11d
	vmovd	%r8d, %xmm3
	vmovd	%edx, %xmm7
	leal	7(%rax), %r8d
	vpinsrd	$1, %r11d, %xmm2, %xmm2
	vbroadcastss	%xmm7, %ymm7
	addq	%rbx, %r10
	vpinsrd	$1, %r8d, %xmm3, %xmm3
	vpunpcklqdq	%xmm2, %xmm6, %xmm2
	vpunpcklqdq	%xmm3, %xmm1, %xmm1
	leaq	(%r15,%r10,4), %rbx
	vbroadcastss	%xmm0, %ymm6
	vinserti128	$0x1, %xmm1, %ymm2, %ymm1
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	movl	-64(%rbp), %r8d
	jmp	L95
	.align 4,0x90
L90:
	vmovdqa	%ymm3, %ymm1
L95:
	vpaddd	%ymm5, %ymm1, %ymm3
	vmovaps	%ymm4, %ymm15
	vpsubd	%ymm7, %ymm1, %ymm1
	vpabsd	%ymm1, %ymm2
	addl	$1, %r11d
	vgatherdps	%ymm15, (%rcx,%ymm2,4), %ymm1
	vmulps	%ymm6, %ymm1, %ymm1
	vaddps	(%rbx,%r10), %ymm1, %ymm1
	vmovaps	%ymm1, (%rbx,%r10)
	addq	$32, %r10
	cmpl	%r8d, %r11d
	jb	L90
	addl	%edi, %eax
	cmpl	%r9d, %edi
	je	L94
	.align 4,0x90
L89:
	movslq	%eax, %rdi
	leaq	(%r15,%rdi,4), %r8
	movl	%eax, %edi
	subl	%edx, %edi
	movl	%edi, %r9d
	sarl	$31, %r9d
	xorl	%r9d, %edi
	subl	%r9d, %edi
	movslq	%edi, %rdi
	vmulss	(%rcx,%rdi,4), %xmm0, %xmm1
	leal	1(%rax), %edi
	cmpl	%edi, %esi
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	jle	L94
	movslq	%edi, %r8
	subl	%edx, %edi
	leaq	(%r15,%r8,4), %r8
	movl	%edi, %r9d
	sarl	$31, %r9d
	xorl	%r9d, %edi
	subl	%r9d, %edi
	movslq	%edi, %rdi
	vmulss	(%rcx,%rdi,4), %xmm0, %xmm1
	leal	2(%rax), %edi
	cmpl	%edi, %esi
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	jle	L94
	movslq	%edi, %r8
	subl	%edx, %edi
	leaq	(%r15,%r8,4), %r8
	movl	%edi, %r9d
	sarl	$31, %r9d
	xorl	%r9d, %edi
	subl	%r9d, %edi
	movslq	%edi, %rdi
	vmulss	(%rcx,%rdi,4), %xmm0, %xmm1
	leal	3(%rax), %edi
	cmpl	%esi, %edi
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	jge	L94
	movslq	%edi, %r8
	subl	%edx, %edi
	leaq	(%r15,%r8,4), %r8
	movl	%edi, %r9d
	sarl	$31, %r9d
	xorl	%r9d, %edi
	subl	%r9d, %edi
	movslq	%edi, %rdi
	vmulss	(%rcx,%rdi,4), %xmm0, %xmm1
	leal	4(%rax), %edi
	cmpl	%esi, %edi
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	jge	L94
	movslq	%edi, %r8
	subl	%edx, %edi
	leaq	(%r15,%r8,4), %r8
	movl	%edi, %r9d
	sarl	$31, %r9d
	xorl	%r9d, %edi
	subl	%r9d, %edi
	movslq	%edi, %rdi
	vmulss	(%rcx,%rdi,4), %xmm0, %xmm1
	leal	5(%rax), %edi
	cmpl	%edi, %esi
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	jle	L94
	movslq	%edi, %r8
	subl	%edx, %edi
	addl	$6, %eax
	leaq	(%r15,%r8,4), %r8
	movl	%edi, %r9d
	sarl	$31, %r9d
	xorl	%r9d, %edi
	subl	%r9d, %edi
	cmpl	%esi, %eax
	movslq	%edi, %rdi
	vmulss	(%rcx,%rdi,4), %xmm0, %xmm1
	vaddss	(%r8), %xmm1, %xmm1
	vmovss	%xmm1, (%r8)
	jge	L94
	movslq	%eax, %rsi
	subl	%edx, %eax
	leaq	(%r15,%rsi,4), %rsi
	cltd
	xorl	%edx, %eax
	subl	%edx, %eax
	cltq
	vmulss	(%rcx,%rax,4), %xmm0, %xmm0
	vaddss	(%rsi), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi)
L94:
	addq	$1, %r12
	cmpl	%r12d, %r13d
	jne	L181
L207:
	vzeroupper
	leaq	-48(%rbp), %rsp
	popq	%rbx
	popq	%r10
LCFI4:
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI5:
	ret
	.align 4,0x90
L160:
LCFI6:
	movl	%r8d, %eax
	jmp	L87
	.align 4,0x90
L213:
	testl	%esi, %esi
	movl	%esi, %r11d
	jne	L101
	movl	%edi, %esi
	xorl	%r8d, %r8d
	jmp	L76
	.align 4,0x90
L210:
	testl	%eax, %eax
	movl	%eax, %r8d
	jne	L98
	leal	1(%rcx), %r14d
	movl	%ecx, %eax
	xorl	%edx, %edx
	jmp	L34
	.align 4,0x90
L152:
	movl	%ebx, %esi
	movl	$7, %r8d
	jmp	L77
	.align 4,0x90
L151:
	movl	%r8d, %esi
	movl	$6, %r8d
	jmp	L77
	.align 4,0x90
L150:
	movl	$5, %r8d
	jmp	L77
	.align 4,0x90
L149:
	movl	%ebx, %esi
	movl	$4, %r8d
	jmp	L77
	.align 4,0x90
L148:
	movl	$3, %r8d
	jmp	L77
	.align 4,0x90
L147:
	movl	%ebx, %esi
	movl	$2, %r8d
	jmp	L77
	.align 4,0x90
L146:
	movl	$1, %r8d
	jmp	L77
	.align 4,0x90
L128:
	movl	%edx, %eax
	movl	$8, %edx
	jmp	L35
	.align 4,0x90
L127:
	movl	$7, %edx
	jmp	L35
	.align 4,0x90
L122:
	movl	%edx, %eax
	movl	$2, %edx
	jmp	L35
	.align 4,0x90
L121:
	movl	$1, %edx
	jmp	L35
	.align 4,0x90
L126:
	movl	%edx, %eax
	movl	$6, %edx
	jmp	L35
	.align 4,0x90
L125:
	movl	$5, %edx
	jmp	L35
	.align 4,0x90
L124:
	movl	%edx, %eax
	movl	$4, %edx
	jmp	L35
	.align 4,0x90
L123:
	movl	$3, %edx
	jmp	L35
	.align 4,0x90
L20:
	testl	%eax, %eax
	jne	L21
	xorl	%edx, %edx
	xorl	%r14d, %r14d
	jmp	L22
	.align 4,0x90
L209:
	testl	%eax, %eax
	movl	%eax, %esi
	jne	L97
	xorl	%edx, %edx
	jmp	L10
	.align 4,0x90
L211:
	testl	%esi, %esi
	movl	%esi, %edx
	jne	L100
	xorl	%esi, %esi
	.p2align 4,,3
	jmp	L50
	.align 4,0x90
L212:
	testl	%ecx, %ecx
	movl	%ecx, %edx
	jne	L99
	xorl	%ecx, %ecx
	.p2align 4,,3
	jmp	L47
L6:
	movl	_Nstrips(%rip), %ebx
	subq	%rax, %rsp
	xorl	%r14d, %r14d
	movq	%rsp, %rax
	movl	%ebx, -92(%rbp)
	jmp	L45
L136:
	movl	$7, %ecx
	jmp	L48
L143:
	movl	$6, %esi
	jmp	L51
L135:
	movl	$6, %ecx
	jmp	L48
L130:
	movl	$1, %ecx
	jmp	L48
L140:
	movl	$3, %esi
	jmp	L51
L141:
	movl	$4, %esi
	jmp	L51
L142:
	movl	$5, %esi
	jmp	L51
L131:
	movl	$2, %ecx
	jmp	L48
L132:
	movl	$3, %ecx
	jmp	L48
L133:
	movl	$4, %ecx
	jmp	L48
L134:
	movl	$5, %ecx
	jmp	L48
L138:
	movl	$1, %esi
	jmp	L51
L139:
	movl	$2, %esi
	jmp	L51
L144:
	movl	$7, %esi
	jmp	L51
L111:
	movl	$8, %edx
	jmp	L23
L112:
	movl	$9, %edx
	jmp	L23
L113:
	movl	$10, %edx
	jmp	L23
L114:
	movl	$11, %edx
	jmp	L23
L109:
	movl	$6, %edx
	jmp	L23
L110:
	movl	$7, %edx
	jmp	L23
L108:
	movl	$5, %edx
	jmp	L23
L119:
	movl	$16, %edx
	jmp	L23
L115:
	movl	$12, %edx
	jmp	L23
L104:
	movl	$1, %edx
	jmp	L23
L105:
	movl	$2, %edx
	jmp	L23
L106:
	movl	$3, %edx
	jmp	L23
L107:
	movl	$4, %edx
	jmp	L23
L116:
	movl	$13, %edx
	jmp	L23
L117:
	movl	$14, %edx
	jmp	L23
L118:
	movl	$15, %edx
	jmp	L23
LFE3763:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
__GLOBAL__sub_I_chargeDistr.cc:
LFB4052:
	pushq	%rbx
LCFI7:
	movq	__ZNSt6vectorIfSaIfEED1Ev@GOTPCREL(%rip), %rbx
	leaq	___dso_handle(%rip), %rdx
	leaq	_localAmplitudes(%rip), %rsi
	movq	$0, _localAmplitudes(%rip)
	movq	$0, 8+_localAmplitudes(%rip)
	movq	$0, 16+_localAmplitudes(%rip)
	movq	%rbx, %rdi
	call	___cxa_atexit
	leaq	___dso_handle(%rip), %rdx
	movq	%rbx, %rdi
	popq	%rbx
LCFI8:
	leaq	_coupling(%rip), %rsi
	movq	$0, _coupling(%rip)
	movq	$0, 8+_coupling(%rip)
	movq	$0, 16+_coupling(%rip)
	jmp	___cxa_atexit
LFE4052:
	.globl _coupling
	.zerofill __DATA,__pu_bss5,_coupling,24,5
	.globl _localAmplitudes
	.zerofill __DATA,__pu_bss5,_localAmplitudes,24,5
	.globl _Nstrips
	.zerofill __DATA,__pu_bss2,_Nstrips,4,2
	.globl _Nsigma
	.zerofill __DATA,__pu_bss2,_Nsigma,4,2
	.literal4
	.align 2
LC0:
	.long	1068827891
	.align 2
LC1:
	.long	1065353216
	.align 2
LC2:
	.long	1077936128
	.align 2
LC3:
	.long	1082130432
	.align 2
LC4:
	.long	1084227584
	.align 2
LC5:
	.long	1086324736
	.align 2
LC6:
	.long	1088421888
	.align 2
LC7:
	.long	1090519040
	.const
	.align 5
LC8:
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.align 5
LC9:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 5
LC10:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 5
LC11:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.align 5
LC12:
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
	.long	4294967295
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
	.quad	LFB4047-.
	.set L$set$2,LFE4047-LFB4047
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB3763-.
	.set L$set$4,LFE3763-LFB3763
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB3763
	.long L$set$5
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI2-LCFI1
	.long L$set$7
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x58
	.byte	0x6
	.byte	0x10
	.byte	0xf
	.byte	0x2
	.byte	0x76
	.byte	0x78
	.byte	0x10
	.byte	0xe
	.byte	0x2
	.byte	0x76
	.byte	0x70
	.byte	0x10
	.byte	0xd
	.byte	0x2
	.byte	0x76
	.byte	0x68
	.byte	0x10
	.byte	0xc
	.byte	0x2
	.byte	0x76
	.byte	0x60
	.byte	0x4
	.set L$set$8,LCFI3-LCFI2
	.long L$set$8
	.byte	0x10
	.byte	0x3
	.byte	0x2
	.byte	0x76
	.byte	0x50
	.byte	0x4
	.set L$set$9,LCFI4-LCFI3
	.long L$set$9
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI5-LCFI4
	.long L$set$10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$11,LCFI6-LCFI5
	.long L$set$11
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$12,LEFDE5-LASFDE5
	.long L$set$12
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB4052-.
	.set L$set$13,LFE4052-LFB4052
	.quad L$set$13
	.byte	0
	.byte	0x4
	.set L$set$14,LCFI7-LFB4052
	.long L$set$14
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$15,LCFI8-LCFI7
	.long L$set$15
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE5:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_chargeDistr.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
