	.text
	.align 4,0x90
	.globl __Z6intLogPfi
__Z6intLogPfi:
LFB239:
	pushq	%rbx
LCFI0:
	subq	$16, %rsp
LCFI1:
	testl	%esi, %esi
	movl	$0, (%rsp)
	movl	$0, 4(%rsp)
	jle	L2
	xorl	%ecx, %ecx
	xorl	%r9d, %r9d
	xorl	%edx, %edx
	jmp	L4
	.align 4,0x90
L9:
	movl	%eax, %ecx
L4:
	movl	(%rdi,%rdx,4), %r8d
	addq	$1, %rdx
	movl	%r8d, %eax
	shrl	$23, %r8d
	andl	$8388607, %eax
	movzbl	%r8b, %r8d
	leal	-127(%r9,%r8), %r9d
	orq	$8388608, %rax
	imulq	%rcx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	movl	%eax, %ecx
	shrl	$31, %ecx
	shrl	%cl, %eax
	addl	%ecx, %r9d
	cmpl	%edx, %esi
	jg	L9
	movl	%eax, 4(%rsp)
	movl	%r9d, (%rsp)
L2:
	movl	4(%rsp), %eax
	xorpd	%xmm0, %xmm0
	movl	%eax, %ebx
	shrl	$31, %ebx
	movl	%ebx, %ecx
	shrl	%cl, %eax
	cvtsi2sdq	%rax, %xmm0
	call	_log2
	addl	(%rsp), %ebx
	xorpd	%xmm1, %xmm1
	addsd	LC0(%rip), %xmm0
	addq	$16, %rsp
LCFI2:
	cvtsi2sd	%ebx, %xmm1
	popq	%rbx
LCFI3:
	addsd	%xmm1, %xmm0
	unpcklpd	%xmm0, %xmm0
	cvtpd2ps	%xmm0, %xmm0
	ret
LFE239:
	.literal8
	.align 3
LC0:
	.long	0
	.long	-1070137344
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
	.quad	LFB239-.
	.set L$set$2,LFE239-LFB239
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB239
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
