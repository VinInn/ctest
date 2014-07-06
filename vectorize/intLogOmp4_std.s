	.text
	.align 4,0x90
	.globl __Z6intLogPfi
__Z6intLogPfi:
LFB238:
	testl	%esi, %esi
	jle	L7
	pushq	%rbx
LCFI0:
	xorl	%edx, %edx
	movl	$8388608, %r9d
	xorl	%ebx, %ebx
	.align 4,0x90
L6:
	movl	(%rdi,%rdx,4), %r8d
	addq	$1, %rdx
	movl	%r8d, %eax
	shrl	$23, %r8d
	andl	$8388607, %eax
	movzbl	%r8b, %r8d
	leal	-127(%rbx,%r8), %ebx
	orq	$8388608, %rax
	imulq	%r9, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	movl	%eax, %ecx
	movl	%eax, %r9d
	shrl	$31, %ecx
	shrl	%cl, %r9d
	addl	%ecx, %ebx
	cmpl	%edx, %esi
	jg	L6
	xorpd	%xmm0, %xmm0
	cvtsi2sdq	%r9, %xmm0
	call	_log2
	xorpd	%xmm1, %xmm1
	cvtsi2sd	%ebx, %xmm1
	popq	%rbx
LCFI1:
	addsd	LC1(%rip), %xmm0
	addsd	%xmm1, %xmm0
	unpcklpd	%xmm0, %xmm0
	cvtpd2ps	%xmm0, %xmm0
	ret
L7:
	xorps	%xmm0, %xmm0
	ret
LFE238:
	.literal8
	.align 3
LC1:
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
	.quad	LFB238-.
	.set L$set$2,LFE238-LFB238
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB238
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xc3
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
