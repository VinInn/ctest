	.text
	.align 4,0x90
	.globl __Z10nestedCondPdS_S_S_ii
__Z10nestedCondPdS_S_S_ii:
LFB0:
	xorl	%r8d, %r8d
	movddup	(%rdx), %xmm5
	.align 4,0x90
L2:
	movsd	(%rdi,%r8), %xmm4
	movapd	%xmm5, %xmm1
	xorl	%eax, %eax
	movhpd	8(%rdi,%r8), %xmm4
	.align 4,0x90
L3:
	movddup	(%rcx,%rax), %xmm0
	cmpltpd	%xmm4, %xmm0
	movddup	8(%rdx,%rax), %xmm3
	addq	$8, %rax
	movapd	%xmm1, %xmm2
	cmpq	$2048, %rax
	movapd	%xmm0, %xmm1
	andpd	%xmm0, %xmm2
	andnpd	%xmm3, %xmm1
	orpd	%xmm2, %xmm1
	jne	L3
	movlpd	%xmm1, (%rsi,%r8)
	movhpd	%xmm1, 8(%rsi,%r8)
	addq	$16, %r8
	cmpq	$2048, %r8
	jne	L2
	ret
LFE0:
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
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
