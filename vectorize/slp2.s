	.text
	.align 4,0x90
	.globl __Z3sumRK1AS1_
__Z3sumRK1AS1_:
LFB0:
	movapd	16(%rsi), %xmm2
	movq	%rdi, %rax
	movapd	32(%rsi), %xmm1
	movapd	48(%rsi), %xmm0
	movapd	(%rsi), %xmm3
	addpd	16(%rdx), %xmm2
	addpd	32(%rdx), %xmm1
	addpd	48(%rdx), %xmm0
	addpd	(%rdx), %xmm3
	movapd	%xmm2, 16(%rdi)
	movapd	%xmm1, 32(%rdi)
	movapd	%xmm0, 48(%rdi)
	movapd	%xmm3, (%rdi)
	ret
LFE0:
	.align 4,0x90
	.globl __Z4sumlRK1AS1_
__Z4sumlRK1AS1_:
LFB1:
	movapd	(%rsi), %xmm0
	movq	%rdi, %rax
	addpd	(%rdx), %xmm0
	movapd	%xmm0, (%rdi)
	movapd	16(%rsi), %xmm0
	addpd	16(%rdx), %xmm0
	movapd	%xmm0, 16(%rdi)
	movapd	32(%rsi), %xmm0
	addpd	32(%rdx), %xmm0
	movapd	%xmm0, 32(%rdi)
	movapd	48(%rsi), %xmm0
	addpd	48(%rdx), %xmm0
	movapd	%xmm0, 48(%rdi)
	ret
LFE1:
	.align 4,0x90
	.globl __Z5dosumv
__Z5dosumv:
LFB2:
	movapd	16+_a2(%rip), %xmm2
	movapd	32+_a2(%rip), %xmm1
	movapd	48+_a2(%rip), %xmm0
	movapd	_a2(%rip), %xmm3
	addpd	16+_a3(%rip), %xmm2
	addpd	32+_a3(%rip), %xmm1
	addpd	48+_a3(%rip), %xmm0
	addpd	_a3(%rip), %xmm3
	movapd	%xmm2, 16+_a1(%rip)
	movapd	%xmm1, 32+_a1(%rip)
	movapd	%xmm0, 48+_a1(%rip)
	movapd	%xmm3, _a1(%rip)
	ret
LFE2:
	.align 4,0x90
	.globl __Z6dosumlv
__Z6dosumlv:
LFB3:
	movapd	_a2(%rip), %xmm0
	addpd	_a3(%rip), %xmm0
	movapd	%xmm0, -88(%rsp)
	movapd	%xmm0, -72(%rsp)
	movapd	16+_a3(%rip), %xmm0
	movq	-88(%rsp), %rsi
	addpd	16+_a2(%rip), %xmm0
	movapd	%xmm0, -88(%rsp)
	movapd	%xmm0, -56(%rsp)
	movapd	32+_a3(%rip), %xmm0
	movq	-88(%rsp), %rcx
	movq	%rsi, _a1(%rip)
	addpd	32+_a2(%rip), %xmm0
	movq	-64(%rsp), %rsi
	movapd	%xmm0, -88(%rsp)
	movapd	%xmm0, -40(%rsp)
	movapd	48+_a2(%rip), %xmm0
	movq	-88(%rsp), %rdx
	movq	%rcx, 16+_a1(%rip)
	addpd	48+_a3(%rip), %xmm0
	movq	%rsi, 8+_a1(%rip)
	movq	-48(%rsp), %rcx
	movapd	%xmm0, -88(%rsp)
	movq	-88(%rsp), %rax
	movapd	%xmm0, -24(%rsp)
	movq	%rdx, 32+_a1(%rip)
	movq	-32(%rsp), %rdx
	movq	%rcx, 24+_a1(%rip)
	movq	%rax, 48+_a1(%rip)
	movq	-16(%rsp), %rax
	movq	%rdx, 40+_a1(%rip)
	movq	%rax, 56+_a1(%rip)
	ret
LFE3:
	.align 4,0x90
	.globl __Z4sum2RK1AS1_
__Z4sum2RK1AS1_:
LFB4:
	movq	(%rsi), %rcx
	movq	%rdi, %rax
	movsd	8(%rdx), %xmm0
	movq	%rcx, (%rdi)
	movq	8(%rsi), %rcx
	movsd	(%rdi), %xmm1
	addsd	(%rdx), %xmm1
	movq	%rcx, 8(%rdi)
	movq	16(%rsi), %rcx
	addsd	8(%rdi), %xmm0
	movsd	%xmm1, (%rdi)
	movq	%rcx, 16(%rdi)
	movq	24(%rsi), %rcx
	movsd	16(%rdi), %xmm1
	movsd	%xmm0, 8(%rdi)
	movsd	24(%rdx), %xmm0
	addsd	16(%rdx), %xmm1
	movq	%rcx, 24(%rdi)
	movq	32(%rsi), %rcx
	addsd	24(%rdi), %xmm0
	movsd	%xmm1, 16(%rdi)
	movq	%rcx, 32(%rdi)
	movq	40(%rsi), %rcx
	movsd	32(%rdi), %xmm1
	movsd	%xmm0, 24(%rdi)
	movsd	40(%rdx), %xmm0
	addsd	32(%rdx), %xmm1
	movq	%rcx, 40(%rdi)
	movq	48(%rsi), %rcx
	addsd	40(%rdi), %xmm0
	movsd	%xmm1, 32(%rdi)
	movq	%rcx, 48(%rdi)
	movq	56(%rsi), %rcx
	movq	%rcx, 56(%rdi)
	movsd	%xmm0, 40(%rdi)
	movsd	48(%rdi), %xmm1
	movsd	56(%rdx), %xmm0
	addsd	48(%rdx), %xmm1
	addsd	56(%rdi), %xmm0
	movsd	%xmm1, 48(%rdi)
	movsd	%xmm0, 56(%rdi)
	ret
LFE4:
	.align 4,0x90
	.globl __Z5suml21ARKS_
__Z5suml21ARKS_:
LFB5:
	movapd	8(%rsp), %xmm0
	movq	%rdi, %rax
	addpd	(%rsi), %xmm0
	movapd	%xmm0, -24(%rsp)
	movapd	%xmm0, 8(%rsp)
	movapd	24(%rsp), %xmm0
	movq	-24(%rsp), %r8
	addpd	16(%rsi), %xmm0
	movapd	%xmm0, -24(%rsp)
	movapd	%xmm0, 24(%rsp)
	movapd	40(%rsp), %xmm0
	movq	-24(%rsp), %rdi
	movq	%r8, (%rax)
	addpd	32(%rsi), %xmm0
	movapd	%xmm0, -24(%rsp)
	movapd	%xmm0, 40(%rsp)
	movapd	56(%rsp), %xmm0
	movq	-24(%rsp), %rcx
	movq	%rdi, 16(%rax)
	addpd	48(%rsi), %xmm0
	movq	16(%rsp), %rsi
	movapd	%xmm0, -24(%rsp)
	movq	-24(%rsp), %rdx
	movapd	%xmm0, 56(%rsp)
	movq	%rsi, 8(%rax)
	movq	32(%rsp), %rsi
	movq	%rcx, 32(%rax)
	movq	48(%rsp), %rcx
	movq	%rdx, 48(%rax)
	movq	64(%rsp), %rdx
	movq	%rsi, 24(%rax)
	movq	%rcx, 40(%rax)
	movq	%rdx, 56(%rax)
	ret
LFE5:
	.globl _a3
	.zerofill __DATA,__pu_bss5,_a3,64,5
	.globl _a2
	.zerofill __DATA,__pu_bss5,_a2,64,5
	.globl _a1
	.zerofill __DATA,__pu_bss5,_a1,64,5
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
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1-.
	.set L$set$4,LFE1-LFB1
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB2-.
	.set L$set$6,LFE2-LFB2
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3-.
	.set L$set$8,LFE3-LFB3
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4-.
	.set L$set$10,LFE4-LFB4
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB5-.
	.set L$set$12,LFE5-LFB5
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
