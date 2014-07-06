	.text
	.align 4,0x90
	.globl __Z3RKSPdS_d
__Z3RKSPdS_d:
LFB0:
	vmovsd	%xmm0, -8(%rsp)
	vmovsd	8+_H1(%rip), %xmm0
	leaq	280(%rdi), %rax
	vmovsd	24+_H0(%rip), %xmm6
	vmovsd	%xmm0, -40(%rsp)
	vmovsd	_H1(%rip), %xmm0
	vmovsd	%xmm6, -72(%rsp)
	vmovsd	8+_H0(%rip), %xmm6
	vmovsd	%xmm0, -32(%rsp)
	vmovsd	16+_H2(%rip), %xmm0
	vmovsd	%xmm6, -64(%rsp)
	vmovsd	_H0(%rip), %xmm6
	vmovsd	%xmm0, -80(%rsp)
	vmovsd	8+_H2(%rip), %xmm0
	vmovsd	%xmm6, -56(%rsp)
	vmovsd	16+_H0(%rip), %xmm6
	vmovsd	%xmm0, -24(%rsp)
	vmovsd	_H2(%rip), %xmm0
	vmovsd	%xmm6, -48(%rsp)
	vmovsd	16+_H1(%rip), %xmm6
	vmovsd	%xmm0, -16(%rsp)
L3:
	vmovsd	16(%rsi), %xmm0
	addq	$56, %rdi
	addq	$56, %rsi
	vmovsd	-56(%rsi), %xmm2
	vmovsd	-56(%rsp), %xmm4
	vmulsd	-48(%rsp), %xmm2, %xmm8
	vmulsd	%xmm4, %xmm0, %xmm14
	vmovsd	-48(%rsi), %xmm1
	vmovsd	-64(%rsp), %xmm3
	vmulsd	-72(%rsp), %xmm1, %xmm15
	vmulsd	%xmm3, %xmm0, %xmm5
	vmovsd	-40(%rsp), %xmm12
	vmulsd	%xmm3, %xmm2, %xmm13
	vmovsd	-32(%rsp), %xmm9
	vmulsd	%xmm4, %xmm1, %xmm3
	vsubsd	%xmm8, %xmm14, %xmm7
	vaddsd	%xmm7, %xmm1, %xmm8
	vmovsd	%xmm7, -96(%rsp)
	vsubsd	%xmm5, %xmm15, %xmm15
	vsubsd	%xmm3, %xmm13, %xmm4
	vmulsd	%xmm6, %xmm8, %xmm5
	vaddsd	%xmm2, %xmm15, %xmm14
	vmulsd	%xmm9, %xmm8, %xmm7
	vaddsd	%xmm4, %xmm0, %xmm13
	vmovsd	%xmm4, -88(%rsp)
	vmulsd	%xmm12, %xmm13, %xmm3
	vaddsd	%xmm2, %xmm5, %xmm5
	vmulsd	%xmm9, %xmm13, %xmm4
	vsubsd	%xmm3, %xmm5, %xmm5
	vmulsd	%xmm6, %xmm14, %xmm3
	vaddsd	%xmm1, %xmm4, %xmm4
	vmulsd	%xmm6, %xmm5, %xmm10
	vsubsd	%xmm3, %xmm4, %xmm4
	vmulsd	%xmm12, %xmm14, %xmm3
	vaddsd	%xmm14, %xmm5, %xmm14
	vmulsd	%xmm6, %xmm4, %xmm11
	vaddsd	%xmm0, %xmm3, %xmm3
	vsubsd	%xmm7, %xmm3, %xmm3
	vmovapd	%xmm12, %xmm7
	vaddsd	%xmm2, %xmm11, %xmm11
	vmulsd	%xmm7, %xmm5, %xmm7
	vaddsd	%xmm5, %xmm5, %xmm5
	vmulsd	%xmm12, %xmm3, %xmm12
	vaddsd	%xmm15, %xmm5, %xmm15
	vaddsd	%xmm0, %xmm7, %xmm7
	vsubsd	%xmm12, %xmm11, %xmm11
	vmovapd	%xmm9, %xmm12
	vmulsd	%xmm9, %xmm3, %xmm9
	vaddsd	%xmm1, %xmm9, %xmm9
	vsubsd	%xmm10, %xmm9, %xmm9
	vmulsd	%xmm12, %xmm4, %xmm10
	vsubsd	%xmm10, %xmm7, %xmm7
	vaddsd	%xmm11, %xmm11, %xmm10
	vaddsd	%xmm7, %xmm7, %xmm12
	vsubsd	%xmm2, %xmm10, %xmm2
	vaddsd	%xmm9, %xmm9, %xmm10
	vsubsd	%xmm0, %xmm12, %xmm0
	vaddsd	%xmm11, %xmm14, %xmm12
	vmovsd	-8(%rsp), %xmm11
	vsubsd	%xmm1, %xmm10, %xmm1
	vaddsd	%xmm2, %xmm15, %xmm15
	vmulsd	%xmm11, %xmm12, %xmm12
	vmulsd	-80(%rsp), %xmm1, %xmm10
	vaddsd	-56(%rdi), %xmm12, %xmm12
	vaddsd	%xmm10, %xmm15, %xmm15
	vaddsd	%xmm8, %xmm4, %xmm10
	vmovsd	%xmm12, -56(%rdi)
	vmovsd	-24(%rsp), %xmm12
	vaddsd	%xmm4, %xmm4, %xmm4
	vaddsd	%xmm13, %xmm3, %xmm8
	vmulsd	%xmm12, %xmm0, %xmm5
	vaddsd	%xmm3, %xmm3, %xmm3
	vaddsd	%xmm9, %xmm10, %xmm10
	vmovsd	-16(%rsp), %xmm9
	vaddsd	-96(%rsp), %xmm4, %xmm14
	vaddsd	-88(%rsp), %xmm3, %xmm3
	vmulsd	-80(%rsp), %xmm2, %xmm4
	vaddsd	%xmm7, %xmm8, %xmm8
	vsubsd	%xmm5, %xmm15, %xmm15
	vmulsd	%xmm12, %xmm2, %xmm2
	vmulsd	%xmm9, %xmm0, %xmm5
	vaddsd	%xmm1, %xmm14, %xmm14
	vaddsd	%xmm0, %xmm3, %xmm0
	vmulsd	%xmm9, %xmm1, %xmm1
	vmulsd	%xmm11, %xmm10, %xmm10
	vmulsd	%xmm11, %xmm8, %xmm8
	vaddsd	%xmm2, %xmm0, %xmm2
	vmulsd	LC0(%rip), %xmm15, %xmm15
	vaddsd	%xmm5, %xmm14, %xmm14
	vsubsd	%xmm1, %xmm2, %xmm1
	vsubsd	%xmm4, %xmm14, %xmm14
	vaddsd	-48(%rdi), %xmm10, %xmm10
	vmovsd	%xmm15, -56(%rsi)
	vmulsd	LC0(%rip), %xmm1, %xmm1
	vmulsd	LC0(%rip), %xmm14, %xmm14
	vaddsd	-40(%rdi), %xmm8, %xmm8
	vmovsd	%xmm10, -48(%rdi)
	vmovsd	%xmm8, -40(%rdi)
	vmovsd	%xmm14, -48(%rsi)
	vmovsd	%xmm1, -40(%rsi)
	cmpq	%rax, %rdi
	jne	L3
	rep; ret
LFE0:
	.globl _H2
	.zerofill __DATA,__pu_bss5,_H2,32,5
	.globl _H1
	.zerofill __DATA,__pu_bss5,_H1,32,5
	.globl _H0
	.zerofill __DATA,__pu_bss5,_H0,32,5
	.literal8
	.align 3
LC0:
	.long	1371607770
	.long	1070945621
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
