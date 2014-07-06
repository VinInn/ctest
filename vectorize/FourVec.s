	.text
	.align 4,0x90
	.globl __Z4sum1R13LorentzVectorfRKS_S2_
__Z4sum1R13LorentzVectorfRKS_S2_:
LFB10:
	movaps	(%rsi), %xmm1
	shufps	$0, %xmm0, %xmm0
	addps	(%rdx), %xmm1
	mulps	%xmm0, %xmm1
	addps	(%rdi), %xmm1
	movaps	%xmm1, (%rdi)
	ret
LFE10:
	.align 4,0x90
	.globl __Z4msumR13LorentzVectorfRKS_S2_
__Z4msumR13LorentzVectorfRKS_S2_:
LFB12:
	movaps	(%rdx), %xmm2
	shufps	$0, %xmm0, %xmm0
	movaps	(%rsi), %xmm3
	movaps	%xmm2, %xmm1
	addps	%xmm2, %xmm2
	addps	%xmm3, %xmm1
	addps	%xmm3, %xmm1
	subps	%xmm2, %xmm1
	mulps	%xmm0, %xmm1
	movaps	%xmm1, (%rdi)
	ret
LFE12:
	.align 4,0x90
	.globl __ZN3aos4lsumEv
__ZN3aos4lsumEv:
LFB13:
	movss	__ZN3aos1sE(%rip), %xmm1
	leaq	12+__ZN3aos1cE(%rip), %rax
	leaq	12+__ZN3aos1bE(%rip), %rcx
	leaq	16396+__ZN3aos1cE(%rip), %rsi
	leaq	__ZN3aos1aE(%rip), %rdx
	.align 4,0x90
L4:
	movss	-12(%rcx), %xmm9
	addq	$16, %rax
	addq	$16, %rcx
	movss	-28(%rax), %xmm5
	addq	$16, %rdx
	movaps	%xmm9, %xmm0
	movss	-24(%rcx), %xmm8
	addss	%xmm5, %xmm0
	movss	-24(%rax), %xmm4
	addss	%xmm5, %xmm5
	movss	-20(%rcx), %xmm7
	movss	-20(%rax), %xmm3
	addss	%xmm9, %xmm0
	movss	-16(%rcx), %xmm6
	movss	-16(%rax), %xmm2
	subss	%xmm5, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -16(%rdx)
	movaps	%xmm8, %xmm0
	addss	%xmm4, %xmm0
	addss	%xmm4, %xmm4
	addss	%xmm8, %xmm0
	subss	%xmm4, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -12(%rdx)
	movaps	%xmm7, %xmm0
	addss	%xmm3, %xmm0
	addss	%xmm3, %xmm3
	addss	%xmm7, %xmm0
	subss	%xmm3, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -8(%rdx)
	movaps	%xmm6, %xmm0
	addss	%xmm2, %xmm0
	addss	%xmm2, %xmm2
	addss	%xmm6, %xmm0
	subss	%xmm2, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -4(%rdx)
	cmpq	%rsi, %rax
	jne	L4
	leaq	__ZN3aos1mE(%rip), %rcx
	leaq	__ZN3aos1bE(%rip), %rdx
	leaq	4096+__ZN3aos1mE(%rip), %rsi
	leaq	__ZN3aos1aE(%rip), %rax
	.align 4,0x90
L8:
	movaps	(%rax), %xmm1
	addq	$16, %rcx
	addq	$64, %rdx
	movaps	32(%rax), %xmm6
	addq	$64, %rax
	movaps	-48(%rax), %xmm3
	movaps	%xmm1, %xmm2
	movaps	-16(%rax), %xmm0
	movaps	%xmm6, %xmm9
	movaps	-64(%rdx), %xmm5
	shufps	$136, %xmm3, %xmm2
	shufps	$221, %xmm3, %xmm1
	movaps	-32(%rdx), %xmm4
	shufps	$136, %xmm0, %xmm9
	shufps	$221, %xmm0, %xmm6
	movaps	-48(%rdx), %xmm3
	movaps	%xmm5, %xmm8
	movaps	-16(%rdx), %xmm0
	movaps	%xmm4, %xmm7
	shufps	$136, %xmm3, %xmm8
	shufps	$221, %xmm3, %xmm5
	movaps	%xmm5, %xmm3
	shufps	$136, %xmm0, %xmm7
	shufps	$221, %xmm0, %xmm4
	movaps	%xmm1, %xmm0
	shufps	$136, %xmm4, %xmm3
	movaps	%xmm8, %xmm10
	shufps	$136, %xmm6, %xmm0
	mulps	%xmm3, %xmm0
	movaps	%xmm2, %xmm3
	shufps	$221, %xmm7, %xmm10
	shufps	$221, %xmm9, %xmm3
	mulps	%xmm10, %xmm3
	shufps	$136, %xmm9, %xmm2
	shufps	$136, %xmm7, %xmm8
	mulps	%xmm8, %xmm2
	shufps	$221, %xmm6, %xmm1
	shufps	$221, %xmm4, %xmm5
	mulps	%xmm5, %xmm1
	addps	%xmm3, %xmm0
	addps	%xmm2, %xmm0
	subps	%xmm1, %xmm0
	movaps	%xmm0, -16(%rcx)
	cmpq	%rsi, %rcx
	jne	L8
	rep; ret
LFE13:
	.align 4,0x90
	.globl __ZN4soa46soAsumEv
__ZN4soa46soAsumEv:
LFB32:
	leaq	__ZN4soa42m1E(%rip), %rax
	movss	__ZN4soa41sE(%rip), %xmm1
	movl	$1024, 8+__ZN4soa41cE(%rip)
	movq	%rax, __ZN4soa41aE(%rip)
	leaq	__ZN4soa42m2E(%rip), %rax
	movq	%rax, __ZN4soa41bE(%rip)
	leaq	__ZN4soa42m3E(%rip), %rax
	movq	%rax, __ZN4soa41cE(%rip)
	leaq	12288+__ZN4soa42m2E(%rip), %rcx
	movl	$1024, 8+__ZN4soa41bE(%rip)
	leaq	12288+__ZN4soa42m3E(%rip), %rax
	movl	$1024, 8+__ZN4soa41aE(%rip)
	leaq	__ZN4soa42m1E(%rip), %rdx
	leaq	16384+__ZN4soa42m3E(%rip), %rsi
	.align 4,0x90
L13:
	movss	-12288(%rax), %xmm5
	addq	$4, %rax
	addq	$4, %rcx
	movss	-12292(%rcx), %xmm9
	addq	$4, %rdx
	movss	-8196(%rcx), %xmm8
	movaps	%xmm9, %xmm0
	movss	-8196(%rax), %xmm4
	addss	%xmm5, %xmm0
	movss	-4100(%rcx), %xmm7
	addss	%xmm5, %xmm5
	movss	-4100(%rax), %xmm3
	movss	-4(%rcx), %xmm6
	addss	%xmm9, %xmm0
	movss	-4(%rax), %xmm2
	subss	%xmm5, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -4(%rdx)
	movaps	%xmm8, %xmm0
	addss	%xmm4, %xmm0
	addss	%xmm4, %xmm4
	addss	%xmm8, %xmm0
	subss	%xmm4, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, 4092(%rdx)
	movaps	%xmm7, %xmm0
	addss	%xmm3, %xmm0
	addss	%xmm3, %xmm3
	addss	%xmm7, %xmm0
	subss	%xmm3, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, 8188(%rdx)
	movaps	%xmm6, %xmm0
	addss	%xmm2, %xmm0
	addss	%xmm2, %xmm2
	addss	%xmm6, %xmm0
	subss	%xmm2, %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, 12284(%rdx)
	cmpq	%rsi, %rax
	jne	L13
	leaq	__ZN4soa41mE(%rip), %rcx
	leaq	12288+__ZN4soa42m1E(%rip), %rdx
	leaq	12288+__ZN4soa42m2E(%rip), %rax
	leaq	4096+__ZN4soa41mE(%rip), %rsi
	.align 4,0x90
L17:
	movaps	-8192(%rax), %xmm0
	addq	$16, %rcx
	addq	$16, %rdx
	movaps	-4096(%rax), %xmm1
	addq	$16, %rax
	mulps	-8208(%rdx), %xmm0
	mulps	-4112(%rdx), %xmm1
	addps	%xmm1, %xmm0
	movaps	-12304(%rax), %xmm1
	mulps	-12304(%rdx), %xmm1
	addps	%xmm1, %xmm0
	movaps	-16(%rax), %xmm1
	mulps	-16(%rdx), %xmm1
	subps	%xmm1, %xmm0
	movaps	%xmm0, -16(%rcx)
	cmpq	%rsi, %rcx
	jne	L17
	rep; ret
LFE32:
	.align 4,0x90
	.globl __ZN4soa36soAsumEv
__ZN4soa36soAsumEv:
LFB33:
	leaq	__ZN4soa32m1E(%rip), %rax
	movss	__ZN4soa31sE(%rip), %xmm4
	movl	$1024, 8+__ZN4soa31cE(%rip)
	movq	%rax, __ZN4soa31aE(%rip)
	leaq	__ZN4soa32m2E(%rip), %rax
	movq	%rax, __ZN4soa31bE(%rip)
	leaq	__ZN4soa32m3E(%rip), %rax
	movq	%rax, __ZN4soa31cE(%rip)
	leaq	8192+__ZN4soa32m2E(%rip), %rcx
	movl	$1024, 8+__ZN4soa31bE(%rip)
	leaq	8192+__ZN4soa32m3E(%rip), %rax
	movl	$1024, 8+__ZN4soa31aE(%rip)
	leaq	__ZN4soa32m1E(%rip), %rdx
	leaq	12288+__ZN4soa32m3E(%rip), %rsi
	.align 4,0x90
L21:
	movss	-8192(%rcx), %xmm7
	addq	$4, %rax
	addq	$4, %rcx
	movss	-8196(%rax), %xmm3
	addq	$4, %rdx
	movaps	%xmm7, %xmm0
	movss	-4100(%rcx), %xmm6
	addss	%xmm3, %xmm0
	movss	-4100(%rax), %xmm2
	addss	%xmm3, %xmm3
	movss	-4(%rcx), %xmm5
	movss	-4(%rax), %xmm1
	addss	%xmm7, %xmm0
	subss	%xmm3, %xmm0
	mulss	%xmm4, %xmm0
	movss	%xmm0, -4(%rdx)
	movaps	%xmm6, %xmm0
	addss	%xmm2, %xmm0
	addss	%xmm2, %xmm2
	addss	%xmm6, %xmm0
	subss	%xmm2, %xmm0
	mulss	%xmm4, %xmm0
	movss	%xmm0, 4092(%rdx)
	movaps	%xmm5, %xmm0
	addss	%xmm1, %xmm0
	addss	%xmm1, %xmm1
	addss	%xmm5, %xmm0
	subss	%xmm1, %xmm0
	mulss	%xmm4, %xmm0
	movss	%xmm0, 8188(%rdx)
	cmpq	%rsi, %rax
	jne	L21
	leaq	__ZN4soa31mE(%rip), %rcx
	leaq	8192+__ZN4soa32m1E(%rip), %rdx
	leaq	8192+__ZN4soa32m2E(%rip), %rax
	leaq	4096+__ZN4soa31mE(%rip), %rsi
	.align 4,0x90
L25:
	movaps	-4096(%rax), %xmm0
	addq	$16, %rcx
	addq	$16, %rdx
	movaps	(%rax), %xmm1
	addq	$16, %rax
	mulps	-4112(%rdx), %xmm0
	mulps	-16(%rdx), %xmm1
	addps	%xmm1, %xmm0
	movaps	-8208(%rax), %xmm1
	mulps	-8208(%rdx), %xmm1
	addps	%xmm1, %xmm0
	movaps	%xmm0, -16(%rcx)
	cmpq	%rsi, %rcx
	jne	L25
	rep; ret
LFE33:
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
__GLOBAL__sub_I_FourVec.cc:
LFB35:
	leaq	__ZN3aos1aE(%rip), %rax
	xorps	%xmm0, %xmm0
	leaq	16384+__ZN3aos1aE(%rip), %rdx
	.align 4
L29:
	movaps	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	L29
	leaq	__ZN3aos1bE(%rip), %rax
	xorps	%xmm0, %xmm0
	leaq	16224+__ZN3aos1bE(%rip), %rdx
	.align 4
L36:
	movaps	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	L36
	leaq	__ZN3aos1cE(%rip), %rax
	xorps	%xmm0, %xmm0
	leaq	16384+__ZN3aos1cE(%rip), %rdx
	.align 4
L35:
	movaps	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%rdx, %rax
	jne	L35
	rep; ret
LFE35:
	.globl __ZN4soa31mE
	.zerofill __DATA,__pu_bss5,__ZN4soa31mE,4096,5
	.globl __ZN4soa31sE
	.zerofill __DATA,__pu_bss2,__ZN4soa31sE,4,2
	.globl __ZN4soa31cE
	.zerofill __DATA,__pu_bss4,__ZN4soa31cE,16,4
	.globl __ZN4soa31bE
	.zerofill __DATA,__pu_bss4,__ZN4soa31bE,16,4
	.globl __ZN4soa31aE
	.zerofill __DATA,__pu_bss4,__ZN4soa31aE,16,4
	.globl __ZN4soa32m3E
	.zerofill __DATA,__pu_bss5,__ZN4soa32m3E,12288,5
	.globl __ZN4soa32m2E
	.zerofill __DATA,__pu_bss5,__ZN4soa32m2E,12288,5
	.globl __ZN4soa32m1E
	.zerofill __DATA,__pu_bss5,__ZN4soa32m1E,12288,5
	.globl __ZN4soa35arenaE
	.zerofill __DATA,__pu_bss5,__ZN4soa35arenaE,36864,5
	.globl __ZN4soa31NE
	.data
	.align 2
__ZN4soa31NE:
	.long	1024
	.globl __ZN4soa41mE
	.zerofill __DATA,__pu_bss5,__ZN4soa41mE,4096,5
	.globl __ZN4soa41sE
	.zerofill __DATA,__pu_bss2,__ZN4soa41sE,4,2
	.globl __ZN4soa41cE
	.zerofill __DATA,__pu_bss4,__ZN4soa41cE,16,4
	.globl __ZN4soa41bE
	.zerofill __DATA,__pu_bss4,__ZN4soa41bE,16,4
	.globl __ZN4soa41aE
	.zerofill __DATA,__pu_bss4,__ZN4soa41aE,16,4
	.globl __ZN4soa42m3E
	.zerofill __DATA,__pu_bss5,__ZN4soa42m3E,16384,5
	.globl __ZN4soa42m2E
	.zerofill __DATA,__pu_bss5,__ZN4soa42m2E,16384,5
	.globl __ZN4soa42m1E
	.zerofill __DATA,__pu_bss5,__ZN4soa42m1E,16384,5
	.globl __ZN4soa45arenaE
	.zerofill __DATA,__pu_bss5,__ZN4soa45arenaE,49152,5
	.globl __ZN4soa41NE
	.data
	.align 2
__ZN4soa41NE:
	.long	1024
	.globl __ZN3aos1mE
	.zerofill __DATA,__pu_bss5,__ZN3aos1mE,4096,5
	.globl __ZN3aos1sE
	.zerofill __DATA,__pu_bss2,__ZN3aos1sE,4,2
	.globl __ZN3aos1cE
	.zerofill __DATA,__pu_bss5,__ZN3aos1cE,16384,5
	.globl __ZN3aos1bE
	.zerofill __DATA,__pu_bss5,__ZN3aos1bE,16224,5
	.globl __ZN3aos1aE
	.zerofill __DATA,__pu_bss5,__ZN3aos1aE,16384,5
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
	.quad	LFB10-.
	.set L$set$2,LFE10-LFB10
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB12-.
	.set L$set$4,LFE12-LFB12
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB13-.
	.set L$set$6,LFE13-LFB13
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB32-.
	.set L$set$8,LFE32-LFB32
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB33-.
	.set L$set$10,LFE33-LFB33
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB35-.
	.set L$set$12,LFE35-LFB35
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_FourVec.cc
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
