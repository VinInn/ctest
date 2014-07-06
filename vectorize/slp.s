	.text
	.align 4,0x90
	.globl __Z3foov
__Z3foov:
LFB0:
	movq	_x(%rip), %rdx
	movq	_y(%rip), %rax
	movss	(%rdx), %xmm0
	movss	%xmm0, (%rax)
	movss	4(%rdx), %xmm0
	movss	%xmm0, 4(%rax)
	movss	8(%rdx), %xmm0
	movss	%xmm0, 8(%rax)
	movss	12(%rdx), %xmm0
	movss	%xmm0, 12(%rax)
	movss	16(%rdx), %xmm0
	movss	%xmm0, 16(%rax)
	movss	20(%rdx), %xmm0
	movss	%xmm0, 20(%rax)
	movss	24(%rdx), %xmm0
	movss	%xmm0, 24(%rax)
	movss	28(%rdx), %xmm0
	movss	%xmm0, 28(%rax)
	ret
LFE0:
	.align 4,0x90
	.globl __Z3voov
__Z3voov:
LFB1:
	movss	_v1(%rip), %xmm0
	movss	%xmm0, _v2(%rip)
	movss	4+_v1(%rip), %xmm0
	movss	%xmm0, 4+_v2(%rip)
	movss	8+_v1(%rip), %xmm0
	movss	%xmm0, 8+_v2(%rip)
	movss	12+_v1(%rip), %xmm0
	movss	%xmm0, 12+_v2(%rip)
	movss	16+_v1(%rip), %xmm0
	movss	%xmm0, 16+_v2(%rip)
	movss	20+_v1(%rip), %xmm0
	movss	%xmm0, 20+_v2(%rip)
	movss	24+_v1(%rip), %xmm0
	movss	%xmm0, 24+_v2(%rip)
	movss	28+_v1(%rip), %xmm0
	movss	%xmm0, 28+_v2(%rip)
	ret
LFE1:
	.align 4,0x90
	.globl __Z3aoov
__Z3aoov:
LFB2:
	movss	_a1(%rip), %xmm0
	movss	%xmm0, _a2(%rip)
	movss	4+_a1(%rip), %xmm0
	movss	%xmm0, 4+_a2(%rip)
	movss	8+_a1(%rip), %xmm0
	movss	%xmm0, 8+_a2(%rip)
	movss	12+_a1(%rip), %xmm0
	movss	%xmm0, 12+_a2(%rip)
	movss	16+_a1(%rip), %xmm0
	movss	%xmm0, 16+_a2(%rip)
	movss	20+_a1(%rip), %xmm0
	movss	%xmm0, 20+_a2(%rip)
	movss	24+_a1(%rip), %xmm0
	movss	%xmm0, 24+_a2(%rip)
	movss	28+_a1(%rip), %xmm0
	movss	%xmm0, 28+_a2(%rip)
	ret
LFE2:
	.align 4,0x90
	.globl __Z4loopv
__Z4loopv:
LFB3:
	movq	_y(%rip), %rcx
	movq	_z(%rip), %rdx
	movq	_x(%rip), %rax
	movups	(%rcx), %xmm1
	movups	(%rdx), %xmm0
	addps	%xmm1, %xmm0
	movups	16(%rdx), %xmm1
	movups	%xmm0, (%rax)
	movups	16(%rcx), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm0, 16(%rax)
	ret
LFE3:
	.align 4,0x90
	.globl __Z4voopv
__Z4voopv:
LFB4:
	movaps	_v2(%rip), %xmm0
	addps	_v3(%rip), %xmm0
	movaps	%xmm0, _v1(%rip)
	movaps	16+_v2(%rip), %xmm0
	addps	16+_v3(%rip), %xmm0
	movaps	%xmm0, 16+_v1(%rip)
	ret
LFE4:
	.align 4,0x90
	.globl __Z4aoopv
__Z4aoopv:
LFB5:
	movaps	_a2(%rip), %xmm0
	addps	_a3(%rip), %xmm0
	movaps	%xmm0, _a1(%rip)
	movaps	16+_a2(%rip), %xmm0
	addps	16+_a3(%rip), %xmm0
	movaps	%xmm0, 16+_a1(%rip)
	ret
LFE5:
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB6:
	movq	_y(%rip), %rcx
	movq	_z(%rip), %rdx
	movq	_x(%rip), %rax
	movups	16(%rcx), %xmm0
	movups	16(%rdx), %xmm1
	movups	(%rdx), %xmm2
	addps	%xmm1, %xmm0
	movups	(%rcx), %xmm1
	addps	%xmm2, %xmm1
	movups	%xmm0, 16(%rax)
	movups	%xmm1, (%rax)
	ret
LFE6:
	.align 4,0x90
	.globl __Z4abarv
__Z4abarv:
LFB7:
	movaps	16+_a2(%rip), %xmm0
	movaps	_a2(%rip), %xmm1
	addps	16+_a3(%rip), %xmm0
	addps	_a3(%rip), %xmm1
	movaps	%xmm0, 16+_a1(%rip)
	movaps	%xmm1, _a1(%rip)
	ret
LFE7:
	.align 4,0x90
	.globl __Z3sum1AS_
__Z3sum1AS_:
LFB8:
	movss	40(%rsp), %xmm7
	movq	%rdi, %rax
	movss	44(%rsp), %xmm6
	movss	48(%rsp), %xmm5
	movss	52(%rsp), %xmm4
	movss	56(%rsp), %xmm3
	movss	60(%rsp), %xmm2
	movss	64(%rsp), %xmm1
	movss	68(%rsp), %xmm0
	addss	8(%rsp), %xmm7
	addss	12(%rsp), %xmm6
	addss	16(%rsp), %xmm5
	addss	20(%rsp), %xmm4
	movss	%xmm7, (%rdi)
	addss	24(%rsp), %xmm3
	movss	%xmm6, 4(%rdi)
	addss	28(%rsp), %xmm2
	movss	%xmm5, 8(%rdi)
	addss	32(%rsp), %xmm1
	movss	%xmm4, 12(%rdi)
	addss	36(%rsp), %xmm0
	movss	%xmm3, 16(%rdi)
	movss	%xmm2, 20(%rdi)
	movss	%xmm1, 24(%rdi)
	movss	%xmm0, 28(%rdi)
	ret
LFE8:
	.align 4,0x90
	.globl __Z4suml1AS_
__Z4suml1AS_:
LFB9:
	movaps	8(%rsp), %xmm0
	movq	%rdi, %rax
	addps	40(%rsp), %xmm0
	movaps	%xmm0, -24(%rsp)
	movaps	%xmm0, 8(%rsp)
	movaps	24(%rsp), %xmm0
	movq	-24(%rsp), %rcx
	addps	56(%rsp), %xmm0
	movaps	%xmm0, -24(%rsp)
	movq	-24(%rsp), %rdx
	movaps	%xmm0, 24(%rsp)
	movq	%rcx, (%rdi)
	movq	16(%rsp), %rcx
	movq	%rdx, 16(%rdi)
	movq	32(%rsp), %rdx
	movq	%rcx, 8(%rdi)
	movq	%rdx, 24(%rdi)
	ret
LFE9:
	.align 4,0x90
	.globl __Z5dosumv
__Z5dosumv:
LFB10:
	movaps	16+_a2(%rip), %xmm0
	movaps	_a2(%rip), %xmm1
	addps	16+_a3(%rip), %xmm0
	addps	_a3(%rip), %xmm1
	movaps	%xmm0, 16+_a1(%rip)
	movaps	%xmm1, _a1(%rip)
	ret
LFE10:
	.align 4,0x90
	.globl __Z6dosumlv
__Z6dosumlv:
LFB11:
	movq	_a2(%rip), %rax
	movq	%rax, -104(%rsp)
	movq	8+_a2(%rip), %rax
	movq	%rax, -96(%rsp)
	movq	16+_a2(%rip), %rax
	movaps	-104(%rsp), %xmm0
	movq	%rax, -88(%rsp)
	movq	24+_a2(%rip), %rax
	movq	%rax, -80(%rsp)
	movq	_a3(%rip), %rax
	movq	%rax, -72(%rsp)
	movq	8+_a3(%rip), %rax
	movq	%rax, -64(%rsp)
	movq	16+_a3(%rip), %rax
	addps	-72(%rsp), %xmm0
	movq	%rax, -56(%rsp)
	movq	24+_a3(%rip), %rax
	movaps	%xmm0, -120(%rsp)
	movaps	%xmm0, -104(%rsp)
	movaps	-88(%rsp), %xmm0
	movq	-120(%rsp), %rdx
	movq	%rax, -48(%rsp)
	addps	-56(%rsp), %xmm0
	movq	%rdx, _a1(%rip)
	movq	-96(%rsp), %rdx
	movaps	%xmm0, -120(%rsp)
	movq	-120(%rsp), %rax
	movaps	%xmm0, -88(%rsp)
	movq	%rdx, 8+_a1(%rip)
	movq	%rax, 16+_a1(%rip)
	movq	-80(%rsp), %rax
	movq	%rax, 24+_a1(%rip)
	ret
LFE11:
	.globl _a3
	.zerofill __DATA,__pu_bss5,_a3,32,5
	.globl _a2
	.zerofill __DATA,__pu_bss5,_a2,32,5
	.globl _a1
	.zerofill __DATA,__pu_bss5,_a1,32,5
	.globl _v3
	.zerofill __DATA,__pu_bss5,_v3,32,5
	.globl _v2
	.zerofill __DATA,__pu_bss5,_v2,32,5
	.globl _v1
	.zerofill __DATA,__pu_bss5,_v1,32,5
	.globl _z
	.zerofill __DATA,__pu_bss4,_z,8,4
	.globl _y
	.zerofill __DATA,__pu_bss4,_y,8,4
	.globl _x
	.zerofill __DATA,__pu_bss4,_x,8,4
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
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$14,LFE6-LFB6
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB7-.
	.set L$set$16,LFE7-LFB7
	.quad L$set$16
	.byte	0
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$17,LEFDE17-LASFDE17
	.long L$set$17
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB8-.
	.set L$set$18,LFE8-LFB8
	.quad L$set$18
	.byte	0
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$19,LEFDE19-LASFDE19
	.long L$set$19
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB9-.
	.set L$set$20,LFE9-LFB9
	.quad L$set$20
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$21,LEFDE21-LASFDE21
	.long L$set$21
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB10-.
	.set L$set$22,LFE10-LFB10
	.quad L$set$22
	.byte	0
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$23,LEFDE23-LASFDE23
	.long L$set$23
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB11-.
	.set L$set$24,LFE11-LFB11
	.quad L$set$24
	.byte	0
	.align 3
LEFDE23:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
