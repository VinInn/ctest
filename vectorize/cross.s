	.text
	.align 4,0x90
	.globl __Z3csfU8__vectorfS_
__Z3csfU8__vectorfS_:
LFB5:
	vpermilps	$9, %xmm0, %xmm3
	vpermilps	$18, %xmm1, %xmm2
	vmulps	%xmm2, %xmm3, %xmm2
	vpermilps	$9, %xmm1, %xmm1
	vpermilps	$18, %xmm0, %xmm0
	vmulps	%xmm1, %xmm0, %xmm0
	vsubps	%xmm0, %xmm2, %xmm0
	ret
LFE5:
	.align 4,0x90
	.globl __Z3csfU8__vectordS_
__Z3csfU8__vectordS_:
LFB6:
	vpermpd	$9, %ymm0, %ymm3
	vpermpd	$18, %ymm1, %ymm2
	vpermpd	$18, %ymm0, %ymm0
	vmulpd	%ymm2, %ymm3, %ymm2
	vpermpd	$9, %ymm1, %ymm1
	vmulpd	%ymm1, %ymm0, %ymm0
	vsubpd	%ymm0, %ymm2, %ymm0
	ret
LFE6:
	.align 4,0x90
	.globl __Z2dpU8__vectorfS_
__Z2dpU8__vectorfS_:
LFB7:
	vmovaps	%xmm0, -40(%rsp)
	vmovss	-40(%rsp), %xmm0
	vmovaps	%xmm1, -24(%rsp)
	vmovss	-36(%rsp), %xmm1
	vmulss	-24(%rsp), %xmm0, %xmm0
	vmulss	-20(%rsp), %xmm1, %xmm1
	vaddss	LC0(%rip), %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	-32(%rsp), %xmm1
	vmulss	-16(%rsp), %xmm1, %xmm1
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	-28(%rsp), %xmm1
	vmulss	-12(%rsp), %xmm1, %xmm1
	vaddss	%xmm0, %xmm1, %xmm0
	ret
LFE7:
	.align 4,0x90
	.globl __Z2dpU8__vectordS_
__Z2dpU8__vectordS_:
LFB8:
	pushq	%rbp
LCFI0:
	movq	%rsp, %rbp
LCFI1:
	andq	$-32, %rsp
	addq	$16, %rsp
	vmovapd	%ymm0, -80(%rsp)
	vmovapd	%ymm1, -48(%rsp)
	vmovsd	-80(%rsp), %xmm0
	vmovsd	-72(%rsp), %xmm1
	vmulsd	-48(%rsp), %xmm0, %xmm0
	vmulsd	-40(%rsp), %xmm1, %xmm1
	vaddsd	LC1(%rip), %xmm0, %xmm0
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-64(%rsp), %xmm1
	vmulsd	-32(%rsp), %xmm1, %xmm1
	vaddsd	%xmm1, %xmm0, %xmm0
	vmovsd	-56(%rsp), %xmm1
	vmulsd	-24(%rsp), %xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vzeroupper
	leave
LCFI2:
	ret
LFE8:
	.align 4,0x90
	.globl __Z13cross_productU8__vectordS_
__Z13cross_productU8__vectordS_:
LFB9:
	vpermpd	$9, %ymm0, %ymm3
	vpermpd	$18, %ymm1, %ymm2
	vpermpd	$18, %ymm0, %ymm0
	vmulpd	%ymm2, %ymm3, %ymm2
	vpermpd	$9, %ymm1, %ymm1
	vmulpd	%ymm1, %ymm0, %ymm0
	vsubpd	%ymm0, %ymm2, %ymm0
	ret
LFE9:
	.literal4
	.align 2
LC0:
	.long	0
	.literal8
	.align 3
LC1:
	.long	0
	.long	0
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
	.quad	LFB5-.
	.set L$set$2,LFE5-LFB5
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB6-.
	.set L$set$4,LFE6-LFB6
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB7-.
	.set L$set$6,LFE7-LFB7
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB8-.
	.set L$set$8,LFE8-LFB8
	.quad L$set$8
	.byte	0
	.byte	0x4
	.set L$set$9,LCFI0-LFB8
	.long L$set$9
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$10,LCFI1-LCFI0
	.long L$set$10
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$11,LCFI2-LCFI1
	.long L$set$11
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$12,LEFDE9-LASFDE9
	.long L$set$12
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB9-.
	.set L$set$13,LFE9-LFB9
	.quad L$set$13
	.byte	0
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
