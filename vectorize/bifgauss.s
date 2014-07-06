	.text
	.align 4,0x90
	.globl __Z6fooOriddd
__Z6fooOriddd:
LFB223:
	vmovsd	LC2(%rip), %xmm3
	vmulsd	%xmm1, %xmm1, %xmm6
	vmovapd	%xmm2, %xmm7
	vmulsd	%xmm2, %xmm2, %xmm4
	vmovsd	LC1(%rip), %xmm5
	vandpd	%xmm3, %xmm1, %xmm1
	vandpd	%xmm3, %xmm7, %xmm7
	vmovsd	LC3(%rip), %xmm3
	vcomisd	%xmm3, %xmm1
	vdivsd	%xmm6, %xmm5, %xmm6
	vdivsd	%xmm4, %xmm5, %xmm4
	jbe	L20
	vcomisd	%xmm3, %xmm7
	vbroadcastsd	%xmm0, %ymm0
	vbroadcastsd	%xmm6, %ymm6
	jbe	L21
	vbroadcastsd	%xmm4, %ymm4
	xorl	%eax, %eax
	vxorpd	%xmm5, %xmm5, %xmm5
	leaq	_x(%rip), %rcx
	leaq	_y(%rip), %rdx
	.align 4,0x90
L6:
	vmovapd	(%rcx,%rax), %ymm1
	vsubpd	%ymm0, %ymm1, %ymm1
	vmulpd	%ymm1, %ymm1, %ymm2
	vcmpltpd	%ymm5, %ymm1, %ymm1
	vmulpd	%ymm6, %ymm2, %ymm3
	vmulpd	%ymm4, %ymm2, %ymm2
	vblendvpd	%ymm1, %ymm3, %ymm2, %ymm1
	vmovapd	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$8192, %rax
	jne	L6
L22:
	vzeroupper
	ret
	.align 4,0x90
L20:
	vcomisd	%xmm3, %xmm7
	jbe	L24
	vbroadcastsd	%xmm0, %ymm0
	vbroadcastsd	%xmm4, %ymm4
	xorl	%eax, %eax
	leaq	_x(%rip), %rcx
	vxorpd	%xmm3, %xmm3, %xmm3
	leaq	_y(%rip), %rdx
	.align 4,0x90
L10:
	vmovapd	(%rcx,%rax), %ymm1
	vsubpd	%ymm0, %ymm1, %ymm1
	vmulpd	%ymm1, %ymm1, %ymm2
	vcmplepd	%ymm1, %ymm3, %ymm1
	vmulpd	%ymm4, %ymm2, %ymm2
	vandpd	%ymm2, %ymm1, %ymm1
	vmovapd	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$8192, %rax
	jne	L10
	jmp	L22
	.align 4,0x90
L24:
	leaq	_y(%rip), %rdi
	movl	$8192, %edx
	xorl	%esi, %esi
	jmp	_memset
	.align 4,0x90
L21:
	leaq	_x(%rip), %rcx
	xorl	%eax, %eax
	vxorpd	%xmm3, %xmm3, %xmm3
	leaq	_y(%rip), %rdx
	.align 4,0x90
L8:
	vmovapd	(%rcx,%rax), %ymm1
	vsubpd	%ymm0, %ymm1, %ymm1
	vmulpd	%ymm1, %ymm1, %ymm2
	vcmpltpd	%ymm3, %ymm1, %ymm1
	vmulpd	%ymm6, %ymm2, %ymm2
	vandpd	%ymm2, %ymm1, %ymm1
	vmovapd	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$8192, %rax
	jne	L8
	jmp	L22
LFE223:
	.align 4,0x90
	.globl __Z3fooddd
__Z3fooddd:
LFB225:
	vmulsd	%xmm1, %xmm1, %xmm1
	vmovsd	LC1(%rip), %xmm3
	xorl	%eax, %eax
	vmulsd	%xmm2, %xmm2, %xmm2
	vbroadcastsd	%xmm0, %ymm0
	leaq	_x(%rip), %rcx
	leaq	_y(%rip), %rdx
	vdivsd	%xmm1, %xmm3, %xmm5
	vdivsd	%xmm2, %xmm3, %xmm4
	vxorpd	%xmm3, %xmm3, %xmm3
	vbroadcastsd	%xmm5, %ymm5
	vbroadcastsd	%xmm4, %ymm4
	.align 4,0x90
L26:
	vmovapd	(%rcx,%rax), %ymm1
	vsubpd	%ymm0, %ymm1, %ymm1
	vmulpd	%ymm1, %ymm1, %ymm2
	vcmpltpd	%ymm3, %ymm1, %ymm1
	vblendvpd	%ymm1, %ymm5, %ymm4, %ymm1
	vmulpd	%ymm1, %ymm2, %ymm1
	vmovapd	%ymm1, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$8192, %rax
	jne	L26
	vzeroupper
	ret
LFE225:
	.align 4,0x90
	.globl __Z3barddd
__Z3barddd:
LFB227:
	vmovapd	LC4(%rip), %ymm6
	vbroadcastsd	%xmm0, %ymm0
	xorl	%eax, %eax
	vbroadcastsd	%xmm1, %ymm1
	vbroadcastsd	%xmm2, %ymm2
	vxorpd	%xmm5, %xmm5, %xmm5
	leaq	_x(%rip), %rcx
	leaq	_y(%rip), %rdx
	.align 4,0x90
L29:
	vmovapd	(%rcx,%rax), %ymm4
	vsubpd	%ymm0, %ymm4, %ymm4
	vcmplepd	%ymm4, %ymm5, %ymm3
	vblendvpd	%ymm3, %ymm2, %ymm1, %ymm3
	vdivpd	%ymm3, %ymm4, %ymm3
	vmulpd	%ymm3, %ymm3, %ymm3
	vmulpd	%ymm6, %ymm3, %ymm3
	vmovapd	%ymm3, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$8192, %rax
	jne	L29
	vzeroupper
	ret
LFE227:
	.globl _y
	.zerofill __DATA,__pu_bss5,_y,8192,5
	.globl _x
	.zerofill __DATA,__pu_bss5,_x,8192,5
	.literal8
	.align 3
LC1:
	.long	0
	.long	-1075838976
	.literal16
	.align 4
LC2:
	.long	4294967295
	.long	2147483647
	.long	0
	.long	0
	.literal8
	.align 3
LC3:
	.long	4276863648
	.long	968116299
	.const
	.align 5
LC4:
	.long	0
	.long	-1075838976
	.long	0
	.long	-1075838976
	.long	0
	.long	-1075838976
	.long	0
	.long	-1075838976
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
	.quad	LFB223-.
	.set L$set$2,LFE223-LFB223
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB225-.
	.set L$set$4,LFE225-LFB225
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB227-.
	.set L$set$6,LFE227-LFB227
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
