	.text
	.align 4,0x90
	.globl __Z3fooddd
__Z3fooddd:
LFB87:
	vmovsd	LC0(%rip), %xmm6
	vbroadcastsd	%xmm2, %ymm2
	xorl	%eax, %eax
	vmovapd	LC1(%rip), %ymm4
	leaq	_x(%rip), %rcx
	vdivsd	%xmm1, %xmm6, %xmm1
	leaq	_y(%rip), %rdx
	vbroadcastsd	%xmm1, %ymm6
	.align 4,0x90
L2:
	vmovapd	(%rcx,%rax), %ymm1
	vmulpd	%ymm6, %ymm1, %ymm0
	vmovapd	%ymm0, %ymm3
	vfnmadd132pd	%ymm0, %ymm4, %ymm3
	vcmpltpd	%ymm4, %ymm0, %ymm0
	vsqrtpd	%ymm3, %ymm5
	vmulpd	%ymm1, %ymm3, %ymm1
	vmulpd	%ymm2, %ymm5, %ymm5
	vmulpd	%ymm1, %ymm5, %ymm1
	vandpd	%ymm1, %ymm0, %ymm0
	vmovapd	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$8192, %rax
	jne	L2
	vzeroupper
	ret
LFE87:
	.globl _y
	.zerofill __DATA,__pu_bss5,_y,8192,5
	.globl _x
	.zerofill __DATA,__pu_bss5,_x,8192,5
	.literal8
	.align 3
LC0:
	.long	0
	.long	1072693248
	.const
	.align 5
LC1:
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
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
	.quad	LFB87-.
	.set L$set$2,LFE87-LFB87
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
