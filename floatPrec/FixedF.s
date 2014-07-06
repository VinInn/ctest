	.text
	.align 4,0x90
	.globl __Z4multoo
__Z4multoo:
LFB1063:
	imulq	%rdx, %rsi
	movq	%rdi, %rax
	imulq	%rdi, %rcx
	mulq	%rdx
	addq	%rcx, %rsi
	addq	%rsi, %rdx
	ret
LFE1063:
	.align 4,0x90
	.globl __Z5multDoo
__Z5multDoo:
LFB1064:
	imulq	%rdx, %rsi
	movq	%rdx, %rax
	mulq	%rdi
	imulq	%rdi, %rcx
	movq	%rax, %r9
	movq	%rdx, %r10
	xorl	%edx, %edx
	movabsq	$2251799813685248, %rax
	addq	%rcx, %rsi
	addq	%rsi, %r10
	addq	%r9, %rax
	adcq	%r10, %rdx
	shrdq	$52, %rdx, %rax
	shrq	$52, %rdx
	ret
LFE1064:
	.align 4,0x90
	.globl __Z4multDv4_yS_
__Z4multDv4_yS_:
LFB1065:
	vpsrlq	$32, %ymm0, %ymm4
	vpsrlq	$32, %ymm1, %ymm3
	vpmuludq	%ymm1, %ymm0, %ymm2
	vpmuludq	%ymm0, %ymm3, %ymm0
	vpmuludq	%ymm1, %ymm4, %ymm1
	vpaddq	%ymm0, %ymm1, %ymm1
	vpsllq	$32, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm2, %ymm0
	vpaddq	LC0(%rip), %ymm0, %ymm0
	vpsrlq	$23, %ymm0, %ymm0
	ret
LFE1065:
	.align 4,0x90
	.globl __Z5multIDv4_yS_
__Z5multIDv4_yS_:
LFB1066:
	vpmuludq	%ymm1, %ymm0, %ymm1
	vpaddq	LC0(%rip), %ymm1, %ymm0
	vpsrlq	$23, %ymm0, %ymm0
	ret
LFE1066:
	.align 4,0x90
	.globl __Z5multIDv2_yS_
__Z5multIDv2_yS_:
LFB1067:
	vpmuludq	%xmm1, %xmm0, %xmm1
	vpaddq	LC1(%rip), %xmm1, %xmm0
	vpsrlq	$23, %xmm0, %xmm0
	ret
LFE1067:
	.align 4,0x90
	.globl __Z4irexDv4_fRDv4_iRDv4_j
__Z4irexDv4_fRDv4_iRDv4_j:
LFB1068:
	vpsrad	$23, %xmm0, %xmm1
	vpand	LC4(%rip), %xmm0, %xmm0
	vpor	LC5(%rip), %xmm0, %xmm0
	vpand	LC2(%rip), %xmm1, %xmm1
	vpsubd	LC3(%rip), %xmm1, %xmm1
	vmovdqa	%xmm1, (%rdi)
	vmovdqa	%xmm0, (%rsi)
	ret
LFE1068:
	.align 4,0x90
	.globl __Z4irexDv8_fRDv8_iRDv8_j
__Z4irexDv8_fRDv8_iRDv8_j:
LFB1069:
	vpsrad	$23, %ymm0, %ymm1
	vpand	LC8(%rip), %ymm0, %ymm0
	vpor	LC9(%rip), %ymm0, %ymm0
	vpand	LC6(%rip), %ymm1, %ymm1
	vpsubd	LC7(%rip), %ymm1, %ymm1
	vmovdqa	%ymm1, (%rdi)
	vmovdqa	%ymm0, (%rsi)
	vzeroupper
	ret
LFE1069:
	.align 4,0x90
	.globl __Z5multIDv4_jS_
__Z5multIDv4_jS_:
LFB1070:
	vpmuludq	%xmm1, %xmm0, %xmm2
	vpshufd	$177, %xmm1, %xmm1
	vpshufd	$177, %xmm0, %xmm0
	vpmuludq	%xmm1, %xmm0, %xmm0
	vmovdqa	LC1(%rip), %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vpaddq	%xmm1, %xmm2, %xmm1
	vpsrlq	$23, %xmm0, %xmm0
	vpsrlq	$23, %xmm1, %xmm1
	vpsllq	$32, %xmm0, %xmm0
	vpor	%xmm1, %xmm0, %xmm0
	ret
LFE1070:
	.align 4,0x90
	.globl __Z5multIDv8_jS_
__Z5multIDv8_jS_:
LFB1071:
	vmovdqa	LC10(%rip), %ymm2
	vpmuludq	%ymm1, %ymm0, %ymm3
	vpermd	%ymm1, %ymm2, %ymm1
	vpermd	%ymm0, %ymm2, %ymm0
	vpmuludq	%ymm1, %ymm0, %ymm0
	vmovdqa	LC0(%rip), %ymm1
	vpaddq	%ymm1, %ymm0, %ymm0
	vpaddq	%ymm1, %ymm3, %ymm1
	vpsrlq	$23, %ymm0, %ymm0
	vpsrlq	$23, %ymm1, %ymm1
	vpsllq	$32, %ymm0, %ymm0
	vpor	%ymm1, %ymm0, %ymm0
	ret
LFE1071:
	.align 4,0x90
	.globl __Z3foov
__Z3foov:
LFB1072:
	vmovdqa	LC0(%rip), %ymm4
	leaq	_b(%rip), %rsi
	xorl	%eax, %eax
	leaq	_a(%rip), %rcx
	leaq	_c(%rip), %rdx
	.align 4,0x90
L11:
	vpermq	$216, (%rsi,%rax), %ymm3
	vpermq	$216, (%rcx,%rax), %ymm1
	vpshufd	$80, %ymm3, %ymm2
	vpshufd	$80, %ymm1, %ymm0
	vpshufd	$250, %ymm1, %ymm1
	vpmuludq	%ymm0, %ymm2, %ymm2
	vpshufd	$250, %ymm3, %ymm0
	vpmuludq	%ymm1, %ymm0, %ymm0
	vpaddq	%ymm4, %ymm2, %ymm2
	vpaddq	%ymm4, %ymm0, %ymm0
	vpsrlq	$23, %ymm2, %ymm2
	vpsrlq	$23, %ymm0, %ymm0
	vperm2i128	$32, %ymm0, %ymm2, %ymm1
	vperm2i128	$49, %ymm0, %ymm2, %ymm0
	vpshufd	$216, %ymm1, %ymm1
	vpshufd	$216, %ymm0, %ymm0
	vpunpcklqdq	%ymm0, %ymm1, %ymm0
	vmovdqa	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L11
	vzeroupper
	ret
LFE1072:
	.align 4,0x90
	.globl __Z3redv
__Z3redv:
LFB1073:
	leaq	_b(%rip), %rdx
	movl	$1, %ecx
	leaq	4096+_b(%rip), %rsi
	.align 4,0x90
L15:
	movl	(%rdx), %eax
	addq	$4, %rdx
	imulq	%rcx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	cmpq	%rsi, %rdx
	movl	%eax, %ecx
	jne	L15
	rep; ret
LFE1073:
	.align 4,0x90
	.globl __Z4redVv
__Z4redVv:
LFB1074:
	vmovdqa	LC11(%rip), %xmm0
	leaq	_b(%rip), %rax
	vmovdqa	LC1(%rip), %xmm2
	leaq	4096+_b(%rip), %rdx
	.align 4,0x90
L18:
	vmovd	8(%rax), %xmm4
	addq	$16, %rax
	vmovd	-16(%rax), %xmm5
	vpinsrd	$1, -4(%rax), %xmm4, %xmm1
	vpinsrd	$1, -12(%rax), %xmm5, %xmm3
	vpunpcklqdq	%xmm1, %xmm3, %xmm1
	vpmuludq	%xmm1, %xmm0, %xmm3
	vpshufd	$177, %xmm0, %xmm0
	vpshufd	$177, %xmm1, %xmm1
	vpmuludq	%xmm1, %xmm0, %xmm1
	cmpq	%rdx, %rax
	vpaddq	%xmm2, %xmm3, %xmm0
	vpaddq	%xmm2, %xmm1, %xmm1
	vpsrlq	$23, %xmm0, %xmm0
	vpsrlq	$23, %xmm1, %xmm1
	vpsllq	$32, %xmm1, %xmm1
	vpor	%xmm0, %xmm1, %xmm0
	jne	L18
	vpextrd	$3, %xmm0, %eax
	vpextrd	$2, %xmm0, %edx
	shrl	$4, %eax
	shrl	$4, %edx
	vmovd	%xmm0, %ecx
	imulq	%rdx, %rax
	shrl	$4, %ecx
	vpextrd	$1, %xmm0, %edx
	shrl	$4, %edx
	imulq	%rcx, %rdx
	addq	$4194304, %rax
	salq	$9, %rax
	addq	$4194304, %rdx
	shrq	$36, %rax
	salq	$9, %rdx
	shrq	$36, %rdx
	imulq	%rdx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	ret
LFE1074:
	.align 4,0x90
	.globl __Z6redExpv
__Z6redExpv:
LFB1075:
	pushq	%rbx
LCFI0:
	leaq	_b(%rip), %rax
	movl	$1, %r11d
	leaq	4096+_b(%rip), %rbx
	movl	$1, %r10d
	movl	$1, %r9d
	movl	$1, %r8d
	.align 4,0x90
L21:
	movl	(%rax), %edi
	movl	%r8d, %r8d
	movl	%r9d, %r9d
	movl	4(%rax), %esi
	movl	%r10d, %r10d
	movl	%r11d, %r11d
	movl	8(%rax), %ecx
	addq	$16, %rax
	movl	-4(%rax), %edx
	imulq	%r8, %rdi
	imulq	%r9, %rsi
	imulq	%r10, %rcx
	imulq	%r11, %rdx
	leaq	4194304(%rdi), %r8
	leaq	4194304(%rsi), %r9
	shrq	$23, %r8
	leaq	4194304(%rcx), %r10
	shrq	$23, %r9
	leaq	4194304(%rdx), %r11
	shrq	$23, %r10
	shrq	$23, %r11
	cmpq	%rbx, %rax
	jne	L21
	movl	%r11d, %eax
	movl	%r10d, %r10d
	movl	%r9d, %edx
	imulq	%r10, %rax
	movl	%r8d, %r8d
	popq	%rbx
LCFI1:
	imulq	%r8, %rdx
	addq	$4194304, %rax
	shrq	$23, %rax
	addq	$4194304, %rdx
	movl	%eax, %eax
	shrq	$23, %rdx
	movl	%edx, %edx
	imulq	%rdx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	ret
LFE1075:
	.align 4,0x90
	.globl __Z7redExplv
__Z7redExplv:
LFB1076:
	movl	$1, %esi
	movl	$1, %edi
	movl	$1, %edx
	leaq	_bl(%rip), %rax
	movl	$1, %ecx
	leaq	8192+_bl(%rip), %r8
	.align 4,0x90
L25:
	imulq	(%rax), %rcx
	addq	$32, %rax
	imulq	-24(%rax), %rdx
	imulq	-16(%rax), %rdi
	imulq	-8(%rax), %rsi
	addq	$4194304, %rcx
	addq	$4194304, %rdx
	shrq	$23, %rcx
	addq	$4194304, %rdi
	shrq	$23, %rdx
	addq	$4194304, %rsi
	shrq	$23, %rdi
	shrq	$23, %rsi
	cmpq	%r8, %rax
	jne	L25
	imulq	%rdi, %rsi
	movq	%rcx, %rax
	imulq	%rdx, %rax
	leaq	4194304(%rsi), %rdx
	addq	$4194304, %rax
	shrq	$23, %rdx
	shrq	$23, %rax
	imulq	%rdx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	ret
LFE1076:
	.align 4,0x90
	.globl __Z4redLv
__Z4redLv:
LFB1077:
	leaq	_b(%rip), %rdx
	movl	$1, %eax
	leaq	4096+_b(%rip), %rsi
	.align 4,0x90
L28:
	movl	(%rdx), %ecx
	addq	$4, %rdx
	imulq	%rcx, %rax
	addq	$4194304, %rax
	shrq	$23, %rax
	cmpq	%rsi, %rdx
	jne	L28
	rep; ret
LFE1077:
	.align 4,0x90
	.globl __Z4prodv
__Z4prodv:
LFB1078:
	leaq	_b(%rip), %rdx
	movl	$1, %eax
	leaq	4096+_b(%rip), %rcx
	.align 4,0x90
L31:
	imull	(%rdx), %eax
	addq	$4, %rdx
	addl	$1024, %eax
	cmpq	%rcx, %rdx
	jne	L31
	rep; ret
LFE1078:
	.align 4,0x90
	.globl __Z5prod0v
__Z5prod0v:
LFB1079:
	leaq	_b(%rip), %rdx
	movl	$1, %eax
	leaq	4096+_b(%rip), %rcx
	.align 4,0x90
L34:
	imull	(%rdx), %eax
	addq	$4, %rdx
	addl	$1024, %eax
	cmpq	%rcx, %rdx
	jne	L34
	rep; ret
LFE1079:
	.align 4,0x90
	.globl __Z5prodLv
__Z5prodLv:
LFB1080:
	leaq	_bl(%rip), %rdx
	movl	$1, %eax
	leaq	8192+_bl(%rip), %rcx
	.align 4,0x90
L37:
	imulq	(%rdx), %rax
	addq	$8, %rdx
	addq	$1024, %rax
	cmpq	%rcx, %rdx
	jne	L37
	rep; ret
LFE1080:
	.align 4,0x90
	.globl __Z6prodL2v
__Z6prodL2v:
LFB1081:
	leaq	_bl(%rip), %rdx
	movl	$1, %eax
	leaq	8192+_bl(%rip), %rcx
	.align 4,0x90
L40:
	imulq	(%rdx), %rax
	addq	$8, %rdx
	addq	$1024, %rax
	cmpq	%rcx, %rdx
	jne	L40
	rep; ret
LFE1081:
	.globl _cl
	.zerofill __DATA,__pu_bss5,_cl,8192,5
	.globl _bl
	.zerofill __DATA,__pu_bss5,_bl,8192,5
	.globl _al
	.zerofill __DATA,__pu_bss5,_al,8192,5
	.globl _c
	.zerofill __DATA,__pu_bss5,_c,4096,5
	.globl _b
	.zerofill __DATA,__pu_bss5,_b,4096,5
	.globl _a
	.zerofill __DATA,__pu_bss5,_a,4096,5
	.const
	.align 5
LC0:
	.quad	4194304
	.quad	4194304
	.quad	4194304
	.quad	4194304
	.literal16
	.align 4
LC1:
	.quad	4194304
	.quad	4194304
	.align 4
LC2:
	.long	255
	.long	255
	.long	255
	.long	255
	.align 4
LC3:
	.long	127
	.long	127
	.long	127
	.long	127
	.align 4
LC4:
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.align 4
LC5:
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.const
	.align 5
LC6:
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.align 5
LC7:
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.align 5
LC8:
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.align 5
LC9:
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.align 5
LC10:
	.long	1
	.long	0
	.long	3
	.long	2
	.long	5
	.long	4
	.long	0
	.long	7
	.literal16
	.align 4
LC11:
	.long	1
	.long	1
	.long	1
	.long	1
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
	.quad	LFB1063-.
	.set L$set$2,LFE1063-LFB1063
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1064-.
	.set L$set$4,LFE1064-LFB1064
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB1065-.
	.set L$set$6,LFE1065-LFB1065
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB1066-.
	.set L$set$8,LFE1066-LFB1066
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB1067-.
	.set L$set$10,LFE1067-LFB1067
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB1068-.
	.set L$set$12,LFE1068-LFB1068
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB1069-.
	.set L$set$14,LFE1069-LFB1069
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB1070-.
	.set L$set$16,LFE1070-LFB1070
	.quad L$set$16
	.byte	0
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$17,LEFDE17-LASFDE17
	.long L$set$17
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB1071-.
	.set L$set$18,LFE1071-LFB1071
	.quad L$set$18
	.byte	0
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$19,LEFDE19-LASFDE19
	.long L$set$19
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB1072-.
	.set L$set$20,LFE1072-LFB1072
	.quad L$set$20
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$21,LEFDE21-LASFDE21
	.long L$set$21
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB1073-.
	.set L$set$22,LFE1073-LFB1073
	.quad L$set$22
	.byte	0
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$23,LEFDE23-LASFDE23
	.long L$set$23
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB1074-.
	.set L$set$24,LFE1074-LFB1074
	.quad L$set$24
	.byte	0
	.align 3
LEFDE23:
LSFDE25:
	.set L$set$25,LEFDE25-LASFDE25
	.long L$set$25
LASFDE25:
	.long	LASFDE25-EH_frame1
	.quad	LFB1075-.
	.set L$set$26,LFE1075-LFB1075
	.quad L$set$26
	.byte	0
	.byte	0x4
	.set L$set$27,LCFI0-LFB1075
	.long L$set$27
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$28,LCFI1-LCFI0
	.long L$set$28
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE25:
LSFDE27:
	.set L$set$29,LEFDE27-LASFDE27
	.long L$set$29
LASFDE27:
	.long	LASFDE27-EH_frame1
	.quad	LFB1076-.
	.set L$set$30,LFE1076-LFB1076
	.quad L$set$30
	.byte	0
	.align 3
LEFDE27:
LSFDE29:
	.set L$set$31,LEFDE29-LASFDE29
	.long L$set$31
LASFDE29:
	.long	LASFDE29-EH_frame1
	.quad	LFB1077-.
	.set L$set$32,LFE1077-LFB1077
	.quad L$set$32
	.byte	0
	.align 3
LEFDE29:
LSFDE31:
	.set L$set$33,LEFDE31-LASFDE31
	.long L$set$33
LASFDE31:
	.long	LASFDE31-EH_frame1
	.quad	LFB1078-.
	.set L$set$34,LFE1078-LFB1078
	.quad L$set$34
	.byte	0
	.align 3
LEFDE31:
LSFDE33:
	.set L$set$35,LEFDE33-LASFDE33
	.long L$set$35
LASFDE33:
	.long	LASFDE33-EH_frame1
	.quad	LFB1079-.
	.set L$set$36,LFE1079-LFB1079
	.quad L$set$36
	.byte	0
	.align 3
LEFDE33:
LSFDE35:
	.set L$set$37,LEFDE35-LASFDE35
	.long L$set$37
LASFDE35:
	.long	LASFDE35-EH_frame1
	.quad	LFB1080-.
	.set L$set$38,LFE1080-LFB1080
	.quad L$set$38
	.byte	0
	.align 3
LEFDE35:
LSFDE37:
	.set L$set$39,LEFDE37-LASFDE37
	.long L$set$39
LASFDE37:
	.long	LASFDE37-EH_frame1
	.quad	LFB1081-.
	.set L$set$40,LFE1081-LFB1081
	.quad L$set$40
	.byte	0
	.align 3
LEFDE37:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
