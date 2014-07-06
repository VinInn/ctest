	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
	.align 1
LCOLDB0:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTB0:
	.align 1
	.align 4
	.globl __ZNKSt5ctypeIcE8do_widenEc
	.weak_definition __ZNKSt5ctypeIcE8do_widenEc
__ZNKSt5ctypeIcE8do_widenEc:
LFB5190:
	movl	%esi, %eax
	ret
LFE5190:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDE0:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB25:
	.text
LHOTB25:
	.align 4,0x90
	.globl __Z6tanhP4f
__Z6tanhP4f:
LFB5440:
	vminss	LC1(%rip), %xmm0, %xmm0
	vmovss	LC2(%rip), %xmm1
	vcomiss	%xmm0, %xmm1
	jbe	L13
	vmovss	LC3(%rip), %xmm1
	vcomiss	%xmm0, %xmm1
	ja	L18
	vmovss	LC9(%rip), %xmm1
	vfnmadd213ss	LC10(%rip), %xmm0, %xmm1
	vfmadd213ss	LC11(%rip), %xmm0, %xmm1
	vfmadd213ss	LC12(%rip), %xmm0, %xmm1
	vfmadd213ss	LC13(%rip), %xmm1, %xmm0
	ret
	.align 4,0x90
L13:
	vmovss	LC14(%rip), %xmm1
	vcomiss	%xmm0, %xmm1
	jbe	L15
	vmovss	LC15(%rip), %xmm1
	vfmadd213ss	LC16(%rip), %xmm0, %xmm1
	vfmadd213ss	LC17(%rip), %xmm0, %xmm1
	vfmadd213ss	LC18(%rip), %xmm0, %xmm1
	vfmadd213ss	LC19(%rip), %xmm0, %xmm1
	vfmadd213ss	LC20(%rip), %xmm1, %xmm0
	ret
	.align 4,0x90
L15:
	vmovss	LC21(%rip), %xmm1
	vfmadd213ss	LC22(%rip), %xmm0, %xmm1
	vfmadd213ss	LC23(%rip), %xmm0, %xmm1
	vfmadd213ss	LC24(%rip), %xmm0, %xmm1
	vfmadd213ss	LC3(%rip), %xmm1, %xmm0
	ret
	.align 4,0x90
L18:
	vmovss	LC4(%rip), %xmm1
	vfmadd213ss	LC5(%rip), %xmm0, %xmm1
	vfmadd213ss	LC6(%rip), %xmm0, %xmm1
	vfmadd213ss	LC7(%rip), %xmm0, %xmm1
	vfmadd213ss	LC8(%rip), %xmm0, %xmm1
	vmulss	%xmm0, %xmm1, %xmm0
	ret
LFE5440:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE25:
	.text
LHOTE25:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB39:
	.text
LHOTB39:
	.align 4,0x90
	.globl __Z5logE4f
__Z5logE4f:
LFB5442:
	vmovss	LC26(%rip), %xmm1
	vcomiss	%xmm0, %xmm1
	ja	L25
	vxorpd	%xmm2, %xmm2, %xmm2
	vcvtss2sd	%xmm0, %xmm2, %xmm2
	vmovsd	LC33(%rip), %xmm0
	vfmadd213sd	LC34(%rip), %xmm2, %xmm0
	vfmadd213sd	LC35(%rip), %xmm2, %xmm0
	vfmadd213sd	LC36(%rip), %xmm2, %xmm0
	vfmadd213sd	LC37(%rip), %xmm2, %xmm0
	vmovapd	%xmm0, %xmm1
	vfmadd213sd	LC38(%rip), %xmm2, %xmm1
	vmulsd	%xmm2, %xmm1, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	ret
	.align 4,0x90
L25:
	vmovsd	LC27(%rip), %xmm1
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vfmadd213sd	LC28(%rip), %xmm0, %xmm1
	vfmadd213sd	LC29(%rip), %xmm0, %xmm1
	vfmadd213sd	LC30(%rip), %xmm0, %xmm1
	vfmadd213sd	LC31(%rip), %xmm0, %xmm1
	vfmadd213sd	LC32(%rip), %xmm1, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	ret
LFE5442:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE39:
	.text
LHOTE39:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDB48:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTB48:
	.align 4
	.globl __Z11approx_logfILi4EEff
	.weak_definition __Z11approx_logfILi4EEff
__Z11approx_logfILi4EEff:
LFB5583:
	vcomiss	LC42(%rip), %xmm0
	vmovss	LC40(%rip), %xmm1
	jae	L27
	vmovd	%xmm0, %ecx
	vmovd	%xmm0, %eax
	vmovd	%xmm0, %edx
	vxorps	%xmm2, %xmm2, %xmm2
	sarl	$22, %ecx
	andl	$8388607, %eax
	sarl	$23, %edx
	vmovss	LC43(%rip), %xmm1
	andl	$1, %ecx
	orl	$1065353216, %eax
	movzbl	%dl, %edx
	movl	%ecx, %esi
	sall	$23, %esi
	subl	%esi, %eax
	vmovd	%eax, %xmm4
	vsubss	LC3(%rip), %xmm4, %xmm3
	vfmadd213ss	LC44(%rip), %xmm3, %xmm1
	vfmadd213ss	LC45(%rip), %xmm3, %xmm1
	vfmadd213ss	LC46(%rip), %xmm3, %xmm1
	leal	-127(%rcx,%rdx), %eax
	vcvtsi2ss	%eax, %xmm2, %xmm2
	vmulss	%xmm3, %xmm1, %xmm1
	vfmadd231ss	LC26(%rip), %xmm2, %xmm1
L27:
	vxorps	%xmm2, %xmm2, %xmm2
	vcomiss	%xmm0, %xmm2
	jae	L30
	vmovaps	%xmm1, %xmm0
	ret
	.align 4
L30:
	vmovss	LC41(%rip), %xmm0
	ret
LFE5583:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDE48:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTE48:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB50:
	.text
LHOTB50:
	.align 4,0x90
	.globl __Z4logEf
__Z4logEf:
LFB5441:
	subq	$8, %rsp
LCFI0:
	vmovss	LC49(%rip), %xmm1
	vxorps	%xmm1, %xmm0, %xmm0
	call	_expf
	vmovss	LC3(%rip), %xmm1
	addq	$8, %rsp
LCFI1:
	vsubss	%xmm0, %xmm1, %xmm0
	jmp	_logf
LFE5441:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE50:
	.text
LHOTE50:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDB60:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTB60:
	.align 4
	.globl __Z11approx_expfILi3EEff
	.weak_definition __Z11approx_expfILi3EEff
__Z11approx_expfILi3EEff:
LFB5582:
	vmaxss	LC51(%rip), %xmm0, %xmm0
	vmovss	LC53(%rip), %xmm2
	vminss	LC52(%rip), %xmm0, %xmm0
	vfmadd213ss	LC54(%rip), %xmm0, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vfnmadd231ss	LC55(%rip), %xmm2, %xmm0
	vmovss	LC57(%rip), %xmm1
	vcvttss2si	%xmm2, %eax
	vfnmadd231ss	LC56(%rip), %xmm2, %xmm0
	vfmadd213ss	LC58(%rip), %xmm0, %xmm1
	vfmadd213ss	LC59(%rip), %xmm0, %xmm1
	vfmadd213ss	LC2(%rip), %xmm0, %xmm1
	addl	$126, %eax
	sall	$23, %eax
	vmovd	%eax, %xmm0
	vmulss	%xmm0, %xmm1, %xmm0
	ret
LFE5582:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDE60:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTE60:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB69:
	.text
LHOTB69:
	.align 4,0x90
	.globl __Z4erf4f
__Z4erf4f:
LFB5444:
	vmovss	LC61(%rip), %xmm4
	vmovaps	%xmm0, %xmm2
	vmovss	LC64(%rip), %xmm3
	vmovss	LC49(%rip), %xmm6
	vandps	%xmm4, %xmm2, %xmm2
	vmovaps	%xmm0, %xmm7
	vminss	LC62(%rip), %xmm2, %xmm2
	vmovss	LC3(%rip), %xmm5
	vandps	%xmm6, %xmm7, %xmm7
	vmulss	%xmm2, %xmm2, %xmm2
	vmovaps	%xmm2, %xmm1
	vxorps	%xmm6, %xmm2, %xmm2
	vfmadd132ss	LC63(%rip), %xmm5, %xmm1
	vdivss	%xmm1, %xmm3, %xmm3
	vmovss	LC53(%rip), %xmm1
	vaddss	%xmm5, %xmm3, %xmm3
	vmulss	%xmm2, %xmm3, %xmm2
	vfmadd213ss	LC54(%rip), %xmm2, %xmm1
	vroundss	$1, %xmm1, %xmm1, %xmm1
	vcvttss2si	%xmm1, %eax
	vfnmadd231ss	LC55(%rip), %xmm1, %xmm2
	vfnmadd231ss	LC56(%rip), %xmm1, %xmm2
	vmovss	LC65(%rip), %xmm3
	vfmadd213ss	LC66(%rip), %xmm2, %xmm3
	vfmadd213ss	LC67(%rip), %xmm2, %xmm3
	vfmadd213ss	LC68(%rip), %xmm2, %xmm3
	vfmadd213ss	LC2(%rip), %xmm3, %xmm2
	addl	$126, %eax
	sall	$23, %eax
	vmovd	%eax, %xmm1
	vfnmadd132ss	%xmm2, %xmm5, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm1
	vandnps	%xmm1, %xmm6, %xmm6
	vorps	%xmm7, %xmm6, %xmm6
	vmovaps	%xmm6, %xmm0
	ret
LFE5444:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE69:
	.text
LHOTE69:
	.cstring
	.align 3
LC70:
	.ascii "launching  exhaustive test for \0"
LC71:
	.ascii "limits \0"
LC72:
	.ascii " \0"
	.align 3
LC73:
	.ascii "absdiff / reldeff/ maxdiff / diff >127 / diff >16393 :  \0"
LC74:
	.ascii " / \0"
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDB75:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTB75:
	.align 4
	.globl __Z7accTestIPFddEPFffEEvT_T0_ff
	.weak_definition __Z7accTestIPFddEPFffEEvT_T0_ff
__Z7accTestIPFddEPFffEEvT_T0_ff:
LFB5580:
	pushq	%r15
LCFI2:
	pushq	%r14
LCFI3:
	pushq	%r13
LCFI4:
	pushq	%r12
LCFI5:
	pushq	%rbp
LCFI6:
	pushq	%rbx
LCFI7:
	subq	$24, %rsp
LCFI8:
	movq	__ZSt4cout@GOTPCREL(%rip), %rax
	movq	(%rax), %rdx
	movq	-24(%rdx), %rdx
	movq	240(%rdx,%rax), %rbx
	testq	%rbx, %rbx
	je	L42
	cmpb	$0, 56(%rbx)
	movq	%rdi, %r14
	movq	%rsi, %r15
	vmovd	%xmm0, %r12d
	vmovd	%xmm1, %ebp
	je	L38
	movsbl	67(%rbx), %esi
L39:
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	leaq	LC70(%rip), %rsi
	movl	$31, %edx
	movq	%rax, %rdi
	movq	%rax, %rbx
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	__ZTSPFffE@GOTPCREL(%rip), %rsi
	testq	%rsi, %rsi
	je	L66
	movl	$5, %edx
	movq	%rbx, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
L41:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	movq	240(%rbx,%rax), %r13
	testq	%r13, %r13
	je	L42
	cmpb	$0, 56(%r13)
	je	L43
	movsbl	67(%r13), %esi
L44:
	movq	%rbx, %rdi
	subl	$1, %ebp
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	leal	1(%r12), %eax
	movl	%ebp, %r12d
	movq	__ZSt4cout@GOTPCREL(%rip), %r13
	movl	$7, %edx
	movl	%eax, %ebx
	subl	%eax, %r12d
	leaq	LC71(%rip), %rsi
	movq	%r13, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	vmovd	%ebx, %xmm4
	movq	%r13, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm4, %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$1, %edx
	leaq	LC72(%rip), %rsi
	movq	%rax, %r13
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	vmovd	%ebp, %xmm5
	movq	%r13, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm5, %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$1, %edx
	leaq	LC72(%rip), %rsi
	movq	%rax, %r13
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%r12d, %esi
	movq	%r13, %rdi
	call	__ZNSo9_M_insertImEERSoT_
	movq	%rax, %r13
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r13,%rax), %r12
	testq	%r12, %r12
	je	L42
	cmpb	$0, 56(%r12)
	je	L45
	movsbl	67(%r12), %esi
L46:
	movq	%r13, %rdi
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	vxorps	%xmm3, %xmm3, %xmm3
	movl	$0, 12(%rsp)
	vmovss	%xmm3, (%rsp)
	vmovss	%xmm3, 4(%rsp)
	.align 4
L47:
	cmpl	%ebx, %ebp
	jbe	L67
L49:
	addl	$1, %ebx
	vmovd	%ebx, %xmm0
	call	*%r15
	vmovd	%ebx, %xmm3
	vmovss	%xmm0, 8(%rsp)
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm3, %xmm0, %xmm0
	call	*%r14
	vmovss	8(%rsp), %xmm1
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovd	%xmm0, %ecx
	vsubss	%xmm0, %xmm1, %xmm2
	vmovd	%xmm1, %eax
	vandps	LC61(%rip), %xmm2, %xmm2
	vmaxss	4(%rsp), %xmm2, %xmm4
	vdivss	%xmm0, %xmm1, %xmm2
	subl	%ecx, %eax
	cltd
	xorl	%edx, %eax
	subl	%edx, %eax
	vmovss	%xmm4, 4(%rsp)
	cmpl	%eax, %r12d
	cmovl	%eax, %r12d
	vsubss	LC3(%rip), %xmm2, %xmm2
	vandps	LC61(%rip), %xmm2, %xmm2
	vmaxss	(%rsp), %xmm2, %xmm6
	vmovss	%xmm6, (%rsp)
	cmpl	$127, %eax
	jle	L47
	addl	$1, %r13d
	cmpl	$16393, %eax
	jle	L47
	addl	$1, 12(%rsp)
	cmpl	%ebx, %ebp
	ja	L49
L67:
	movq	__ZSt4cout@GOTPCREL(%rip), %rbx
	movl	$56, %edx
	leaq	LC73(%rip), %rsi
	movq	%rbx, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	4(%rsp), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rbx
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	(%rsp), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rbx
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	%r12d, %esi
	call	__ZNSolsEi
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rbx
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	%r13d, %esi
	call	__ZNSolsEi
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rdi
	movq	%rax, %rbx
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	12(%rsp), %esi
	movq	%rbx, %rdi
	call	__ZNSolsEi
	movq	%rax, %rbp
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rbp,%rax), %rbx
	testq	%rbx, %rbx
	je	L42
	cmpb	$0, 56(%rbx)
	je	L50
	movsbl	67(%rbx), %esi
L51:
	movq	%rbp, %rdi
	call	__ZNSo3putEc
	addq	$24, %rsp
LCFI9:
	popq	%rbx
LCFI10:
	movq	%rax, %rdi
	popq	%rbp
LCFI11:
	popq	%r12
LCFI12:
	popq	%r13
LCFI13:
	popq	%r14
LCFI14:
	popq	%r15
LCFI15:
	jmp	__ZNSo5flushEv
	.align 4
L38:
LCFI16:
	movq	%rbx, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L39
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L39
	.align 4
L50:
	movq	%rbx, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L51
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L51
	.align 4
L45:
	movq	%r12, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L46
	movq	%r12, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L46
	.align 4
L43:
	movq	%r13, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	0(%r13), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L44
	movq	%r13, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L44
L66:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	addq	-24(%rax), %rdi
	movl	32(%rdi), %esi
	orl	$1, %esi
	call	__ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	L41
L42:
	call	__ZSt16__throw_bad_castv
LFE5580:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDE75:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTE75:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDB76:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTB76:
	.align 4
	.globl __Z7accTestIPFffES1_EvT_T0_ff
	.weak_definition __Z7accTestIPFffES1_EvT_T0_ff
__Z7accTestIPFffES1_EvT_T0_ff:
LFB5581:
	pushq	%r15
LCFI17:
	pushq	%r14
LCFI18:
	pushq	%r13
LCFI19:
	pushq	%r12
LCFI20:
	pushq	%rbp
LCFI21:
	pushq	%rbx
LCFI22:
	subq	$24, %rsp
LCFI23:
	movq	__ZSt4cout@GOTPCREL(%rip), %rax
	movq	(%rax), %rdx
	movq	-24(%rdx), %rdx
	movq	240(%rdx,%rax), %rbx
	testq	%rbx, %rbx
	je	L74
	cmpb	$0, 56(%rbx)
	movq	%rdi, %r14
	movq	%rsi, %r15
	vmovd	%xmm0, %r12d
	vmovd	%xmm1, %ebp
	je	L70
	movsbl	67(%rbx), %esi
L71:
	movq	__ZSt4cout@GOTPCREL(%rip), %rdi
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	leaq	LC70(%rip), %rsi
	movl	$31, %edx
	movq	%rax, %rdi
	movq	%rax, %rbx
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	__ZTSPFffE@GOTPCREL(%rip), %rsi
	testq	%rsi, %rsi
	je	L98
	movl	$5, %edx
	movq	%rbx, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
L73:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	movq	240(%rbx,%rax), %r13
	testq	%r13, %r13
	je	L74
	cmpb	$0, 56(%r13)
	je	L75
	movsbl	67(%r13), %esi
L76:
	movq	%rbx, %rdi
	subl	$1, %ebp
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	leal	1(%r12), %eax
	movl	%ebp, %r12d
	movq	__ZSt4cout@GOTPCREL(%rip), %r13
	movl	$7, %edx
	movl	%eax, %ebx
	subl	%eax, %r12d
	leaq	LC71(%rip), %rsi
	movq	%r13, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	vmovd	%ebx, %xmm7
	movq	%r13, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm7, %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$1, %edx
	leaq	LC72(%rip), %rsi
	movq	%rax, %r13
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	vmovd	%ebp, %xmm7
	movq	%r13, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm7, %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$1, %edx
	leaq	LC72(%rip), %rsi
	movq	%rax, %r13
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	%r12d, %esi
	movq	%r13, %rdi
	call	__ZNSo9_M_insertImEERSoT_
	movq	%rax, %r13
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r13,%rax), %r12
	testq	%r12, %r12
	je	L74
	cmpb	$0, 56(%r12)
	je	L77
	movsbl	67(%r12), %esi
L78:
	movq	%r13, %rdi
	xorl	%r12d, %r12d
	xorl	%r13d, %r13d
	call	__ZNSo3putEc
	movq	%rax, %rdi
	call	__ZNSo5flushEv
	vxorps	%xmm7, %xmm7, %xmm7
	movl	$0, 12(%rsp)
	vmovss	%xmm7, (%rsp)
	vmovss	%xmm7, 4(%rsp)
	.align 4
L79:
	cmpl	%ebx, %ebp
	jbe	L99
L81:
	addl	$1, %ebx
	vmovd	%ebx, %xmm0
	call	*%r15
	vmovss	%xmm0, 8(%rsp)
	vmovd	%ebx, %xmm0
	call	*%r14
	vmovss	8(%rsp), %xmm1
	vmovd	%xmm0, %ecx
	vsubss	%xmm0, %xmm1, %xmm2
	vmovd	%xmm1, %eax
	vandps	LC61(%rip), %xmm2, %xmm2
	vmaxss	4(%rsp), %xmm2, %xmm3
	vdivss	%xmm0, %xmm1, %xmm2
	subl	%ecx, %eax
	cltd
	xorl	%edx, %eax
	subl	%edx, %eax
	vmovss	%xmm3, 4(%rsp)
	cmpl	%eax, %r12d
	cmovl	%eax, %r12d
	vsubss	LC3(%rip), %xmm2, %xmm2
	vandps	LC61(%rip), %xmm2, %xmm2
	vmaxss	(%rsp), %xmm2, %xmm5
	vmovss	%xmm5, (%rsp)
	cmpl	$127, %eax
	jle	L79
	addl	$1, %r13d
	cmpl	$16393, %eax
	jle	L79
	addl	$1, 12(%rsp)
	cmpl	%ebx, %ebp
	ja	L81
L99:
	movq	__ZSt4cout@GOTPCREL(%rip), %rbx
	movl	$56, %edx
	leaq	LC73(%rip), %rsi
	movq	%rbx, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	4(%rsp), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rbx
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	vxorpd	%xmm0, %xmm0, %xmm0
	vcvtss2sd	(%rsp), %xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rbx
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	%r12d, %esi
	call	__ZNSolsEi
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rbx
	movq	%rax, %rdi
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	%r13d, %esi
	call	__ZNSolsEi
	movl	$3, %edx
	leaq	LC74(%rip), %rsi
	movq	%rax, %rdi
	movq	%rax, %rbx
	call	__ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	12(%rsp), %esi
	movq	%rbx, %rdi
	call	__ZNSolsEi
	movq	%rax, %rbp
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%rbp,%rax), %rbx
	testq	%rbx, %rbx
	je	L74
	cmpb	$0, 56(%rbx)
	je	L82
	movsbl	67(%rbx), %esi
L83:
	movq	%rbp, %rdi
	call	__ZNSo3putEc
	addq	$24, %rsp
LCFI24:
	popq	%rbx
LCFI25:
	movq	%rax, %rdi
	popq	%rbp
LCFI26:
	popq	%r12
LCFI27:
	popq	%r13
LCFI28:
	popq	%r14
LCFI29:
	popq	%r15
LCFI30:
	jmp	__ZNSo5flushEv
	.align 4
L70:
LCFI31:
	movq	%rbx, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L71
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L71
	.align 4
L82:
	movq	%rbx, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L83
	movq	%rbx, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L83
	.align 4
L77:
	movq	%r12, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L78
	movq	%r12, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L78
	.align 4
L75:
	movq	%r13, %rdi
	call	__ZNKSt5ctypeIcE13_M_widen_initEv
	movq	0(%r13), %rax
	movl	$10, %esi
	movq	48(%rax), %rax
	cmpq	__ZNKSt5ctypeIcE8do_widenEc@GOTPCREL(%rip), %rax
	je	L76
	movq	%r13, %rdi
	call	*%rax
	movsbl	%al, %esi
	jmp	L76
L98:
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	addq	-24(%rax), %rdi
	movl	32(%rdi), %esi
	orl	$1, %esi
	call	__ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	L73
L74:
	call	__ZSt16__throw_bad_castv
LFE5581:
	.section __TEXT,__text_cold_coal,coalesced,pure_instructions
LCOLDE76:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
LHOTE76:
	.cstring
LC78:
	.ascii "%a\12\0"
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB83:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB83:
	.align 4
	.globl _main
_main:
LFB5445:
	subq	$8, %rsp
LCFI32:
	movl	$1, %eax
	vmovsd	LC77(%rip), %xmm0
	leaq	LC78(%rip), %rdi
	call	_printf
	movq	_erf@GOTPCREL(%rip), %rdi
	leaq	__Z4erf4f(%rip), %rsi
	vmovss	LC1(%rip), %xmm1
	vmovss	LC79(%rip), %xmm0
	call	__Z7accTestIPFddEPFffEEvT_T0_ff
	leaq	__Z5logE4f(%rip), %rsi
	vmovss	LC2(%rip), %xmm1
	vmovss	LC26(%rip), %xmm0
	leaq	__Z4logEf(%rip), %rdi
	call	__Z7accTestIPFffES1_EvT_T0_ff
	movq	_tanh@GOTPCREL(%rip), %rdi
	leaq	__Z6tanhP4f(%rip), %rsi
	vmovss	LC1(%rip), %xmm1
	vmovss	LC80(%rip), %xmm0
	call	__Z7accTestIPFddEPFffEEvT_T0_ff
	movq	__Z11approx_expfILi3EEff@GOTPCREL(%rip), %rsi
	movq	_exp@GOTPCREL(%rip), %rdi
	vmovss	LC81(%rip), %xmm1
	vmovss	LC82(%rip), %xmm0
	call	__Z7accTestIPFddEPFffEEvT_T0_ff
	movq	__Z11approx_logfILi4EEff@GOTPCREL(%rip), %rsi
	movq	_log@GOTPCREL(%rip), %rdi
	vmovss	LC42(%rip), %xmm1
	vmovss	LC82(%rip), %xmm0
	call	__Z7accTestIPFddEPFffEEvT_T0_ff
	xorl	%eax, %eax
	addq	$8, %rsp
LCFI33:
	ret
LFE5445:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE83:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE83:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB84:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB84:
	.align 4
__GLOBAL__sub_I_accTest.cpp:
LFB5838:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI34:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	leaq	___dso_handle(%rip), %rdx
	addq	$8, %rsp
LCFI35:
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE5838:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE84:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE84:
	.globl __ZTSPFffE
	.weak_definition __ZTSPFffE
	.section __TEXT,__const_coal,coalesced
__ZTSPFffE:
	.ascii "PFffE\0"
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal4
	.align 2
LC1:
	.long	1090519040
	.align 2
LC2:
	.long	1073741824
	.align 2
LC3:
	.long	1065353216
	.align 2
LC4:
	.long	1009497230
	.align 2
LC5:
	.long	1040100092
	.align 2
LC6:
	.long	3200378104
	.align 2
LC7:
	.long	1002080020
	.align 2
LC8:
	.long	1065351168
	.align 2
LC9:
	.long	1016809278
	.align 2
LC10:
	.long	1045249816
	.align 2
LC11:
	.long	3209660938
	.align 2
LC12:
	.long	1069680276
	.align 2
LC13:
	.long	3187671040
	.align 2
LC14:
	.long	1082130432
	.align 2
LC15:
	.long	985119268
	.align 2
LC16:
	.long	3167420032
	.align 2
LC17:
	.long	1043714876
	.align 2
LC18:
	.long	3206931966
	.align 2
LC19:
	.long	1067135000
	.align 2
LC20:
	.long	1031798784
	.align 2
LC21:
	.long	924927328
	.align 2
LC22:
	.long	3108626152
	.align 2
LC23:
	.long	984004222
	.align 2
LC24:
	.long	3140851448
	.align 2
LC26:
	.long	1060205080
	.literal8
	.align 3
LC27:
	.long	0
	.long	-1069616885
	.align 3
LC28:
	.long	0
	.long	1078855531
	.align 3
LC29:
	.long	1073741824
	.long	-1069148432
	.align 3
LC30:
	.long	3221225472
	.long	1076902228
	.align 3
LC31:
	.long	2147483648
	.long	-1072919586
	.align 3
LC32:
	.long	0
	.long	-1082130432
	.align 3
LC33:
	.long	0
	.long	1072248700
	.align 3
LC34:
	.long	3221225472
	.long	-1072178253
	.align 3
LC35:
	.long	0
	.long	1077027213
	.align 3
LC36:
	.long	0
	.long	-1069807100
	.align 3
LC37:
	.long	2147483648
	.long	1077321852
	.align 3
LC38:
	.long	0
	.long	-1071644672
	.literal4
	.align 2
LC40:
	.long	2139095040
	.align 2
LC41:
	.long	2143289344
	.align 2
LC42:
	.long	2139095039
	.align 2
LC43:
	.long	3191480302
	.align 2
LC44:
	.long	1051741632
	.align 2
LC45:
	.long	3204529758
	.align 2
LC46:
	.long	1065350586
	.literal16
	.align 4
LC49:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.literal4
	.align 2
LC51:
	.long	3266227280
	.align 2
LC52:
	.long	1118925336
	.align 2
LC53:
	.long	1069066811
	.align 2
LC54:
	.long	1056964608
	.align 2
LC55:
	.long	1060205056
	.align 2
LC56:
	.long	901758606
	.align 2
LC57:
	.long	1051476100
	.align 2
LC58:
	.long	1065423432
	.align 2
LC59:
	.long	1073740748
	.literal16
	.align 4
LC61:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.literal4
	.align 2
LC62:
	.long	1084227584
	.align 2
LC63:
	.long	1041663787
	.align 2
LC64:
	.long	1049357818
	.align 2
LC65:
	.long	1034664616
	.align 2
LC66:
	.long	1051456386
	.align 2
LC67:
	.long	1065352836
	.align 2
LC68:
	.long	1073741192
	.literal8
	.align 3
LC77:
	.long	0
	.long	1072692992
	.literal4
	.align 2
LC79:
	.long	1008981770
	.align 2
LC80:
	.long	119291904
	.align 2
LC81:
	.long	1117782016
	.align 2
LC82:
	.long	8388608
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
	.quad	LFB5190-.
	.set L$set$2,LFE5190-LFB5190
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB5440-.
	.set L$set$4,LFE5440-LFB5440
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB5442-.
	.set L$set$6,LFE5442-LFB5442
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB5583-.
	.set L$set$8,LFE5583-LFB5583
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB5441-.
	.set L$set$10,LFE5441-LFB5441
	.quad L$set$10
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI0-LFB5441
	.long L$set$11
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$12,LCFI1-LCFI0
	.long L$set$12
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$13,LEFDE11-LASFDE11
	.long L$set$13
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB5582-.
	.set L$set$14,LFE5582-LFB5582
	.quad L$set$14
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$15,LEFDE13-LASFDE13
	.long L$set$15
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB5444-.
	.set L$set$16,LFE5444-LFB5444
	.quad L$set$16
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$17,LEFDE15-LASFDE15
	.long L$set$17
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB5580-.
	.set L$set$18,LFE5580-LFB5580
	.quad L$set$18
	.byte	0
	.byte	0x4
	.set L$set$19,LCFI2-LFB5580
	.long L$set$19
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$20,LCFI3-LCFI2
	.long L$set$20
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$21,LCFI4-LCFI3
	.long L$set$21
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$22,LCFI5-LCFI4
	.long L$set$22
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$23,LCFI6-LCFI5
	.long L$set$23
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$24,LCFI7-LCFI6
	.long L$set$24
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$25,LCFI8-LCFI7
	.long L$set$25
	.byte	0xe
	.byte	0x50
	.byte	0x4
	.set L$set$26,LCFI9-LCFI8
	.long L$set$26
	.byte	0xa
	.byte	0xe
	.byte	0x38
	.byte	0x4
	.set L$set$27,LCFI10-LCFI9
	.long L$set$27
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$28,LCFI11-LCFI10
	.long L$set$28
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$29,LCFI12-LCFI11
	.long L$set$29
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$30,LCFI13-LCFI12
	.long L$set$30
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$31,LCFI14-LCFI13
	.long L$set$31
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$32,LCFI15-LCFI14
	.long L$set$32
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$33,LCFI16-LCFI15
	.long L$set$33
	.byte	0xb
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$34,LEFDE17-LASFDE17
	.long L$set$34
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB5581-.
	.set L$set$35,LFE5581-LFB5581
	.quad L$set$35
	.byte	0
	.byte	0x4
	.set L$set$36,LCFI17-LFB5581
	.long L$set$36
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$37,LCFI18-LCFI17
	.long L$set$37
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$38,LCFI19-LCFI18
	.long L$set$38
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$39,LCFI20-LCFI19
	.long L$set$39
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$40,LCFI21-LCFI20
	.long L$set$40
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$41,LCFI22-LCFI21
	.long L$set$41
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$42,LCFI23-LCFI22
	.long L$set$42
	.byte	0xe
	.byte	0x50
	.byte	0x4
	.set L$set$43,LCFI24-LCFI23
	.long L$set$43
	.byte	0xa
	.byte	0xe
	.byte	0x38
	.byte	0x4
	.set L$set$44,LCFI25-LCFI24
	.long L$set$44
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$45,LCFI26-LCFI25
	.long L$set$45
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$46,LCFI27-LCFI26
	.long L$set$46
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$47,LCFI28-LCFI27
	.long L$set$47
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$48,LCFI29-LCFI28
	.long L$set$48
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$49,LCFI30-LCFI29
	.long L$set$49
	.byte	0xe
	.byte	0x8
	.byte	0x4
	.set L$set$50,LCFI31-LCFI30
	.long L$set$50
	.byte	0xb
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$51,LEFDE19-LASFDE19
	.long L$set$51
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB5445-.
	.set L$set$52,LFE5445-LFB5445
	.quad L$set$52
	.byte	0
	.byte	0x4
	.set L$set$53,LCFI32-LFB5445
	.long L$set$53
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$54,LCFI33-LCFI32
	.long L$set$54
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$55,LEFDE21-LASFDE21
	.long L$set$55
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB5838-.
	.set L$set$56,LFE5838-LFB5838
	.quad L$set$56
	.byte	0
	.byte	0x4
	.set L$set$57,LCFI34-LFB5838
	.long L$set$57
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$58,LCFI35-LCFI34
	.long L$set$58
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE21:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_accTest.cpp
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
