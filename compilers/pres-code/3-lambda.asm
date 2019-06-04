	section .data
str: db `%d\n`, 0

	section .text
	align 4
	global main
	extern printf

func:
	push rbp
	mov rbp, rsp

	mov rax, [rbp+16] ; ip
	mov rbx, [rbp+24] ; ep

	sub rsp, 8 ; alinha stack
	push rbx   ; ep
	call rax   ; ip

	mov rsp, rbp
	pop rbp
	ret

lambda:
	push rbp
	mov rbp, rsp

	mov rax, [rbp+16] ; pega 'ep'
	mov rax, [rax-16]  ; pega variavel 'n'

	mov rdi, str
	mov rsi, rax
	call printf

	mov rsp, rbp
	pop rbp
	ret

main:
	push rbp
	mov rbp, rsp
	sub rsp, 8 ; alinha a stack

	; declara e inicializa 'n'
	sub rsp, 8
	mov QWORD [rsp], 5

	; passa lambda como argumento para 'func'
	push rbp          ; ep
	mov rax, lambda
	push rax          ; ip
	call func

	mov rsp, rbp
	pop rbp
	ret
