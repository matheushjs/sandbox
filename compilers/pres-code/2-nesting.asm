	section .data
str: db `%d\n`, 0

	section .text
	align 4
	global main
	extern printf

func1:
	push rbp
	mov rbp, rsp

	mov rax, [rbp+16] ; pega o access link
	mov rbx, [rax-8]  ; acessa 'n'

	mov rdi, str
	mov rsi, rbx
	call printf

	mov rsp, rbp
	pop rbp
	ret

func2:
	push rbp
	mov rbp, rsp

	sub rsp, 8 ; alinhamento da stack

	mov rax, [rbp+16] ; pega o access link
	push rax          ; passa access link como argumento
	call func1

	mov rsp, rbp
	pop rbp
	ret

main:
	push rbp
	mov rbp, rsp

	sub rsp, 8
	mov QWORD [rsp], 10 ; declara e inicializa n

	push rbp ; passa access link como argumento
	call func2

	mov rsp, rbp
	pop rbp
	ret
