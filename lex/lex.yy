%option noyywrap
%{
#include <stdio.h>
%}
num [0-9]
letter [a-zA-Z]
%%
{letter}({letter}|{num})* printf("Found: {%s}\n", yytext);
%%
int main(int argc, char *argv[]){
	yylex();
	return 0;
}
